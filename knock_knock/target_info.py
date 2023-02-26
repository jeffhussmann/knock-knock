import copy
import functools
import logging
import operator
import textwrap
from pathlib import Path
from collections import defaultdict

import yaml
import pysam
import numpy as np

import Bio.SeqIO
import Bio.SeqUtils

import hits.visualize
from hits import fasta, gff, utilities, mapping_tools, interval, sam, sw, genomes

import knock_knock.pegRNAs
import knock_knock.integrases

memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

class Effector():
    def __init__(self, name, PAM_pattern, PAM_side, cut_after_offset):
        self.name = name
        self.PAM_pattern = PAM_pattern
        self.PAM_side = PAM_side
        # cut_after_offset is relative to the 5'-most nt of the PAM
        self.cut_after_offset = cut_after_offset

    def __repr__(self):
        return f"{type(self).__name__}('{self.name}', '{self.PAM_pattern}', {self.PAM_side}, {self.cut_after_offset})"

    def PAM_slice(self, protospacer_feature):
        before_slice = slice(protospacer_feature.start - len(self.PAM_pattern), protospacer_feature.start)
        after_slice = slice(protospacer_feature.end + 1, protospacer_feature.end + 1 + len(self.PAM_pattern))

        if (protospacer_feature.strand == '+' and self.PAM_side == 5) or (protospacer_feature.strand == '-' and self.PAM_side == 3):
            PAM_slice = before_slice
        else:
            PAM_slice = after_slice

        return PAM_slice

    def PAM_matches_pattern(self, protospacer_feature, target_sequence):
        PAM_seq = target_sequence[self.PAM_slice(protospacer_feature)].upper()
        if protospacer_feature.strand == '-':
            PAM_seq = utilities.reverse_complement(PAM_seq)

        pattern, *matches = Bio.SeqUtils.nt_search(PAM_seq, self.PAM_pattern) 

        return 0 in matches

    def cut_afters(self, protospacer_feature):
        ''' Returns a dictionary of {strand: position after which nick is made} '''

        if protospacer_feature.strand == '+':
            offset_strand_order = '+-'
        else:
            offset_strand_order = '-+'

        if len(set(self.cut_after_offset)) == 1:
            # Blunt DSB
            offsets = list(set(self.cut_after_offset))
            strands = ['both']
        else:
            offsets = [offset for offset in self.cut_after_offset if offset is not None]
            strands = [strand for strand, offset in zip(offset_strand_order, self.cut_after_offset) if offset is not None]

        cut_afters = {}
        PAM_slice = self.PAM_slice(protospacer_feature)

        for offset, strand in zip(offsets, strands):
            if protospacer_feature.strand == '+':
                PAM_5 = PAM_slice.start
                cut_after = PAM_5 + offset
            else:
                PAM_5 = PAM_slice.stop - 1
                # -1 extra because cut_after is on the other side of the cut
                cut_after = PAM_5 - offset - 1
            
            cut_afters[strand] = cut_after

        return cut_afters

effectors = {
    'SpCas9': Effector('SpCas9', 'NGG', 3, (-4, -4)),
    'SpCas9H840A': Effector('SpCas9H840A', 'NGG', 3, (-4, None)),
    'SpCas9N863A': Effector('SpCas9N863A', 'NGG', 3, (-4, None)),
    'SpCas9H840A_VRQR': Effector('SpCas9H840A_VRQR', 'NGA', 3, (-4, None)),
    'SaCas9': Effector('SaCas9', 'NNGRRT', 3, (-4, -4)),
    'SaCas9H840A': Effector('SaCas9H840A', 'NNGRRT', 3, (-4, None)),
    'Cpf1': Effector('Cpf1', 'TTTN', 5, (20, 25)),
    'AsCas12a': Effector('AsCas12a', 'TTTN', 5, (20, 25)),
}

# Hack because 'sgRNA_SaCas9H840A' is one character too long for genbank format.
effectors['SaCas9H840'] = effectors['SaCas9H840A']
effectors['SpCas9H840'] = effectors['SpCas9H840A']

class TargetInfo():
    def __init__(self, base_dir, name,
                 primer_names=None,
                 sgRNAs=None,
                 donor=None,
                 nonhomologous_donor=None,
                 sequencing_start_feature_name=None,
                 supplemental_indices=None,
                 gb_records=None,
                 infer_homology_arms=False,
                 min_relevant_length=None,
                 feature_to_replace=None,
                 manifest=None,
                 manual_sgRNA_components=None,
                 max_programmed_deletion_length=None,
                ):
        self.name = name

        self.base_dir = Path(base_dir)
        self.targets_dir = self.base_dir / 'targets'
        self.dir = self.targets_dir / name

        # If not None, feature_to_replace is a tuple (ref_name, feature_name, sequence)
        # for which the region of ref_name covered by feature_name
        # will be replaced with sequence (e.g. when sequence is
        # one of a library of elements that was cloned into that site).
        self.feature_to_replace = feature_to_replace

        self.fns = {
            'sgRNAs': self.dir / 'sgRNAs.csv',

            'protospacer_fasta': self.dir / 'protospacers.fasta',
            'protospacer_STAR_prefix_template': self.dir / 'protospacers_{}.',
            'protospacer_bam_template': self.dir / 'protospacers_{}.bam',
        }
        
        if manifest is None:
            manifest_fn = self.dir / 'manifest.yaml'
            manifest = yaml.safe_load(manifest_fn.read_text())

        self.manifest = manifest

        self.target = self.manifest['target']

        self.sources = [s[:-len('.gb')] if s.endswith('.gb') else s for s in self.manifest['sources']]
        self.gb_records = gb_records
        self.manual_sgRNA_components = manual_sgRNA_components

        def populate_attribute(attribute_name, value, force_list=False, default_value=None):
            if value is None:
                value = self.manifest.get(attribute_name, default_value)

            if force_list:
                if isinstance(value, str):
                    if value == '':
                        value = []
                    else:
                        value = value.split(';')

            setattr(self, attribute_name, value)

        populate_attribute('sgRNAs', sgRNAs, force_list=True)
        populate_attribute('primer_names', primer_names, force_list=True, default_value=['forward_primer', 'reverse_primer'])
        populate_attribute('donor', donor)
        populate_attribute('nonhomologous_donor', nonhomologous_donor)
        populate_attribute('sequencing_start_feature_name', sequencing_start_feature_name)

        self.donor_type = self.manifest.get('donor_type')
        self.donor_specific = self.manifest.get('donor_specific', 'GFP11') 
        self.default_HAs = self.manifest.get('default_HAs')
        self.manual_features_to_show = self.manifest.get('features_to_show')
        self.genome_source = self.manifest.get('genome_source')

        if supplemental_indices is None:
            supplemental_indices = {}
        self.supplemental_indices = supplemental_indices

        self.infer_homology_arms = infer_homology_arms

        self.max_programmed_deletion_length = max_programmed_deletion_length

        self.min_relevant_length = min_relevant_length

    def __repr__(self):
        if len(self.pegRNA_names) > 0:
            representation = f'''\
                TargetInfo:
                    name = {self.name}
                    base_dir = {self.base_dir}
                    target = {self.target}
                    pegRNAs = [{','.join(self.pegRNA_names)}]
                    sgRNAs = [{','.join(self.sgRNA_names) if self.sgRNA_names is not None else ''}]
            '''
        else:
            representation = f'''\
                TargetInfo:
                    name = {self.name}
                    base_dir = {self.base_dir}
                    target = {self.target}
                    sgRNAs = [{','.join(self.sgRNA_names)}]
                    donor = {self.donor}
            '''
        return textwrap.dedent(representation)

    @memoized_property
    def header(self):
        ref_seqs = sorted(self.reference_sequences.items())
        names = [name for name, seq in ref_seqs]
        lengths = [len(seq) for name, seq in ref_seqs]

        header = pysam.AlignmentHeader.from_references(names, lengths)

        return header

    @memoized_property
    def primary_protospacer(self):
        primary_protospacer = self.manifest.get('primary_protospacer')
        if primary_protospacer is None and len(self.protospacer_names) > 0:
            if len(self.pegRNA_names) > 0:
                primary_protospacer = knock_knock.pegRNAs.protospacer_name(self.pegRNA_names[0])
            else:
                primary_protospacer = self.protospacer_names[0]
        return primary_protospacer

    @memoized_property
    def protospacer_color(self):
        return self.features[self.target, self.primary_protospacer].attribute['color']

    @memoized_property
    def supplemental_headers(self):
        return {name: sam.header_from_STAR_index(d['STAR']) for name, d in self.supplemental_indices.items()}

    @memoized_property
    def genomic_region_fetchers(self):
        fetchers = {name: genomes.build_region_fetcher(fns['fasta']) for name, fns in self.supplemental_indices.items()}
        return fetchers

    @memoized_property
    def protospacer_names(self):
        ''' Names of all features representing protospacers at which cutting was expected to occur '''
        if self.sgRNA_components is None:
            fs = []
        else:
            fs = [knock_knock.pegRNAs.protospacer_name(gRNA_name) for gRNA_name in self.sgRNA_components]
        return fs

    @memoized_property
    def sgRNA_components(self):
        if self.sgRNAs is None:
            sgRNA_components = None
        else:
            if self.manual_sgRNA_components is None:
                all_components = knock_knock.pegRNAs.read_csv(self.fns['sgRNAs'])
            else:
                all_components = self.manual_sgRNA_components

            sgRNA_components = {name: all_components[name] for name in self.sgRNAs}

        return sgRNA_components

    @memoized_property
    def pegRNA_names(self):
        ''' pegRNAs are sgRNAs that have an extension. '''
        if self.sgRNA_components is None:
            names = []
        else:
            names = [n for n, cs in self.sgRNA_components.items() if len(cs['extension']) > 1]

        return names

    @memoized_property
    def sgRNA_names(self):
        ''' For historical reasons, sgRNA_names are sgRNAs that aren't pegRNAs. '''
        if self.sgRNA_components is None:
            names = []
        else:
            names = [n for n, cs in self.sgRNA_components.items() if len(cs['extension']) == 0]

        return names
    
    @memoized_property
    def fasta_records_and_gff_features(self):
        ''' If self.gb_records is set, these override files. '''
        fasta_fns = [self.dir / (source + '.fasta') for source in self.sources]
        fasta_fns = [fn for fn in fasta_fns if fn.exists()]
        
        fasta_records = []
        all_gff_features = []
            
        if self.gb_records is None:
            gb_fns = [self.dir / (source + '.gb') for source in self.sources]

            for fn in gb_fns:
                if not fn.exists():
                    logging.warning(f'{self.name}: {fn} does not exist')

            gb_fns = [fn for fn in gb_fns if fn.exists()]

            self.gb_records = []
            for gb_fn in gb_fns:
                for record in Bio.SeqIO.parse(gb_fn, 'genbank'):
                    self.gb_records.append(record)

        for gb_record in self.gb_records:
            fasta_record, gff_features = parse_benchling_genbank(gb_record)

            if self.feature_to_replace is not None:
                ref_name, feature_name, new_sequence = self.feature_to_replace
                if ref_name == fasta_record.name:

                    # Find the feature start and end.
                    feature_matches = [f for f in gff_features if f.attribute.get('ID') == feature_name]
                    if len(feature_matches) != 1:
                        raise ValueError(f'Expected 1 feature named "{feature_name}", found {len(feature_matches)}')

                    feature_to_replace = feature_matches[0]
                    existing_start = feature_to_replace.start
                    existing_end = feature_to_replace.end

                    # Splice in the new sequence.
                    before = fasta_record.seq[:existing_start]
                    after = fasta_record.seq[existing_end + 1:]

                    fasta_record.seq = before + new_sequence + after

                    change_in_length = len(new_sequence) - len(feature_to_replace)

                    # Find any features whose end is past the start of the replaced feature,
                    # and shift any of its boundaries that are past the start that by the change in length.

                    for feature in gff_features:
                        if feature.end > existing_start:
                            feature.end += change_in_length
                            if feature.start > existing_start:
                                feature.start += change_in_length

            fasta_records.append(fasta_record)
            all_gff_features.extend(gff_features)

        for fasta_fn in fasta_fns:
            for fasta_record in fasta.records(fasta_fn):
                fasta_records.append(fasta_record)

        return fasta_records, all_gff_features

    def make_protospacer_fastas(self):
        ''' Protospacer locations will be used to determine if genomic alignments
        are in the vicinity of targeted sites, so want to include all possible
        pegRNA protospacers as well as explicitly annotated ones. '''

        all_components = knock_knock.pegRNAs.read_csv(self.fns['sgRNAs'])

        with open(self.fns['protospacer_fasta'], 'w') as fh:
            protospacers = {name: cs['protospacer'] for name, cs in all_components.items()}
            for name, seq in sorted(protospacers.items()):
                ps_name = knock_knock.pegRNAs.protospacer_name(name)
                record = fasta.Read(ps_name, seq)
                fh.write(str(record))

    def map_protospacers(self, index_name):
        indices = locate_supplemental_indices(self.base_dir)

        index_dir = indices[index_name]['STAR']
        output_prefix = str(self.fns['protospacer_STAR_prefix_template']).format(index_name)
        bam_fn = str(self.fns['protospacer_bam_template']).format(index_name)
        mapping_tools.map_STAR(self.fns['protospacer_fasta'], index_dir, output_prefix, mode='guide_alignment', bam_fn=bam_fn)

        mapping_tools.clean_up_STAR_output(output_prefix)

    def mapped_protospacer_locations(self, sgRNA_name, index_name):
        bam_fn = str(self.fns['protospacer_bam_template']).format(index_name)
        locations = set()

        if Path(bam_fn).exists():
            with pysam.AlignmentFile(bam_fn) as bam_fh:
                for al in bam_fh:
                    if al.query_name == sgRNA_name:
                        locations.add((al.reference_name, al.reference_start, sam.get_strand(al)))

        return locations
    
    @memoized_with_args
    def mapped_protospacer_location(self, index_name):
        protospacer = self.primary_protospacer

        locations = self.mapped_protospacer_locations(protospacer, index_name)
        
        if len(locations) != 1:
            location = None
        else:
            location = locations.pop()

        return location

    @memoized_property
    def reference_name_in_genome_source(self):
        if self.genome_source is None:
            return None
        else:
            location = self.mapped_protospacer_location(self.genome_source)
            if location is None:
                return None
            else:
                rname, pos, strand = location
                return f'{self.genome_source}_{rname}'

    def convert_genomic_alignment_to_target_coordinates(self, al):
        try:
            organism, al = self.remove_organism_from_alignment(al)
        except:
            return False

        protospacer_rname, protospacer_start, protospacer_strand = self.mapped_protospacer_location(organism)

        if al.reference_name != protospacer_rname:
            return False
        else:
            if self.protospacer_feature.strand == protospacer_strand:
                # The target is in the same orientation as the reference genome,
                # so coordinate transform is a simple offset.
                offset = self.protospacer_feature.start - protospacer_start
                converted = {
                    'start': al.reference_start + offset,
                    'end': al.reference_end + offset,
                    'strand': sam.get_strand(al),
                }
            else:
                # The target is in the opposite orientation as the reference genome,
                # so coordinate transform is more complex.
                if self.protospacer_feature.strand == '+':
                    raise NotImplementedError
                else:
                    offset = self.protospacer_feature.end + protospacer_start
                    converted = {
                        'start': offset - al.reference_end,
                        'end': offset - al.reference_start,
                        'strand': sam.get_opposite_strand(al),
                    }

            return converted

    @memoized_property
    def features(self):
        fasta_records, gff_features = self.fasta_records_and_gff_features

        features = {(f.seqname, f.attribute['ID']): f for f in gff_features if 'ID' in f.attribute}

        if len(self.pegRNA_names) > 0:
            for pegRNA_name in self.pegRNA_names:
                pegRNA_features, target_features = knock_knock.pegRNAs.infer_features(pegRNA_name,
                                                                                      self.sgRNA_components[pegRNA_name],
                                                                                      self.target,
                                                                                      self.target_sequence,
                                                                                     )
                features.update({**pegRNA_features, **target_features})

            if len(self.pegRNA_names) == 1:
                edit_features, _, _ = knock_knock.pegRNAs.infer_edit_features(self.pegRNA_names[0],
                                                                              self.target,
                                                                              features,
                                                                              self.reference_sequences,
                                                                              max_deletion_length=self.max_programmed_deletion_length,
                                                                             )

                features.update(edit_features)

            elif len(self.pegRNA_names) == 2:
                results = knock_knock.pegRNAs.infer_twin_pegRNA_features(self.pegRNA_names,
                                                                         self.target,
                                                                         features,
                                                                         self.reference_sequences,
                                                                        )

                features.update(results['new_features'])
        
        features.update(self.integrase_sites)

        for name, feature in {**self.protospacer_features, **self.PAM_features}.items():
            features[self.target, name] = feature

        features.update(self.inferred_HA_features)

        # Override colors of protospacers in pooled screening vector
        # to ensure consistency.

        override_colors = {
            ('pooled_vector', 'sgRNA-5'): 'tab:green',
            ('pAX198', 'SpCas9 target 1'): 'tab:green',

            ('pooled_vector', 'sgRNA-3'): 'tab:orange',
            ('pAX198', 'SpCas9 target 2'): 'tab:orange',

            ('pooled_vector', 'sgRNA-2'): 'tab:blue',
            ('pAX198', 'SpCas9 target 3'): 'tab:blue',

            ('pooled_vector', 'sgRNA-7'): 'tab:red',
            ('pAX198', 'SpCas9 target 4'): 'tab:red',

            ('pooled_vector', 'sgRNA-Cpf1'): 'tab:brown',
            ('pAX198', 'AsCas12a target'): 'tab:brown',
        }

        for name, color in override_colors.items():
            override_colors[name] = hits.visualize.apply_alpha(color, 0.5)

        for name, feature in features.items():
            if name in override_colors:
                feature.attribute['color'] = override_colors[name]

        return features

    @memoized_property
    def integrase_sites(self):
        return knock_knock.integrases.identify_split_recognition_sequences(self.reference_sequences)

    @memoized_property
    def features_to_show(self):
        if self.manual_features_to_show is not None:
            features_to_show = {tuple(f) for f in self.manual_features_to_show}
        else:
            features_to_show = set()

            if self.donor is not None:
                features_to_show.update({
                (self.donor, 'GFP'),
                (self.donor, 'GFP11'),
                (self.donor, 'PPX'),
                (self.donor, 'donor_specific'),
                (self.donor, 'PCR_adapter_1'),
                (self.donor, 'PCR_adapter_2'),
            })

            for protospacer_name in self.protospacer_features:
                features_to_show.add((self.target, protospacer_name))

            for side in [5, 3]:
                primer = (self.target, self.primers_by_side_of_target[side].attribute['ID'])
                features_to_show.add(primer)

                if self.homology_arms is not None and len(self.pegRNA_names) == 0:
                    target_HA = (self.target, self.homology_arms[side]['target'].attribute['ID'])
                    features_to_show.add(target_HA)

                    if self.has_shared_homology_arms:
                        donor_HA = (self.donor, self.homology_arms[side]['donor'].attribute['ID'])
                        features_to_show.add(donor_HA)

            features_to_show.update(set(self.PAM_features))

        if len(self.pegRNA_names) > 0:
            if len(self.pegRNA_programmed_insertion_features) > 0:
                for insertion in self.pegRNA_programmed_insertion_features:
                    features_to_show.add((insertion.seqname, insertion.attribute['ID']))

                for pegRNA_name in self.pegRNA_names:
                    for name in ['protospacer', 'scaffold', 'PBS', f'HA_RT_{pegRNA_name}']:
                        features_to_show.add((pegRNA_name, name))

            else:
                if len(self.pegRNA_programmed_deletions) > 0:
                    for deletion in self.pegRNA_programmed_deletions:
                        features_to_show.add((deletion.seqname, deletion.attribute['ID']))

                for pegRNA_name in self.pegRNA_names:
                    features_to_show.update({(pegRNA_name, name) for name in ['protospacer', 'scaffold', 'PBS', 'RTT']})

            for pegRNA_name in self.pegRNA_names:
                ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
                features_to_show.add((self.target, ps_name))

        return features_to_show

    @memoized_property
    def sequencing_start(self):
        feature_name = self.sequencing_start_feature_name
        if feature_name is None:
            feature_name = 'sequencing_start'
        return self.features.get((self.target, feature_name))

    @memoized_property
    def reference_sequences(self):
        fasta_records, gff_features = self.fasta_records_and_gff_features

        seqs = {record.name: record.seq for record in fasta_records}

        if len(self.pegRNA_names) > 0:
            for pegRNA_name in self.pegRNA_names:
                seqs[pegRNA_name] = self.sgRNA_components[pegRNA_name]['full_sequence']

        return seqs

    @memoized_property
    def reference_sequence_bytes(self):
        ''' sequences as bytes for use in alignment code that requires this '''
        seqs = {name: seq.encode() for name, seq in self.reference_sequences.items()}
        return seqs

    @property
    def target_sequence(self):
        return self.reference_sequences[self.target]
    
    @memoized_property
    def seed_and_extender(self):
        def fake_extender(*args):
            return [] 

        extenders = {
            'target': sw.SeedAndExtender(self.reference_sequence_bytes[self.target], 20, self.header, self.target).seed_and_extend,
        }
        if self.donor is not None:
            extenders['donor'] = sw.SeedAndExtender(self.reference_sequence_bytes[self.donor], 20, self.header, self.donor).seed_and_extend
        else:
            # If there isn't a donor, always return no alignments.
            extenders['donor'] = fake_extender

        return extenders

    @property
    def donor_sequence(self):
        if self.donor is None:
            return None
        else:
            return self.reference_sequences[self.donor]

    @memoized_property
    def protospacer_features(self):
        features = {}
        if self.sgRNA_components is not None:
            for name, components in self.sgRNA_components.items():
                feature = knock_knock.pegRNAs.identify_protospacer_in_target(self.target_sequence,
                                                                             components['protospacer'],
                                                                             components['effector'],
                                                                            )
                feature.seqname = self.target
                ps_name = knock_knock.pegRNAs.protospacer_name(name)
                feature.attribute['ID'] = ps_name
                features[ps_name] = feature
                                                                        
        return features
    
    @memoized_property
    def protospacer_feature(self):
        if self.primary_protospacer is None:
            return None
        else:
            return self.protospacer_features[self.primary_protospacer]

    @memoized_property
    def protospacer_sequence(self):
        if self.protospacer_feature is None:
            return None
        else:
            return self.feature_sequence(self.target, self.primary_protospacer)

    def feature_sequence(self, seq_name, feature_name):
        feature = self.features[seq_name, feature_name]
        return feature.sequence(self.reference_sequences)

    @memoized_property
    def PAM_slices(self):
        PAM_slices = {}

        for name, protospacer in self.protospacer_features.items():
            effector = effectors[protospacer.attribute['effector']]

            PAM_slices[name] = effector.PAM_slice(protospacer)

            if not effector.PAM_matches_pattern(protospacer, self.target_sequence):
                print(f'Warning: {name} PAM doesn\'t match {effector.PAM_pattern}')

        return PAM_slices

    @memoized_property
    def PAM_features(self):
        PAM_features = {}

        for name, sl in self.PAM_slices.items():
            protospacer = self.protospacer_features[name]

            PAM_name = f'{name}_PAM'
            PAM_feature = gff.Feature.from_fields(seqname=self.target,
                                                  feature='PAM',
                                                  start=sl.start,
                                                  end=sl.stop - 1,
                                                  strand=protospacer.strand,
                                                 )

            protospacer_color = protospacer.attribute['color']
            PAM_color = hits.visualize.scale_darkness(protospacer_color, 1.3)

            PAM_feature.attribute = {
                'ID': PAM_name,
                'color': PAM_color,
                'short_name': 'PAM',
            }

            PAM_features[PAM_name] = PAM_feature

        return PAM_features
                                    
    @memoized_property
    def PAM_slice(self):
        return self.PAM_slices[self.primary_protospacer]

    @memoized_property
    def PAM_color(self):
        return self.PAM_features[f'{self.primary_protospacer}_PAM'].attribute['color']

    @memoized_property
    def effector(self):
        return effectors[self.protospacer_feature.attribute['effector']]

    @memoized_property
    def anchor(self):
        feature = self.features.get((self.target, 'anchor'))
        if feature is not None:
            return feature.start
        else:
            return 0

    @memoized_property
    def cut_afters(self):
        cut_afters = {}
        for name, protospacer in self.protospacer_features.items():
            effector = effectors[protospacer.attribute['effector']]

            for strand, cut_after in effector.cut_afters(protospacer).items():
                cut_afters[f'{name}_{strand}'] = cut_after

        return cut_afters
    
    @memoized_property
    def cut_after(self):
        ''' when processing assumes there will be only one cut, use this '''
        primary_cut_afters = []
        for name, cut_after in self.cut_afters.items():
            protospacer_name = name.rsplit('_', 1)[0]
            if protospacer_name == self.primary_protospacer:
                primary_cut_afters.append(cut_after)

        if len(primary_cut_afters) == 0:
            return None
        else:
            return min(primary_cut_afters)

    @memoized_with_args
    def around_cuts(self, each_side):
        if len(self.cut_afters) == 0:
            around_cuts = interval.Interval.empty()
        else:
            intervals = [interval.Interval(cut_after - each_side + 1, cut_after + each_side) for cut_after in self.cut_afters.values()]
            around_cuts = functools.reduce(operator.or_, intervals) 

        return around_cuts

    @memoized_with_args
    def not_around_cuts(self, each_side):
        whole_target = interval.Interval(0, len(self.target_sequence))
        return whole_target - self.around_cuts(each_side)

    def around_or_between_cuts(self, each_side):
        left = min(self.cut_afters.values()) - each_side
        right = max(self.cut_afters.values()) + each_side
        return interval.Interval(left, right)

    @memoized_property
    def overlaps_cut(self):
        ''' only really makes sense if only one cut or Cpf1 '''
        cut_after_ps = self.cut_afters.values()
        cut_interval = interval.Interval(min(cut_after_ps), max(cut_after_ps))

        def overlaps_cut(al):
            return bool(sam.reference_interval(al) & cut_interval)

        return overlaps_cut
        
    @memoized_property
    def most_extreme_primer_names(self):
        # Default to the primers farthest to the left and right on the target.
        all_primer_names = {name: feature for (_, name), feature in self.features.items()
                            if name.startswith('forward_primer') or name.startswith('reverse_primer')
                            }
        left_most = min(all_primer_names, key=lambda n: all_primer_names[n].start)
        right_most = max(all_primer_names, key=lambda n: all_primer_names[n].end)

        primer_names = [left_most, right_most]

        return primer_names

    @memoized_property
    def primers(self):
        primer_names = self.primer_names
        if primer_names is None:
            primer_names = self.most_extreme_primer_names

        primers = {name: self.features[self.target, name] for name in primer_names}

        return primers

    @memoized_property
    def primers_by_PAM_side(self):
        primers = {}

        for primer_name, primer in self.primers.items():
            primer = self.features[self.target, primer_name]
            primer_interval = interval.Interval.from_feature(primer)
            
            for side, PAM_side_interval in self.PAM_side_intervals.items():
                if primer_interval & PAM_side_interval:
                    primers[side] = primer

        if len(primers) != 2:
            raise ValueError(f'not a primer on each side of cut: {primers}')

        return primers
    
    @memoized_property
    def primers_by_side_of_target(self):
        by_side = {}
        primers = [primer for primer in self.primers.values()]
        by_side[5], by_side[3] = sorted(primers, key=lambda p: p.start)
        return by_side

    @memoized_property
    def primers_by_side_of_read(self):
        if self.sequencing_direction == '+':
            by_side = {
                'left': self.primers_by_side_of_target[5],
                'right': self.primers_by_side_of_target[3],
            }
        elif self.sequencing_direction == '-':
            by_side = {
                'left': self.primers_by_side_of_target[3],
                'right': self.primers_by_side_of_target[5],
            }
        else:
            raise ValueError(self.sequencing_direction)

        return by_side

    @memoized_property
    def primers_by_strand(self):
        by_strand = {primer.strand: primer for primer in self.primers.values()}
        return by_strand

    @memoized_property
    def between_primers_interval(self):
        start = self.primers_by_side_of_target[5].end + 1
        end = self.primers_by_side_of_target[3].start - 1
        return interval.Interval(start, end)

    @memoized_property
    def sequencing_direction(self):
        if self.sequencing_start is None:
            return None
        else:
            return self.sequencing_start.strand

    @memoized_property
    def combined_primer_length(self):
        if len(self.primers) != 2:
            raise ValueError(self.primers)
        else:
            return sum(len(f) for name, f in self.primers.items())

    @memoized_with_args
    def ref_p_to_feature_offset(self, ref_name, feature_name):
        feature = self.features[ref_name, feature_name]
        
        if feature.strand == '+':
            ref_p_order = range(feature.start, feature.end + 1)
        elif feature.strand == '-':
            ref_p_order = range(feature.end, feature.start - 1, -1)
        else:
            raise ValueError('feature needs to be stranded')
            
        return {ref_p: offset for offset, ref_p in enumerate(ref_p_order)}

    @memoized_with_args
    def feature_offset_to_ref_p(self, ref_name, feature_name):
        return utilities.reverse_dictionary(self.ref_p_to_feature_offset(ref_name, feature_name))

    @memoized_property
    def target_side_intervals(self):
        ''' intervals of target 5' and 3' of cut '''
        return {
            5: interval.Interval(0, self.cut_after),
            3: interval.Interval(self.cut_after + 1, np.inf)
        }

    @memoized_property
    def PAM_side_intervals(self):
        return {self.target_side_to_PAM_side[side]: intvl for side, intvl in self.target_side_intervals.items()}
    
    @memoized_property
    def read_side_intervals(self):
        return {self.target_side_to_read_side[side]: intvl for side, intvl in self.target_side_intervals.items()}

    @memoized_property
    def target_side_to_PAM_side(self):
        if self.protospacer_feature is None:
            return None

        strand = self.protospacer_feature.strand
        PAM_side = self.effector.PAM_side

        if (strand == '+' and PAM_side == 3) or (strand == '-' and PAM_side == 5):
            target_to_PAM = {3: 'PAM-proximal', 5: 'PAM-distal'}
        elif (strand == '+' and PAM_side == 5) or (strand == '-' and PAM_side == 3):
            target_to_PAM = {5: 'PAM-proximal', 3: 'PAM-distal'}

        return target_to_PAM

    @memoized_property
    def target_side_to_read_side(self):
        if self.sequencing_start is None:
            return None
        
        read_start_interval = interval.Interval.from_feature(self.sequencing_start)
        
        if self.target_side_intervals[5] & read_start_interval:
            target_to_read = {5: 'left', 3: 'right'}
        elif self.target_side_intervals[3] & read_start_interval:
            target_to_read = {5: 'right', 3: 'left'}
        else:
            raise ValueError
            
        return target_to_read

    @memoized_property
    def read_side_to_target_side(self):
        return utilities.reverse_dictionary(self.target_side_to_read_side)

    @memoized_property
    def PAM_side_to_target_side(self):
        return utilities.reverse_dictionary(self.target_side_to_PAM_side)

    @memoized_property
    def read_side_to_PAM_side(self):
        read_to_PAM = {}

        for target_side, read_side in self.target_side_to_read_side.items():
            PAM_side = self.target_side_to_PAM_side[target_side]
            read_to_PAM[read_side] = PAM_side

        return read_to_PAM
    
    @memoized_property
    def PAM_side_to_read_side(self):
        return utilities.reverse_dictionary(self.read_side_to_PAM_side)

    @memoized_property
    def homology_arms(self):
        ''' HAs keyed by:
                name,
                side of cut (PAM-proximal/PAM-distal),
                side of target (5/3),
                expected side of read (left/right),
        '''
        if self.donor is not None and len(self.pegRNA_names) > 0:
            # integrase
            donor = None
        elif self.donor is None and len(self.pegRNA_names) == 1:
            donor = self.pegRNA_names[0]
        else:
            donor = self.donor

        if donor is None:
            return None

        ref_seqs = {
            'target': self.reference_sequences[self.target],
            'donor': self.reference_sequences[donor],
         }

        # Load homology arms from gff features.

        if self.infer_homology_arms:
            donor_SNVs, HAs = self.inferred_donor_SNVs_and_HAs
            HAs = copy.deepcopy(HAs)
        else:
            HAs = defaultdict(dict)

            ref_name_to_source = {
                self.target: 'target',
                donor: 'donor',
            }

            for (ref_name, feature_name), feature in self.features.items():
                source = ref_name_to_source.get(ref_name)
                if source is not None:
                    if feature_name.startswith('HA_'):
                        HAs[feature_name][source] = feature

        if len(HAs) == 0:
            return None

        paired_HAs = {}

        if self.donor is None and len(self.pegRNA_names) == 0:
            donor = self.pegRNA_names[0]
        else:
            donor = self.donor

        # Check if every HA name that exists on both the target and donor has the same
        # sequence on each.
        for name in HAs:
            if 'target' not in HAs[name] or 'donor' not in HAs[name]:
                continue
            
            seqs = {}
            for source in ['target', 'donor']:
                ref_seq = ref_seqs[source]
                HA = HAs[name][source]
                HA_seq = ref_seq[HA.start:HA.end + 1]
                if HA.strand == '-':
                    HA_seq = utilities.reverse_complement(HA_seq)
                    
                seqs[source] = HA_seq
                
            if seqs['target'] != seqs['donor']:
                logging.warning(f'{name} not identical sequence on target and donor')

            paired_HAs[name] = HAs[name]

        # We expect two HAs to exists on both target and donor.
        if len(paired_HAs) != 2:
            # If there is only one set of homology arms to use, just use it.
            if len(HAs) == 2:
                for name in HAs:
                    paired_HAs[name] = HAs[name]
            else:
                # otherwise need a default set to be specified
                if self.default_HAs is None:
                    raise ValueError('need to specify default_HAs if no homologous donor')
                else:
                    for name in self.default_HAs:
                        paired_HAs[name] = HAs[name]

        # Remove any HAs not present in both target and donor.
        for name in sorted(HAs):
            if name not in paired_HAs:
                HAs.pop(name)

        by_target_side = {}
        by_target_side[5], by_target_side[3] = sorted(paired_HAs, key=lambda n: HAs[n]['target'].start)

        for target_side, name in by_target_side.items():
            if self.target_side_to_PAM_side is not None:
                PAM_side = self.target_side_to_PAM_side[target_side]
            else:
                PAM_side = None

            # If sequencing start isn't annotated, don't populate by read side.
            if self.target_side_to_read_side is not None:
                read_side = self.target_side_to_read_side[target_side]
            else:
                read_side = None

            for key in [target_side, PAM_side, read_side]:
                if key is not None:
                    if key in HAs:
                        raise ValueError(key)
                    HAs[key] = HAs[name]

        return HAs

    @memoized_property
    def has_shared_homology_arms(self):
        if self.homology_arms is None:
            return False
        else:
            has_shared_arms = False

            for name, d in self.homology_arms.items():
                if 'target' in d and 'donor' in d:
                    has_shared_arms = True

            return has_shared_arms

    @memoized_property
    def HA_ref_p_to_offset(self):
        ''' register which positions in target and donor sequences correspond
        to the same offset in each homology arm
        '''
        if self.homology_arms is None:
            return None

        ref_p_to_offset = {}

        for PAM_side in self.homology_arms:
            for name in self.homology_arms[PAM_side]:
                feature = self.homology_arms[PAM_side][name]
            
                if feature.strand == '+':
                    order = range(feature.start, feature.end + 1)
                else:
                    order = range(feature.end, feature.start - 1, -1)
                    
                ref_p_to_offset[name, PAM_side] = {ref_p: offset for offset, ref_p in enumerate(order)}  

        for name in ['target', 'donor']:
            for side in ['left', 'right']:
                ref_p_to_offset[name, side] = ref_p_to_offset[name, self.read_side_to_PAM_side[side]]

        return ref_p_to_offset

    @memoized_property
    def offset_to_HA_ref_ps(self):
        offset_to_HA_ref_ps = defaultdict(dict)
        for (name, HA_label), p_to_offset in self.HA_ref_p_to_offset.items():
            for p, offset in p_to_offset.items():
                offset_to_HA_ref_ps[offset][name, HA_label] = p
        return offset_to_HA_ref_ps

    @memoized_property
    def past_HA_in_sequencing_read_interval(self):
        right_HA = self.homology_arms['right']['target']

        if self.sequencing_start.strand == '+':
            target_past_HA_interval = interval.Interval(right_HA.end + 1, len(self.target_sequence))
        else:
            target_past_HA_interval = interval.Interval(0, right_HA.start - 1)

        return target_past_HA_interval

    @memoized_property
    def donor_HA_intervals(self):
        ''' DisjointIntervals of the regions of the donor covered by homology arms '''
        return interval.Interval.from_feature(self.homology_arms[5]['donor']) | interval.Interval.from_feature(self.homology_arms[3]['donor'])

    @memoized_property
    def donor_specific_intervals(self):
        ''' DisjointIntervals of the regions of the donor NOT covered by homology arms '''
        return interval.Interval(0, len(self.donor_sequence) - 1) - self.donor_HA_intervals

    @memoized_property
    def amplicon_interval(self):
        primers = self.primers_by_side_of_target
        return interval.Interval(primers[5].start, primers[3].end)

    @memoized_property
    def amplicon_length(self):
        return len(self.amplicon_interval)

    @memoized_property
    def wild_type_amplicon_sequence(self):
        seq = self.target_sequence[self.amplicon_interval.start:self.amplicon_interval.end + 1]
        if self.sequencing_direction == '-':
            seq = utilities.reverse_complement(seq)

        return seq

    @memoized_property
    def clean_HDR_length(self):
        if self.has_shared_homology_arms:
            def gap_between_HAs(HA_1, HA_2):
                HA_1, HA_2 = sorted([HA_1, HA_2], key=lambda f: f.start)
                return HA_2.start - HA_1.end - 1
            HAs = self.homology_arms
            added_by_donor = gap_between_HAs(HAs[5]['donor'], HAs[3]['donor'])
            removed_by_HAs = gap_between_HAs(HAs[5]['target'], HAs[3]['target'])
            return self.amplicon_length + added_by_donor - removed_by_HAs
        else:
            return None

    @memoized_property
    def fingerprints(self):
        fps = {
            self.target: [],
            self.donor: [],
        }

        for seq_type, name in [('donor', self.donor), ('target', self.target)]:
            fps[name] = [((v['strand'], v['position']), v['base']) for _, v in sorted(self.donor_SNVs[seq_type].items())]

        return fps

    @memoized_property
    def SNV_names(self):
        if self.donor is None and len(self.pegRNA_names) == 1:
            SNV_source = self.pegRNA_names[0]
        else:
            SNV_source = self.donor

        return sorted([name for seq_name, name in self.features if seq_name == SNV_source and name.startswith('SNV')])

    @memoized_property
    def donor_SNVs_manual(self):
        SNVs = {
            'target': {},
            'donor': {},
        }

        if self.donor is None and len(self.pegRNA_names) == 1:
            donor = self.pegRNA_names[0]
        else:
            donor = self.donor

        if donor is None:
            return SNVs

        for key, seq_name in [('target', self.target), ('donor', donor)]:
            seq = self.reference_sequences[seq_name]
            for name in self.SNV_names:
                feature = self.features[seq_name, name]
                strand = feature.strand
                position = feature.start
                b = seq[position:position + 1]
                if strand == '-':
                    b = utilities.reverse_complement(b)

                SNVs[key][name] = {
                    'position': position,
                    'strand': strand,
                    'base': b,
                }
            
        return SNVs

    @memoized_property
    def best_donor_target_alignment(self):
        donor_bytes = {
            False: self.reference_sequence_bytes[self.donor],
        }
        donor_bytes[True] = utilities.reverse_complement(donor_bytes[False])
        
        target_bytes = self.reference_sequence_bytes[self.target]
        
        candidates = []
        for is_reverse_complement in [False, True]:
            mismatches = sw.mismatches_at_offset(donor_bytes[is_reverse_complement], target_bytes)
            candidates.extend([(m, i, is_reverse_complement) for i, m in enumerate(mismatches)])
            
        num_mismatches, offset, is_reverse_complement = min(candidates) 

        return num_mismatches, offset, is_reverse_complement

    @memoized_property
    def inferred_donor_SNVs_and_HAs(self):
        if self.donor is None or not self.infer_homology_arms:
            return None

        SNVs = {
            'target': {},
            'donor': {},
        }
        
        donor_bytes = {
            False: self.reference_sequence_bytes[self.donor],
        }
        donor_bytes[True] = utilities.reverse_complement(donor_bytes[False])
        
        target_bytes = self.reference_sequence_bytes[self.target]
            
        num_mismatches, offset, is_reverse_complement = self.best_donor_target_alignment

        if num_mismatches > 8:
            return SNVs, {}
        
        donor_substring = donor_bytes[is_reverse_complement].decode()
        target_substring = target_bytes[offset:offset + len(donor_bytes[is_reverse_complement])].decode()
        
        for i, (d, t) in enumerate(zip(donor_substring, target_substring)):
            if d != t:
                name = f'SNV_{offset + i:05d}_{t}-{d}'
                
                SNVs['target'][name] = {
                    'position': offset + i,
                    'strand': '+',
                    'base': t,
                }
                
                if is_reverse_complement:
                    donor_position = len(donor_substring) - 1 - i
                    donor_strand = '-'
                else:
                    donor_position = i
                    donor_strand = '+'
                    
                SNVs['donor'][name] = {
                    'position': donor_position,
                    'strand': donor_strand,
                    'base': d,
                }

        donor_ps = [d['position'] for d in SNVs['donor'].values()]
        target_ps = [d['position'] for d in SNVs['target'].values()]
        
        names = {
            1: f'HA_{self.donor}_1',
            2: f'HA_{self.donor}_2',
        }
        
        colors = {
            1: '%23b7e6d7',
            2: '%2385dae9',
        }
        
        lengths = {
            1: min(donor_ps),
            2: len(donor_substring) - 1 - max(donor_ps),
        }
        
        bounds = {
            1: (0, min(donor_ps) - 1),
            2: (max(donor_ps) + 1, len(donor_substring) - 1),
        }
            
        if is_reverse_complement:
            left_num = 2
            right_num = 1
            
            donor_strand = '-'
            
        else:
            left_num = 1
            right_num = 2
            
            donor_strand = '+'

        target_strand = '+'
            
        def make_feature(seqname, start, end, name, strand, color):
            return gff.Feature.from_fields(seqname=seqname,
                                           feature='misc_feature',
                                           start=start,
                                           end=end,
                                           strand=strand,
                                           attribute_string=f'ID={name};color={color}',
                                          )
        
        HAs = {
            names[left_num]: {
                'donor': make_feature(self.donor, bounds[left_num][0], bounds[left_num][1], names[left_num], donor_strand, colors[left_num]),
                'target': make_feature(self.target, min(target_ps) - lengths[left_num], min(target_ps) - 1, names[left_num], target_strand, colors[left_num]),
            },
            names[right_num]: {
                'donor': make_feature(self.donor, bounds[right_num][0], bounds[right_num][1], names[right_num], donor_strand, colors[right_num]),
                'target': make_feature(self.target, max(target_ps) + 1, max(target_ps) + lengths[right_num], names[right_num], target_strand, colors[right_num]),
            },
        }
                
        return SNVs, HAs

    @memoized_property
    def inferred_HA_features(self):
        features = {}
        if self.donor is None or not self.infer_homology_arms:
            pass
        else:
            donor_SNVs, HAs = self.inferred_donor_SNVs_and_HAs
            
            for HA_name in HAs:
                features[self.target, HA_name] = HAs[HA_name]['target']
                features[self.donor, HA_name] = HAs[HA_name]['donor']

        return features
    
    @memoized_property
    def donor_SNVs(self):
        if self.donor is None:
            donor_SNVs = self.donor_SNVs_manual
        elif self.infer_homology_arms:
            donor_SNVs, HAs = self.inferred_donor_SNVs_and_HAs
        else:
            donor_SNVs = self.donor_SNVs_manual

        return donor_SNVs
    
    @memoized_property
    def simple_donor_SNVs(self):
        ''' {ref_name, position: base identity in donor if on forward strand}}'''
        if self.donor_SNVs is None:
            return None

        if self.donor is None and len(self.pegRNA_names) == 1:
            donor = self.pegRNA_names[0]
        else:
            donor = self.donor

        SNVs = {}

        for SNV_name, donor_SNV_details in self.donor_SNVs['donor'].items():
            target_SNV_details = self.donor_SNVs['target'][SNV_name]

            donor_base = donor_SNV_details['base']
            if donor_SNV_details['strand'] == '-':
                # We want the forward strand base.
                donor_base = utilities.reverse_complement(donor_base)

            SNVs[donor, donor_SNV_details['position']] = donor_base

            # Confusing: we want the base that would be read on the forward
            # strand of target if it is actually the donor SNV.
            if target_SNV_details['strand'] != donor_SNV_details['strand']:
                possibly_flipped_donor_base = utilities.reverse_complement(donor_base)
            else:
                possibly_flipped_donor_base = donor_base

            SNVs[self.target, target_SNV_details['position']] = possibly_flipped_donor_base
        
        return SNVs
    
    @memoized_property
    def donor_deletions(self):
        donor_deletions = []
        for (seq_name, name), feature in self.features.items():
            if seq_name == self.target and feature.feature.startswith('donor_deletion'):
                deletion_donor_name = feature.feature[len('donor_deletion_'):]
                if deletion_donor_name == self.donor:
                    deletion = DegenerateDeletion([feature.start], len(feature))
                    deletion = self.expand_degenerate_indel(deletion)
                    donor_deletions.append(deletion)

        return donor_deletions

    @memoized_property
    def donor_insertions(self):
        donor_insertions = []
        for (seq_name, name), feature in self.features.items():
            if seq_name == self.donor and feature.feature == 'donor_insertion':
                donor_insertions.append(feature)

        return donor_insertions

    @memoized_property
    def wild_type_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.target]])
    
    @memoized_property
    def donor_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.donor]])

    @memoized_property
    def degenerate_indels(self):
        degenerate_dels = defaultdict(list)

        # Indels that aren't contained in the amplicon can't be observed,
        # so there is no need to register them.

        amplicon_start = self.primers_by_side_of_target[5].start
        amplicon_end = self.primers_by_side_of_target[3].end

        possible_starts = range(amplicon_start, amplicon_end)

        singleton_to_full = {}

        for starts_at in possible_starts:
            before = self.target_sequence[:starts_at]
            for length in range(1, amplicon_end - starts_at + 1):
                after = self.target_sequence[starts_at + length:]
                result = before + after
                deletion = DegenerateDeletion([starts_at], length)
                degenerate_dels[result].append(deletion)

        classes = degenerate_dels.values()
        for degenerate_class in degenerate_dels.values():
            if len(degenerate_class) > 1:
                collapsed = DegenerateDeletion.collapse(degenerate_class)
                for deletion in degenerate_class:
                    singleton_to_full[deletion] = collapsed

        degenerate_inss = defaultdict(list)

        mers = {length: list(utilities.mers(length)) for length in range(1, 5)}

        for starts_after in possible_starts:
            before = self.target_sequence[:starts_after + 1]
            after = self.target_sequence[starts_after + 1:]
            for length in mers:
                for mer in mers[length]:
                    result = before + mer + after
                    insertion = DegenerateInsertion([starts_after], [mer])
                    degenerate_inss[result].append(insertion)

        for degenerate_class in degenerate_inss.values():
            if len(degenerate_class) > 1:
                collapsed = DegenerateInsertion.collapse(degenerate_class)
                for insertion in degenerate_class:
                    singleton_to_full[insertion] = collapsed

        return singleton_to_full

    def expand_degenerate_indel(self, indel):
        return self.degenerate_indels.get(indel, indel)

    def overhang_insertion(self, length, copies=1):
        ''' Return DegenerateInsertion from offset RuvC cut producing 5' overhang. '''
        if self.protospacer_feature.strand == '-':
            start_after = self.cut_after
            seq = self.target_sequence[self.cut_after + 1:self.cut_after + 1 + length]
        else:
            start_after = self.cut_after
            seq = self.target_sequence[self.cut_after - (length - 1):self.cut_after + 1]

        actual_insertion = DegenerateInsertion([start_after], [seq * copies])
        degenerate_insertion = self.expand_degenerate_indel(actual_insertion)

        shifted = DegenerateInsertion([s - self.anchor for s in degenerate_insertion.starts_afters], degenerate_insertion.seqs)

        return ('insertion', 'near cut', str(shifted))

    @memoized_property
    def edit_name(self):
        SNVs = self.donor_SNVs

        if len(SNVs['target']) == 1:
            SNV_name = sorted(SNVs['target'])[0]
            position = SNVs['target'][SNV_name]['position']
            strand = SNVs['target'][SNV_name]['strand']
            target_base = SNVs['target'][SNV_name]['base']
            donor_base = SNVs['donor'][SNV_name]['base']
            
            if strand != self.protospacer_feature.strand:
                raise ValueError
                
            if strand == '+':
                offset = position - self.cut_after
            else:
                offset = self.cut_after - position + 1

            edit_name = f'+{offset:02d}_{target_base}_to_{donor_base}'

        elif len(SNVs['target']) > 1:
            raise NotImplementedError

        elif len(self.donor_insertions) == 1:
            edit_name = 'insertion'

        else:
            edit_name = 'no_edit'

        return edit_name

    @memoized_property
    def intended_prime_edit_type(self):
        if len(self.pegRNA_names) == 0:
            edit_type = None
        else:
            pegRNA_name = self.pegRNA_names[0]

            if (self.target, f'deletion_{pegRNA_name}') in self.features:
                edit_type = 'deletion'
            elif (pegRNA_name, f'insertion_{pegRNA_name}') in self.features:
                edit_type = 'insertion'
            elif (pegRNA_name, f'combination_{pegRNA_name}') in self.features:
                edit_type = 'combination'
            else:
                edit_type = 'SNV'

        return edit_type

    def calculate_microhomology_lengths(self, donor_to_use='homologous', donor_strand='+'):
        def num_matches_at_edge(first, second, relevant_edge):
            ''' Count the number of identical characters at the beginning
            or end of strings first and second.
            '''
            if relevant_edge == 'beginning':
                pass
            elif relevant_edge == 'end':
                first = first[::-1]
                second = second[::-1]
            else:
                raise ValueError(relevant_edge)
                
            num_matches = 0
            
            for f, s in zip(first, second):
                if f == s:
                    num_matches += 1
                else:
                    break
            
            return num_matches

        def count_total_MH_nts(first_seq, second_seq, first_seq_edge, second_seq_edge, include_before=True, include_after=True):
            ''' Count the number of consecutive exact matches that span a junction between truncations of
            first_seq and second_seq such that the first_seq up to and including first_seq_edge is joined
            to second_seq from second_seq_edge onward.
            '''
            first_before, first_after = first_seq[:first_seq_edge + 1], first_seq[first_seq_edge + 1:]
            second_before, second_after = second_seq[:second_seq_edge], second_seq[second_seq_edge:]
            
            total = 0
            
            if include_before:
                total += num_matches_at_edge(first_before, second_before, 'end')
            
            if include_after:
                total += num_matches_at_edge(first_after, second_after, 'beginning')
            
            return total

        expected_MH_lengths = {
            5: {
                'fixed': np.zeros(1000),
                'variable': np.zeros(1000),
            },
            3 : {
                'fixed': np.zeros(1000),
                'variable': np.zeros(1000),
            } 
        }

        offsets_that_use_HAs = set()

        if donor_to_use == 'homologous':
            donor_seq = self.donor_sequence

            # Exclude offsets that actually use the full HAs, since junctions that involve
            # pairing of the actual HAs don't get counted as donor MH.

            if self.homology_arms is not None:
                for side in [5, 3]:
                    donor_feature = self.homology_arms[side]['donor']
                    target_feature = self.homology_arms[side]['target']
                    
                    if donor_feature.strand != target_feature.strand:
                        raise ValueError
                        
                    HA_offset = donor_feature.start - target_feature.start
                    
                    offsets_that_use_HAs.add(HA_offset)
        else:
            donor_seq = self.reference_sequences[self.nonhomologous_donor]

        if donor_strand == '-':
            donor_seq = utilities.reverse_complement(donor_seq)

        # Compare junctions between 3' truncations of donor (i.e. donor sequence missing some amount from the 3' end)
        # and the target after the cut. This overestimates the true number of nts capable of annealing because
        # it credits nts on the target on the other side of the cut.

        target_seq = self.target_sequence

        donor_start = 0
        donor_end = len(donor_seq)

        target_edge = self.cut_after + 1
        for donor_edge in range(donor_start, donor_end):
            offset = donor_edge - (target_edge - 1)
            if offset in offsets_that_use_HAs:
                continue

            total_MH_nts = count_total_MH_nts(donor_seq, target_seq, donor_edge, target_edge)
            expected_MH_lengths[3]['fixed'][total_MH_nts] += 1

        # Compare junctions between 3' truncations of donor (i.e. donor sequence missing some amount from the 3' end)
        # and resections of the target after the cut.    

        target_start = self.cut_after + 1
        target_end = target_start + 100

        for target_edge in range(target_start, target_end):
            for donor_edge in range(donor_start, donor_end):
                offset = donor_edge - (target_edge - 1)
                if offset in offsets_that_use_HAs:
                    continue

                MH = count_total_MH_nts(donor_seq, target_seq, donor_edge, target_edge)
                expected_MH_lengths[3]['variable'][MH] += 1

        # Compare junctions between 5' truncations of donor (i.e. donor sequence missing some amount from the 5' end)
        # and the target before the cut. This overestimates the true number of nts capable of annealing because
        # it credits nts on the target on the other side of the cut.

        donor_start = 0
        donor_end = len(donor_seq)

        target_edge = self.cut_after
        for donor_edge in range(donor_start, donor_end):
            offset = (donor_edge - 1) - target_edge
            if offset in offsets_that_use_HAs:
                continue

            MH = count_total_MH_nts(target_seq, donor_seq, target_edge, donor_edge)
            expected_MH_lengths[5]['fixed'][MH] += 1

        # Compare junctions between 5' truncations of donor (i.e. donor sequence missing some amount from the 5' end)
        # and resections of the target after the cut.         

        target_start = self.cut_after - 100
        target_end = self.cut_after

        for target_edge in range(target_start, target_end):
            for donor_edge in range(donor_start, donor_end):
                offset = (donor_edge - 1) - target_edge
                if offset in offsets_that_use_HAs:
                    continue

                MH = count_total_MH_nts(target_seq, donor_seq, target_edge, donor_edge)
                expected_MH_lengths[5]['variable'][MH] += 1
                
        for side in [5, 3]:
            for edge in ['fixed', 'variable']:
                expected_MH_lengths[side][edge] = expected_MH_lengths[side][edge] / sum(expected_MH_lengths[side][edge])
        
        return expected_MH_lengths

    @memoized_property
    def nick_offset(self):
        nicking_sgRNAs = [n for n in self.protospacer_names if n != self.primary_protospacer]
        if len(nicking_sgRNAs) == 1:
            nicking_sgRNA = nicking_sgRNAs[0]

            primary_cut_after = [v for k, v in self.cut_afters.items() if k.rsplit('_', 1)[0] == self.primary_protospacer][0]

            nick_cut_after = [v for k, v in self.cut_afters.items() if k.rsplit('_', 1)[0] == nicking_sgRNA][0]
            
            sign_multiple = -1 if self.protospacer_feature.strand == '-' else 1
            offset = sign_multiple * (nick_cut_after - primary_cut_after)
        else:
            offset = 0

        return offset

    def remove_organism_from_alignment(self, al):
        organism, original_name = al.reference_name.split('_', 1)
        organism_matches = {n for n in self.supplemental_headers if al.reference_name.startswith(n)}
        if len(organism_matches) != 1:
            raise ValueError(al.reference_name, self.supplemental_headers)
        else:
            organism = organism_matches.pop()
            original_name = al.reference_name[len(organism) + 1:]

        header = self.supplemental_headers[organism]
        al_dict = al.to_dict()
        al_dict['ref_name'] = original_name
        original_al = pysam.AlignedSegment.from_dict(al_dict, header)

        return organism, original_al

    @memoized_property
    def PBS_names_by_side_of_target(self):
        if len(self.pegRNA_names) == 0:
            by_side = {}
        else:
            by_side = knock_knock.pegRNAs.PBS_names_by_side_of_target(self.pegRNA_names,
                                                                      self.target,
                                                                      self.features,
                                                                     )

        return by_side

    @memoized_property
    def PBS_names_by_side_of_read(self):
        if self.sequencing_direction == '+':
            read_side_to_target_side = {
                'left': 5,
                'right': 3,
            }
        else:
            read_side_to_target_side = {
                'right': 5,
                'left': 3,
            }

        by_side = {}
        for read_side, target_side in read_side_to_target_side.items():
            PBS_name = self.PBS_names_by_side_of_target.get(target_side)
            if PBS_name is not None:
                by_side[read_side] = PBS_name

        return by_side

    @memoized_property
    def pegRNA_names_by_side_of_read(self):
        return {side: knock_knock.pegRNAs.extract_pegRNA_name(PBS_name) for side, PBS_name in self.PBS_names_by_side_of_read.items()}

    @memoized_property
    def pegRNA_name_to_side_of_read(self):
        return utilities.reverse_dictionary(self.pegRNA_names_by_side_of_read)

    @memoized_property
    def pegRNA_name_to_color(self):
        return {name: f'C{i + 2}' for i, name in enumerate(self.pegRNA_names)}

    @memoized_property
    def pegRNA_names_by_side_of_target(self):
        return {side: knock_knock.pegRNAs.extract_pegRNA_name(PBS_name) for side, PBS_name in self.PBS_names_by_side_of_target.items()}

    @memoized_property
    def pegRNA_intended_deletion(self):
        if len(self.pegRNA_names) == 0:
            deletion = None

        elif len(self.pegRNA_names) == 1:
            _, _, deletion = knock_knock.pegRNAs.infer_edit_features(self.pegRNA_names[0],
                                                                     self.target,
                                                                     self.features,
                                                                     self.reference_sequences,
                                                                     max_deletion_length=self.max_programmed_deletion_length,
                                                                    )

        elif len(self.pegRNA_names) == 2:
            results = knock_knock.pegRNAs.infer_twin_pegRNA_features(self.pegRNA_names,
                                                                     self.target,
                                                                     self.features,
                                                                     self.reference_sequences,
                                                                    )
            deletion = results['deletion']

        deletion = self.expand_degenerate_indel(deletion)

        return deletion

    @memoized_property
    def is_prime_del(self):
        if len(self.pegRNA_names) == 2:
            results = knock_knock.pegRNAs.infer_twin_pegRNA_features(self.pegRNA_names,
                                                                     self.target,
                                                                     self.features,
                                                                     self.reference_sequences,
                                                                    )
            is_prime_del = results['is_prime_del']
        else:
            is_prime_del = False

        return is_prime_del

    @memoized_property
    def pegRNA_SNVs(self):
        ''' Format:
        {
            target_name: {
                SNV_name: {
                    'position': position in target,
                    'strand': '+',
                    'base': base on + strand of target at position,
                },
                ...,
            },
            pegRNA_name: {
                SNV_name: {
                    'position': position in pegRNA,
                    'strand': opposite of the strand of the protospacer targeted by pegRNA,
                    'base': base on + strand of pegRNA at position, so needs to be RC'ed if strand
                            is '-' to be compared to base of target SNV,
                },
                ...,
            },
            ...,
        } 
        '''
        if len(self.pegRNA_names) == 0:
            SNVs = None

        elif len(self.pegRNA_names) == 1:
            _, SNVs, _ = knock_knock.pegRNAs.infer_edit_features(self.pegRNA_names[0],
                                                                 self.target,
                                                                 self.features,
                                                                 self.reference_sequences,
                                                                 max_deletion_length=self.max_programmed_deletion_length,
                                                                )

        elif len(self.pegRNA_names) == 2:
            results = knock_knock.pegRNAs.infer_twin_pegRNA_features(self.pegRNA_names,
                                                                     self.target,
                                                                     self.features,
                                                                     self.reference_sequences,
                                                                    )
            SNVs = results['SNVs']

        else:
            raise ValueError

        return SNVs

    @memoized_property
    def pegRNA_programmed_deletions(self):
        deletions = []

        if len(self.pegRNA_names) > 0:
            feature_names = []

            for pegRNA_name in self.pegRNA_names:
                feature_names.append(f'deletion_{pegRNA_name}')

            if len(self.pegRNA_names) == 2:
                feature_names.append(f'deletion_{self.pegRNA_names[0]}_{self.pegRNA_names[1]}')

            for feature_name in feature_names:
                if (self.target, feature_name) in self.features:
                    deletions.append(self.features[self.target, feature_name])

        return deletions

    @memoized_property
    def pegRNA_programmed_insertion_features(self):
        insertions = []

        if len(self.pegRNA_names) > 0:
            for pegRNA_name in self.pegRNA_names:
                feature_name = f'insertion_{pegRNA_name}'
                if (pegRNA_name, feature_name) in self.features:
                    insertions.append(self.features[pegRNA_name, feature_name])

        return insertions

    @memoized_property
    def pegRNA_programmed_insertion(self):
        ''' Returns an unexpanded DegenerateInsertion representing the insertion
        programmed by a single pegRNA.
        '''
        if len(self.pegRNA_programmed_insertion_features) != 1:
            insertion = None
        else:
            pegRNA_name = self.pegRNA_names[0]

            HA_RT = self.features[self.target, f'HA_RT_{pegRNA_name}']

            insertion_seq = self.feature_sequence(pegRNA_name, f'insertion_{pegRNA_name}')

            if HA_RT.strand == '+':
                starts_after = HA_RT.start - 1
            else:
                starts_after = HA_RT.end
                # insertion_seq as returned by self.feature_sequence should be the 
                # RC of the relevant part of the pegRNA, but if the protospacer is
                # on the minus strand, this needs to be RCed to represent the inserted
                # sequence relative to the plus strand. 
                insertion_seq = utilities.reverse_complement(insertion_seq)

            insertion = DegenerateInsertion([starts_after], [insertion_seq]) 

        return insertion

    #@memoized_property
    #def intended_edit_sequence(self):
    #    if len(self.pegRNA_names) != 2:
    #        raise NotImplementedError

    #    results = knock_knock.pegRNAs.infer_twin_pegRNA_features(self.pegRNA_names,
    #                                                             self.target,
    #                                                             self.features,
    #                                                             self.reference_sequences,
    #                                                            )

    #    primer_seqs = {side: self.primers_by_side_of_read[side].sequence(self.reference_sequences) for side in ['left', 'right']}
    #    primer_seqs['right'] = utilities.reverse_complement(primer_seqs['right'])

    #    extended_expected_seq = results['intended_edit_seq']
    #    if self.sequencing_direction == '-':    
    #        extended_expected_seq = utilities.reverse_complement(extended_expected_seq)

    #    start = extended_expected_seq.index(primer_seqs['left'])
    #    end = extended_expected_seq.index(primer_seqs['right']) + len(primer_seqs['right'])

    #    expected_seq = extended_expected_seq[start:end]

    #    return expected_seq

def degenerate_indel_from_string(details_string):
    if details_string is None:
        return None
    else:
        kind, rest = details_string.split(':')

        if kind == 'D':
            return DegenerateDeletion.from_string(details_string)
        elif kind == 'I':
            return DegenerateInsertion.from_string(details_string)

class DegenerateDeletion():
    def __init__(self, starts_ats, length):
        self.kind = 'D'
        self.starts_ats = tuple(starts_ats)
        self.num_MH_nts = len(self.starts_ats) - 1
        self.length = length
        self.ends_ats = [s + self.length - 1 for s in self.starts_ats]

    @classmethod
    def from_string(cls, details_string):
        kind, rest = details_string.split(':', 1)
        starts_string, length_string = rest.split(',')

        starts_ats = [int(s) for s in starts_string.strip('{}').split('|')]
        length = int(length_string)

        return DegenerateDeletion(starts_ats, length)

    @classmethod
    def collapse(cls, degenerate_deletions):
        lengths = {d.length for d in degenerate_deletions}
        if len(lengths) > 1:
            for d in degenerate_deletions:
                print(d)
            raise ValueError
        length = lengths.pop()

        starts_ats = set()
        for d in degenerate_deletions:
            starts_ats.update(d.starts_ats)

        starts_ats = sorted(starts_ats)

        return DegenerateDeletion(starts_ats, length)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_ats))
        if len(self.starts_ats) > 1:
            starts_string = '{' + starts_string + '}'

        full_string = f'D:{starts_string},{self.length}'

        return full_string

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.starts_ats == other.starts_ats and self.length == other.length

    def __hash__(self):
        return hash((self.starts_ats, self.length))

    def singletons(self):
        return (DegenerateDeletion([starts_at], self.length) for starts_at in self.starts_ats)
    
class DegenerateInsertion():
    def __init__(self, starts_afters, seqs):
        self.kind = 'I'
        self.starts_afters = tuple(starts_afters)
        self.seqs = tuple(seqs)
        
        lengths = set(len(seq) for seq in self.seqs)
        if len(lengths) > 1:
            raise ValueError
        self.length = lengths.pop()
    
        self.pairs = list(zip(self.starts_afters, self.seqs))

    @classmethod
    def from_string(cls, details_string):
        kind, rest = details_string.split(':', 1)
        starts_string, seqs_string = rest.split(',')
        starts_afters = [int(s) for s in starts_string.strip('{}').split('|')]
        seqs = [seq for seq in seqs_string.strip('{}').split('|')]

        return DegenerateInsertion(starts_afters, seqs)
    
    @classmethod
    def from_pairs(cls, pairs):
        starts_afters, seqs = zip(*pairs)
        return DegenerateInsertion(starts_afters, seqs)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_afters))
        seqs_string = '|'.join(self.seqs)

        if len(self.starts_afters) > 1:
            starts_string = '{' + starts_string + '}'
            seqs_string = '{' + seqs_string + '}'

        full_string = f'I:{starts_string},{seqs_string}'

        return full_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.starts_afters == other.starts_afters and self.seqs == other.seqs
    
    def __hash__(self):
        return hash((self.starts_afters, self.seqs))

    def singletons(self):
        return (DegenerateInsertion([starts_after], [seq]) for starts_after, seq in self.pairs)
    
    @classmethod
    def collapse(cls, degenerate_insertions):
        all_pairs = []

        for d in degenerate_insertions:
            all_pairs.extend(d.pairs)

        all_pairs = sorted(all_pairs)

        return DegenerateInsertion.from_pairs(all_pairs)

class SNV():
    def __init__(self, position, basecall, quality):
        self.position = position
        self.basecall = basecall
        self.quality = quality

    @classmethod
    def from_string(cls, details_string):
        basecall = details_string[-1]

        if basecall.islower():
            quality = 0
        else:
            quality = 40

        position = int(details_string[:-1])

        return SNV(position, basecall, quality)

    def __str__(self):
        if self.quality < 30:
            bc = self.basecall.lower()
        else:
            bc = self.basecall

        return f'{self.position}{bc}'

class SNVs():
    def __init__(self, snvs):
        self.snvs = sorted(snvs, key=lambda snv: snv.position)
    
    def __str__(self):
        return ','.join(str(snv) for snv in self.snvs)

    @classmethod
    def from_string(cls, details_string):
        return SNVs([SNV.from_string(s) for s in details_string.split(',')])

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.snvs)

    def __iter__(self):
        return iter(self.snvs)

    @property
    def positions(self):
        return [snv.position for snv in self.snvs]
    
    @property
    def basecalls(self):
        return [snv.basecall for snv in self.snvs]

    def __lt__(self, other):
        if max(self.positions) != max(other.positions):
            return max(self.positions) < max(other.positions)
        else:
            if len(self) < len(other):
                return True
            elif len(self) == len(other):
                if self.positions != other.positions:
                    return self.positions < other.positions
                else: 
                    return self.basecalls < other.basecalls
            else:
                return False

def parse_benchling_genbank(gb_record):
    convert_strand = {
        -1: '-',
        1: '+',
    }

    fasta_record = fasta.Read(gb_record.name, str(gb_record.seq))

    gff_features = []

    for gb_feature in gb_record.features:
        feature = gff.Feature.from_fields(
            seqname=gb_record.name,
            feature=gb_feature.type,
            start=gb_feature.location.start,
            end=gb_feature.location.end - 1,
            strand=convert_strand[gb_feature.location.strand],
        )
        attribute = {}
        for k, v in gb_feature.qualifiers.items():
            if isinstance(v, list):
                v = v[0]
            attribute[k] = v

        if 'label' in attribute:
            attribute['ID'] = attribute.pop('label')

        if 'ApEinfo_fwdcolor' in attribute:
            attribute['color'] = attribute.pop('ApEinfo_fwdcolor')

        if 'ApEinfo_revcolor' in attribute:
            attribute.pop('ApEinfo_revcolor')

        feature.attribute = attribute

        gff_features.append(feature)

    return fasta_record, gff_features

def get_all_targets(base_dir):
    targets_dir = Path(base_dir) / 'targets'
    names = (p.name for p in targets_dir.glob('*') if p.is_dir())
    targets = [TargetInfo(base_dir, n) for n in names]
    return targets

def locate_supplemental_indices(base_dir):
    base_dir = Path(base_dir)
    override_fn = base_dir / 'index_locations.yaml'
    if override_fn.exists():
        locations = yaml.safe_load(override_fn.read_text())
    else:
        locations = {}
        indices_dir = base_dir / 'indices'
        if indices_dir.is_dir():
            for d in indices_dir.iterdir():
                if d.is_dir():
                    locations[d.name] = {
                        'STAR': d / 'STAR',
                        'fasta': d / 'fasta',
                        'minimap2': d / f'minimap2/{d.name}.mmi',
                    }

    return locations
