import array
import warnings
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import yaml
import pysam
import Bio.SeqIO
import Bio.SeqUtils
import numpy as np
import pandas as pd

from Bio import BiopythonWarning
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna

from hits import fasta, fastq, gff, utilities, mapping_tools, interval, sam, sw, genomes

memoized_property = utilities.memoized_property

class Effector():
    def __init__(self, name, PAM_pattern, PAM_side, cut_after_offset):
        self.name = name
        self.PAM_pattern = PAM_pattern
        self.PAM_side = PAM_side
        # cut_after_offset is relative to the 5'-most nt of the PAM
        self.cut_after_offset = cut_after_offset

effectors = {
    'SpCas9': Effector('SpCas9', 'NGG', 3, -4),
    'SaCas9': Effector('SaCas9', 'NNGRRT', 3, -4),
    'Cpf1': Effector('Cpf1', 'TTTN', 5, (22, 26)),
}

class TargetInfo():
    def __init__(self, base_dir, name,
                 donor=None,
                 nonhomologous_donor=None,
                 sgRNA=None,
                 primer_names=None,
                 supplemental_headers=None,
                 gb_records=None,
                 ):
        self.name = name
        self.dir = Path(base_dir) / 'targets' / name

        manifest_fn = self.dir / 'manifest.yaml'
        manifest = yaml.safe_load(manifest_fn.open())
        self.target = manifest['target']
        self.sources = manifest['sources']
        self.gb_records = gb_records

        self.sgRNA = sgRNA
        self.donor = donor
        self.nonhomologous_donor = nonhomologous_donor
        self.donor_specific = manifest.get('donor_specific', 'GFP11') 
        self.supplemental_headers = supplemental_headers

        self.default_HAs = manifest.get('default_HAs')

        self.fns = {
            'ref_fasta': self.dir / 'refs.fasta',
            'ref_gff': self.dir / 'refs.gff',

            'protospacer_fasta': self.dir / 'protospacers.fasta',
            'protospacer_STAR_prefix_template': self.dir / 'protospacers_{}.',
            'protospacer_bam_template': self.dir / 'protospacers_{}.bam',

            'bowtie2_index': self.dir / 'refs',
            'STAR_index_dir': self.dir / 'STAR_index',

            'degenerate_insertions': self.dir / 'degenerate_insertions.txt',
            'degenerate_deletions': self.dir / 'degenerate_deletions.txt',
        }
        
        self.primer_names = primer_names

    @memoized_property
    def header(self):
        return sam.header_from_fasta(self.fns['ref_fasta'])

    @memoized_property
    def sgRNAs(self):
        if self.sgRNA is None:
            sgRNAs = sorted(n for t, n in self.all_sgRNA_features)
        elif isinstance(self.sgRNA, list):
            sgRNAs = self.sgRNA
        else:
            sgRNAs = [self.sgRNA]

        return sgRNAs

    def make_references(self):
        ''' Generate fasta and gff files from genbank inputs. '''
        gb_fns = [self.dir / (source + '.gb') for source in self.sources]
        
        fasta_records = []
        all_gff_features = []
            
        if self.gb_records is None:
            self.gb_records = []
            for gb_fn in gb_fns:
                self.gb_records.append(Bio.SeqIO.read(str(gb_fn), 'genbank'))

        for gb_record in self.gb_records:
            fasta_record, gff_features = parse_benchling_genbank(gb_record)

            fasta_records.append(fasta_record)
            all_gff_features.extend(gff_features)
            
        with self.fns['ref_fasta'].open('w') as fasta_fh:
            for record in fasta_records:
                fasta_fh.write(str(record))
                
        pysam.faidx(str(self.fns['ref_fasta']))
                
        with self.fns['ref_gff'].open('w') as gff_fh:
            gff_fh.write('##gff-version 3\n')
            for feature in sorted(all_gff_features):
                # Note: all sgRNAs should have feature field format 'sgRNA_{effector}'
                if feature.feature.startswith('sgRNA'):
                    try:
                        _, effector = feature.feature.split('_')
                    except:
                        print(self.name, feature)
                        raise
                    feature.feature = 'sgRNA'
                    feature.attribute['effector'] = effector

                gff_fh.write(str(feature) + '\n')

    def make_protospacer_fastas(self):
        with open(self.fns['protospacer_fasta'], 'w') as fh:
            for (target, sgRNA_name), feature in sorted(self.all_sgRNA_features.items()):
                seq = self.target_sequence[feature.start:feature.end + 1]
                record = fasta.Read(sgRNA_name, seq)
                fh.write(str(record))

    def map_protospacers(self):
        indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'GRCh38': '/nvme/indices/refdata-cellranger-GRCh38-1.2.0/star/',
        }

        for index_name, index_dir in indices.items():
            output_prefix = str(self.fns['protospacer_STAR_prefix_template']).format(index_name)
            bam_fn = str(self.fns['protospacer_bam_template']).format(index_name)
            mapping_tools.map_STAR(self.fns['protospacer_fasta'], index_dir, output_prefix, mode='guide_alignment', bam_fn=bam_fn)

    def mapped_protospacer_locations(self, sgRNA_name, index_name):
        bam_fn = str(self.fns['protospacer_bam_template']).format(index_name)
        locations = set()

        with pysam.AlignmentFile(bam_fn) as bam_fh:
            for al in bam_fh:
                if al.query_name == sgRNA_name:
                    locations.add((al.reference_name, al.reference_start))

        return locations
    
    @memoized_property
    def mapped_protospacer_location(self):
        if len(self.sgRNAs) > 1:
            raise ValueError
        else:
            sgRNA = self.sgRNAs[0]

        index_name = 'GRCh38'
        locations = self.mapped_protospacer_locations(sgRNA, index_name)
        
        if len(locations) > 1:
            raise ValueError(locations)
        else:
            location = locations.pop()

        return location

    def make_bowtie2_index(self):
        mapping_tools.build_bowtie2_index(self.fns['bowtie2_index'], [self.fns['ref_fasta']])
    
    @memoized_property
    def features(self):
        features = {
            (f.seqname, f.attribute['ID']): f
            for f in gff.get_all_features(self.fns['ref_gff'])
            if 'ID' in f.attribute
        }
        return features

    @memoized_property
    def features_to_show(self):
        whitelist = {
            (self.donor, 'GFP'),
            (self.donor, 'GFP11'),
            (self.donor, 'PPX'),
            (self.donor, 'donor_specific'),
        }

        for side in [5, 3]:
            primer = (self.target, self.primers_by_side_of_target[side].attribute['ID'])
            whitelist.add(primer)

            if self.homology_arms is not None:
                target_HA = (self.target, self.homology_arms[side]['target'].attribute['ID'])
                whitelist.add(target_HA)

                if self.has_shared_homology_arms:
                    donor_HA = (self.donor, self.homology_arms[side]['donor'].attribute['ID'])
                    whitelist.add(donor_HA)

        return whitelist

    @memoized_property
    def sequencing_start(self):
        return self.features.get((self.target, 'sequencing_start'))

    @memoized_property
    def reference_sequences(self):
        if self.fns['ref_fasta'].exists():
            seqs = fasta.to_dict(self.fns['ref_fasta'])
        else:
            seqs = {}

        return seqs

    @memoized_property
    def target_sequence(self):
        return self.reference_sequences[self.target]
    
    @memoized_property
    def target_sequence_bytes(self):
        return self.target_sequence.encode()
    
    @memoized_property
    def donor_sequence_bytes(self):
        if self.donor_sequence is None:
            return None
        else:
            return self.donor_sequence.encode()

    @memoized_property
    def seed_and_extender(self):
        def fake_extender(*args):
            return [] 

        extenders = {
            'target': sw.SeedAndExtender(self.target_sequence_bytes, 20, self.header, self.target).seed_and_extend,
        }
        if self.donor_sequence_bytes is not None:
            extenders['donor'] = sw.SeedAndExtender(self.donor_sequence_bytes, 20, self.header, self.donor).seed_and_extend
        else:
            # If there isn't a donor, always return no alignments.
            extenders['donor'] = fake_extender

        return extenders

    @memoized_property
    def donor_sequence(self):
        if self.donor is None:
            return None
        else:
            return self.reference_sequences[self.donor]

    @memoized_property
    def sgRNA_features(self):
        return {name: self.features[self.target, name] for name in self.sgRNAs}
    
    @memoized_property
    def sgRNA_feature(self):
        features = list(self.sgRNA_features.values())
        if len(features) > 1:
            raise ValueError(len(features))
        return features[0]

    @memoized_property
    def all_sgRNA_features(self):
        return {name: feature for name, feature in self.features.items() if feature.feature == 'sgRNA'}
    
    @memoized_property
    def guide_slices(self):
        guide_slices = {name: slice(sgRNA.start, sgRNA.end + 1) for name, sgRNA in self.sgRNA_features.items()}
        return guide_slices
    
    @memoized_property
    def guide_slice(self):
        if len(self.guide_slices) > 1:
            raise ValueError(self.guide_slices)
        else:
            return list(self.guide_slices.values())[0]
    
    @memoized_property
    def PAM_slices(self):
        PAM_slices = {}

        for name, sgRNA in self.sgRNA_features.items():
            effector = effectors[sgRNA.attribute['effector']]

            before_slice = slice(sgRNA.start - len(effector.PAM_pattern), sgRNA.start)
            after_slice = slice(sgRNA.end + 1, sgRNA.end + 1 + len(effector.PAM_pattern))

            if (sgRNA.strand == '+' and effector.PAM_side == 5) or (sgRNA.strand == '-' and effector.PAM_side == 3):
                PAM_slice = before_slice
            else:
                PAM_slice = after_slice

            PAM_slices[name] = PAM_slice

            PAM_seq = self.target_sequence[PAM_slice].upper()
            if sgRNA.strand == '-':
                PAM_seq = utilities.reverse_complement(PAM_seq)
            pattern, *matches = Bio.SeqUtils.nt_search(PAM_seq, effector.PAM_pattern) 
            if 0 not in matches:
                raise ValueError('{}: {} doesn\'t match {} PAM'.format(name, PAM_seq, pattern))

        return PAM_slices
    
    @memoized_property
    def PAM_slice(self):
        if len(self.PAM_slices) > 1:
            raise ValueError(self.PAM_slices)
        else:
            return list(self.PAM_slices.values())[0]

    @memoized_property
    def effector(self):
        return effectors[self.sgRNA_feature.attribute['effector']]

    @memoized_property
    def cut_afters(self):
        cut_afters = {}
        for name, sgRNA in self.sgRNA_features.items():
            effector = effectors[sgRNA.attribute['effector']]
            if isinstance(effector.cut_after_offset, int):
                offsets = [effector.cut_after_offset]
                key_suffixes = ['']
            else:
                offsets = effector.cut_after_offset
                key_suffixes = ['_{}'.format(i) for i in range(len(offsets))]

            for offset, key_suffix in zip(offsets, key_suffixes):
                if sgRNA.strand == '+':
                    PAM_5 = self.PAM_slices[name].start
                    cut_after = PAM_5 + offset
                else:
                    PAM_5 = self.PAM_slices[name].stop - 1
                    # -1 extra because cut_after is on the other side of the cut
                    cut_after = PAM_5 - offset - 1
                
                cut_afters[name + key_suffix] = cut_after

        return cut_afters
    
    @memoized_property
    def cut_after(self):
        ''' when processing assumes there will be only one cut, use this '''
        return min(self.cut_afters.values())

    def around_cuts(self, each_side):
        intervals = [interval.Interval(cut_after - each_side + 1, cut_after + each_side) for cut_after in self.cut_afters.values()]
        return interval.DisjointIntervals(intervals)

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
            raise ValueError('not a primer on each side of cut: {}'.format(primers))

        return primers
    
    @memoized_property
    def primers_by_side_of_target(self):
        by_side = {}
        primers = [primer for primer in self.primers.values()]
        by_side[5], by_side[3] = sorted(primers, key=lambda p: p.start)
        return by_side

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
        strand = self.sgRNA_feature.strand
        PAM_side = self.effector.PAM_side

        if (strand == '+' and PAM_side == 3) or (strand == '-' and PAM_side == 5):
            target_to_PAM = {5: 'PAM-proximal', 3: 'PAM-distal'}
        elif (strand == '+' and PAM_side == 5) or (strand == '-' and PAM_side == 3):
            target_to_PAM = {3: 'PAM-proximal', 5: 'PAM-distal'}

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

        # Load homology arms from gff features.
        HAs = defaultdict(dict)

        ref_name_to_source = {
            self.target: 'target',
            self.donor: 'donor',
        }

        for (ref_name, feature_name), feature in self.features.items():
            source = ref_name_to_source.get(ref_name)
            if source is not None:
                if feature_name.startswith('HA_'):
                    HAs[feature_name][source] = feature

        if len(HAs) == 0:
            return None
                    
        paired_HAs = {}

        # Confirm that every HA name that exists on both the target and donor has the same
        # sequence on each.
        for name in HAs:
            if 'target' not in HAs[name] or 'donor' not in HAs[name]:
                continue
            
            seqs = {}
            for source in ['target', 'donor']:
                ref_seq = self.reference_sequences[getattr(self, source)]
                HA = HAs[name][source]
                HA_seq = ref_seq[HA.start:HA.end + 1]
                if HA.strand == '-':
                    HA_seq = utilities.reverse_complement(HA_seq)
                    
                seqs[source] = HA_seq
                
            if seqs['target'] != seqs['donor']:
                # May want to warn here:
                #print(f'warning: {name} not identical sequence on target and donor')
                pass

            paired_HAs[name] = HAs[name]

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

        by_target_side = {}
        by_target_side[5], by_target_side[3] = sorted(paired_HAs, key=lambda n: HAs[n]['target'].start)

        for target_side, name in by_target_side.items():
            PAM_side = self.target_side_to_PAM_side[target_side]

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
    def amplicon_interval(self):
        primers = self.primers_by_side_of_target
        return interval.Interval(primers[5].start, primers[3].end)

    @memoized_property
    def amplicon_length(self):
        return len(self.amplicon_interval)

    @memoized_property
    def clean_HDR_length(self):
        if self.has_shared_homology_arms:
            HAs = self.homology_arms
            return self.amplicon_length + HAs[3]['donor'].start - HAs[5]['donor'].end - 1
        else:
            return None

    @memoized_property
    def fingerprints(self):
        fps = {
            self.target: [],
            self.donor: [],
        }

        for name in self.SNP_names:
            fs = {k: self.features[k, name] for k in (self.target, self.donor)}
            ps = {k: (f.strand, f.start) for k, f in fs.items()}
            bs = {k: self.reference_sequences[k][p:p + 1] for k, (strand, p) in ps.items()}
            
            for k in bs:
                if fs[k].strand == '-':
                    bs[k] = utilities.reverse_complement(bs[k])
                
                fps[k].append((ps[k], bs[k]))

        return fps

    @memoized_property
    def SNP_names(self):
        return sorted([name for seq_name, name in self.features if seq_name == self.donor and name.startswith('SNP')])

    @memoized_property
    def donor_SNVs(self):
        SNVs = {'target': {}, 'donor': {}}

        if self.donor is None:
            return SNVs

        for key, seq_name in [('target', self.target), ('donor', self.donor)]:
            seq = self.reference_sequences[seq_name]
            for name in self.SNP_names:
                feature = self.features[seq_name, name]
                strand = feature.strand
                position = feature.start
                b = seq[position:position + 1]
                if strand == '-':
                    b = utilities.reverse_complement(b)

                SNVs[key][name] = {
                    'position': position,
                    'strand': strand,
                    'base': b
                }
            
        return SNVs
    
    @memoized_property
    def simple_donor_SNVs(self):
        ''' {ref_name, position: base identity in donor if on forward strand}}'''
        SNVs = {}

        for SNV_name, donor_SNV_details in self.donor_SNVs['donor'].items():
            target_SNV_details = self.donor_SNVs['target'][SNV_name]

            donor_base = donor_SNV_details['base']
            if donor_SNV_details['strand'] == '-':
                # We want the forward strand base.
                donor_base = utilities.reverse_complement(donor_base)

            SNVs[self.donor, donor_SNV_details['position']] = donor_base

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
            if seq_name == self.target and feature.feature == 'donor_deletion':
                donor_name = name[:-len('_deletion')]
                if donor_name == self.donor:
                    deletion = DegenerateDeletion([feature.start], len(feature))
                    deletion = self.expand_degenerate_indel(deletion)
                    donor_deletions.append(deletion)

        return donor_deletions

    @memoized_property
    def wild_type_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.target]])
    
    @memoized_property
    def donor_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.donor]])

    def identify_degenerate_indels(self):
        degenerate_dels = defaultdict(list)

        possible_starts = range(self.primers_by_side_of_target[5].start, self.primers_by_side_of_target[3].end)
        for starts_at in possible_starts:
            before = self.target_sequence[:starts_at]
            for length in range(1, 200):
                if starts_at + length >= len(self.target_sequence):
                    continue
                after = self.target_sequence[starts_at + length:]
                result = before + after
                deletion = DegenerateDeletion([starts_at], length)
                degenerate_dels[result].append(deletion)

        with self.fns['degenerate_deletions'].open('w') as fh:
            classes = sorted(degenerate_dels.values(), key=len, reverse=True)
            for degenerate_class in classes:
                if len(degenerate_class) > 1:
                    collapsed = DegenerateDeletion.collapse(degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

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

        with self.fns['degenerate_insertions'].open('w') as fh:
            classes = sorted(degenerate_inss.values(), key=len, reverse=True)
            for degenerate_class in classes:
                if len(degenerate_class) > 1:
                    collapsed = DegenerateInsertion.collapse(degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

    @memoized_property
    def degenerate_indels(self):
        singleton_to_full = {}

        for line in self.fns['degenerate_deletions'].open():
            deletion = degenerate_indel_from_string(line.strip())

            for singleton in deletion.singletons():
                singleton_to_full[singleton] = deletion
        
        for line in self.fns['degenerate_insertions'].open():
            insertion = degenerate_indel_from_string(line.strip())

            for singleton in insertion.singletons():
                singleton_to_full[singleton] = insertion

        return singleton_to_full

    def expand_degenerate_indel(self, indel):
        return self.degenerate_indels.get(indel, indel)

def degenerate_indel_from_string(details_string):
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

        full_string = 'D:{0},{1}'.format(starts_string, self.length)

        return full_string

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
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

        full_string = 'I:{0},{1}'.format(starts_string, seqs_string)

        return full_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
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

        return '{}{}'.format(self.position, bc)

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
            gb_record.name,
            '.',
            gb_feature.type,
            gb_feature.location.start,
            gb_feature.location.end - 1,
            '.',
            convert_strand[gb_feature.location.strand],
            '.',
            '.',
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

def design_amplicon_primers_from_csv(base_dir, genome='hg19'):
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'sgRNAs.csv'

    index_locations = locate_supplemental_indices(base_dir)
    if genome not in index_locations:
        print(f'Error: can\'t locate indices for {genome}')
        sys.exit(0)

    df = pd.read_csv(csv_fn).replace({np.nan: None})

    flanking_sequence = {}

    for _, row in df.iterrows():
        name = row['name']
        print(f'Designing {name}...')
        best_candidate = design_amplicon_primers(base_dir, row, genome)
        if best_candidate is not None:
            flanking_sequence[name] = best_candidate['target_seq']

    df['flanking_sequence'] = [flanking_sequence.get(name, '') for name in df['name']]
    final_csv_fn = base_dir / 'targets' / 'sgRNAs_flanking_sequence.csv'
    df.to_csv(final_csv_fn)

def design_amplicon_primers(base_dir, info, genome):
    base_dir = Path(base_dir)

    name = info['name']

    target_dir = base_dir / 'targets' / name
    target_dir.mkdir(parents=True, exist_ok=True)

    protospacer, *other_protospacers = info['sgRNA_sequence'].upper().split(';')
    
    protospacer_dir = target_dir / 'protospacer_alignment'
    protospacer_dir.mkdir(exist_ok=True)
    fastq_fn = protospacer_dir / 'protospacer.fastq'
    STAR_prefix = protospacer_dir / 'protospacer_'
    bam_fn = protospacer_dir / 'protospacer.bam'

    index_locations = locate_supplemental_indices(base_dir)
    STAR_index = index_locations[genome]['STAR']

    # Make a fastq file with a single read containing the protospacer sequence.
    
    with fastq_fn.open('w') as fh:
        quals = fastq.encode_sanger([40]*len(protospacer))
        read = fastq.Read('protospacer', protospacer, quals)
        fh.write(str(read))
        
    # Align the protospacer to the reference genome.
    mapping_tools.map_STAR(fastq_fn, STAR_index, STAR_prefix, mode='guide_alignment', bam_fn=bam_fn, sort=False)

    with pysam.AlignmentFile(bam_fn) as bam_fh:
        perfect_als = [al for al in bam_fh if not al.is_unmapped and sam.total_edit_distance(al) == 0]
    
    region_fetcher = genomes.build_region_fetcher(index_locations[genome]['fasta'])

    def evaluate_candidate(al):
        results = {
            'location': f'{al.reference_name} {al.reference_start:,} {sam.get_strand(al)}',
        }

        full_window_around = 5000

        full_around = region_fetcher(al.reference_name, al.reference_start - full_window_around, al.reference_end + full_window_around).upper()

        if sam.get_strand(al) == '+':
            ps_seq = protospacer
            ps_strand = 1
        else:
            ps_seq = utilities.reverse_complement(protospacer)
            ps_strand  = -1
        
        ps_start = full_around.index(ps_seq)

        protospacer_locations = [(ps_seq, ps_start, ps_strand)]

        for other_protospacer in other_protospacers:
            if other_protospacer in full_around:
                ps_seq = other_protospacer
                ps_strand = 1
            else:
                ps_seq =  utilities.reverse_complement(other_protospacer)
                if ps_seq not in full_around:
                    results['failed'] = f'protospacer {other_protospacer} not present near protospacer {protospacer}'
                    return results
                ps_strand = -1

            ps_start = full_around.index(ps_seq)
            protospacer_locations.append((ps_seq, ps_start, ps_strand))

        for ps_seq, ps_start, ps_strand in protospacer_locations:
            if ps_strand == 1:
                PAM_offset = len(protospacer)
                PAM_transform = utilities.identity
            else:
                PAM_offset = -3
                PAM_transform = utilities.reverse_complement

            PAM_start = ps_start + PAM_offset
            PAM = PAM_transform(full_around[PAM_start:PAM_start + 3])
            pattern, *matches = Bio.SeqUtils.nt_search(PAM, 'NGG')

            if 0 not in matches:
                # Note: this could incorrectly fail if there are multiple exact matches for an other_protospacer
                # in full_around.
                results['failed'] = f'bad PAM: {PAM} next to {ps_seq} (strand {ps_strand})'
                return results

        min_start = min(ps_start for ps_seq, ps_start, ps_strand in protospacer_locations)
        max_start = max(ps_start for ps_seq, ps_start, ps_strand in protospacer_locations)
        
        final_window_around = 500    

        final_start = min_start - final_window_around
        final_end = max_start + final_window_around

        target_seq = full_around[final_start:final_end]
        results['target_seq'] = target_seq

        return results

    good_candidates = []
    bad_candidates = []
    
    for al in perfect_als:
        results = evaluate_candidate(al)
        if 'failed' in results:
            bad_candidates.append(results)
        else:
            good_candidates.append(results)

    if len(good_candidates) == 0:
        if len(bad_candidates) == 0:
            print(f'Error building {name}: no perfect matches to sgRNA {protospacer} found in {genome}')
            return 

        else:
            print(f'Error building {name}: no valid genomic locations for {name}')

            for results in bad_candidates:
                print(f'\t{results["location"]}: {results["failed"]}')

            return 

    elif len(good_candidates) > 1:
        print(f'Warning: multiple valid genomic locations for {name}:')
        for results in good_candidates:
            print(f'\t{results["location"]}')
        best_candidate = good_candidates[0]
        print(f'Arbitrarily choosing {best_candidate["location"]}')
    else:
        best_candidate = good_candidates[0]

    return best_candidate

def build_target_info(base_dir, info, genome):
    base_dir = Path(base_dir)

    name = info['name']
    donor_name = f'{name}_donor'

    target_dir = base_dir / 'targets' / name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    protospacer, *other_protospacers = info['sgRNA_sequence'].upper().split(';')
    amplicon_primers = info['amplicon_primers'].split(';')
    donor_seq = info['donor_sequence']
    if donor_seq is None:
        has_donor = False
    else:
        has_donor = True
        donor_seq = donor_seq.upper()
    
    protospacer_dir = target_dir / 'protospacer_alignment'
    protospacer_dir.mkdir(exist_ok=True)
    fastq_fn = protospacer_dir / 'protospacer.fastq'
    STAR_prefix = protospacer_dir / 'protospacer_'
    bam_fn = protospacer_dir / 'protospacer.bam'

    index_locations = locate_supplemental_indices(base_dir)
    STAR_index = index_locations[genome]['STAR']

    target_gb_fn = target_dir / f'{name}.gb'
    donor_gb_fn = target_dir / f'{donor_name}.gb'

    # Make a fastq file with a single read containing the protospacer sequence.
    
    with fastq_fn.open('w') as fh:
        quals = fastq.encode_sanger([40]*len(protospacer))
        read = fastq.Read('protospacer', protospacer, quals)
        fh.write(str(read))
        
    # Align the protospacer to the reference genome.
    mapping_tools.map_STAR(fastq_fn, STAR_index, STAR_prefix, mode='guide_alignment', bam_fn=bam_fn, sort=False)

    with pysam.AlignmentFile(bam_fn) as bam_fh:
        perfect_als = [al for al in bam_fh if not al.is_unmapped and sam.total_edit_distance(al) == 0]
    
    region_fetcher = genomes.build_region_fetcher(index_locations[genome]['fasta'])
    
    def evaluate_candidate(al):
        results = {
            'location': f'{al.reference_name} {al.reference_start:,} {sam.get_strand(al)}',
        }

        full_window_around = 5000

        full_around = region_fetcher(al.reference_name, al.reference_start - full_window_around, al.reference_end + full_window_around).upper()

        if sam.get_strand(al) == '+':
            ps_seq = protospacer
            ps_strand = 1
        else:
            ps_seq = utilities.reverse_complement(protospacer)
            ps_strand  = -1
        
        ps_start = full_around.index(ps_seq)

        protospacer_locations = [(ps_seq, ps_start, ps_strand)]

        for other_protospacer in other_protospacers:
            if other_protospacer in full_around:
                ps_seq = other_protospacer
                ps_strand = 1
            else:
                ps_seq =  utilities.reverse_complement(other_protospacer)
                if ps_seq not in full_around:
                    results['failed'] = f'protospacer {other_protospacer} not present near protospacer {protospacer}'
                    return results
                ps_strand = -1

            ps_start = full_around.index(ps_seq)
            protospacer_locations.append((ps_seq, ps_start, ps_strand))

        for ps_seq, ps_start, ps_strand in protospacer_locations:
            if ps_strand == 1:
                PAM_offset = len(protospacer)
                PAM_transform = utilities.identity
            else:
                PAM_offset = -3
                PAM_transform = utilities.reverse_complement

            PAM_start = ps_start + PAM_offset
            PAM = PAM_transform(full_around[PAM_start:PAM_start + 3])
            pattern, *matches = Bio.SeqUtils.nt_search(PAM, 'NGG')

            if 0 not in matches:
                # Note: this could incorrectly fail if there are multiple exact matches for an other_protospacer
                # in full_around.
                results['failed'] = f'bad PAM: {PAM} next to {ps_seq} (strand {ps_strand})'
                return results

        if amplicon_primers[0] in full_around:
            final_fwd_primer = amplicon_primers[0]
            final_rev_primer = utilities.reverse_complement(amplicon_primers[1])
            if final_rev_primer not in full_around:
                results['failed'] = f'primer {amplicon_primers[1]} not present near protospacer'
                return results
        else:
            final_fwd_primer = amplicon_primers[1]
            final_rev_primer = utilities.reverse_complement(amplicon_primers[0])

            if final_fwd_primer not in full_around:
                results['failed'] = f'primer {amplicon_primers[1]} not present near protospacer'
                return results

            if final_rev_primer not in full_around:
                results['failed'] = f'primer {amplicon_primers[0]} not present near protospacer'
                return results

        fwd_start = full_around.index(final_fwd_primer)
        rev_start = full_around.index(final_rev_primer)
        
        if fwd_start >= rev_start:
            results['failed'] = f'primer don\'t flank protospacer'
            return results
        
        final_window_around = 500    

        offset = fwd_start - final_window_around

        final_start = fwd_start - final_window_around
        final_end = rev_start + len(final_rev_primer) + final_window_around

        target_seq = full_around[final_start:final_end]

        colors = {
            'HA_1': '#c7b0e3',
            'HA_2': '#85dae9',
            'forward_primer': '#ff9ccd',
            'reverse_primer': '#9eafd2',
            'sgRNA': '#c6c9d1',
            'donor_specific': '#b1ff67',
        }
        
        if has_donor:
            # Identify the homology arms.

            required_match_length = 15
            donor_prefix = donor_seq[:required_match_length]
            
            if donor_prefix in target_seq:
                possibly_flipped_donor_seq = donor_seq
            elif utilities.reverse_complement(donor_prefix) in target_seq:
                possibly_flipped_donor_seq = utilities.reverse_complement(donor_seq)
                donor_prefix = possibly_flipped_donor_seq[:required_match_length]
            else:
                results['failed'] = f'beginning of donor sequence ({donor_prefix}) not present near protospacer'
                return results

            donor_suffix = possibly_flipped_donor_seq[-required_match_length:]

            try:
                donor_start = target_seq.index(donor_prefix)
                donor_end = target_seq.index(donor_suffix) + len(donor_suffix)
            except ValueError:
                results['failed'] = f'end of donor sequence ({donor_suffix}) not present near protospacer'
                return results

            relevant_target_seq = target_seq[donor_start:donor_end]

            total_HA_length = donor_end - donor_start

            mismatches_before_deletion = np.cumsum([t != d for t, d in zip(relevant_target_seq, possibly_flipped_donor_seq)])

            flipped_target = relevant_target_seq[::-1]
            flipped_donor = possibly_flipped_donor_seq[::-1]
            mismatches_after_deletion = np.cumsum([0] + [t != d for t, d in zip(flipped_target, flipped_donor)][:-1])[::-1]

            total_mismatches = mismatches_before_deletion + mismatches_after_deletion

            last_index_in_HA_1 = int(np.argmin(total_mismatches))

            lengths = {}
            lengths['HA_1'] = last_index_in_HA_1 + 1
            lengths['HA_2'] = total_HA_length - lengths['HA_1']
            lengths['donor_specific'] = len(donor_seq) - total_HA_length
            
            # Support for donors that have fixed sequences outside of their homology arms added by PCR.
            if info.get('donor_primers') is not None:
                donor_primers = info['donor_primers'].split(';')
                
                suffix_nts = 15

                forward_suffix = donor_primers[0][-suffix_nts:]
                
                if forward_suffix in possibly_flipped_donor_seq:
                    donor_fwd = donor_primers[0]
                    donor_rev = utilities.reverse_complement(donor_primers[1])

                else:
                    forward_suffix = donor_primers[1][-suffix_nts:]

                    if forward_suffix in possibly_flipped_donor_seq:
                        donor_fwd = donor_primers[1]
                        donor_rev = utilities.reverse_complement(donor_primers[0])

                    else:
                        results['failed'] = f'at least one donor primer doesn\'t end in at least {suffix_nts} nts that are present in the donor'
                        return results

                forward_suffix_start = possibly_flipped_donor_seq.index(forward_suffix)
                extra_nts_at_start = len(donor_fwd[:-suffix_nts]) - forward_suffix_start

                reverse_suffix = donor_rev[:suffix_nts]

                try:
                    reverse_suffix_start = possibly_flipped_donor_seq.index(reverse_suffix)
                except ValueError:
                    results['failed'] = f'at least one donor primer doesn\'t end in at least {suffix_nts} nts that are present in the donor'
                    return results

                possibly_flipped_donor_seq = donor_fwd[:-suffix_nts] + possibly_flipped_donor_seq[forward_suffix_start:reverse_suffix_start] + donor_rev
            else:
                extra_nts_at_start = 0
            
            starts = {
                'HA_1': extra_nts_at_start + 0,
                'donor_specific': extra_nts_at_start + lengths['HA_1'],
                'HA_2': extra_nts_at_start + len(donor_seq) - lengths['HA_2'], # note: donor_seq is still original
            }
        
            donor_features = [
                SeqFeature(location=FeatureLocation(starts['HA_1'], starts['HA_1'] + lengths['HA_1'], strand=1),
                        id='HA_1',
                        type='misc_feature',
                        qualifiers={'label': 'HA_1',
                                    'ApEinfo_fwdcolor': colors['HA_1'],
                                    },
                        ),
                SeqFeature(location=FeatureLocation(starts['donor_specific'], starts['donor_specific'] + lengths['donor_specific'], strand=1),
                        id='donor_specific',
                        type='misc_feature',
                        qualifiers={'label': 'donor_specific',
                                    'ApEinfo_fwdcolor': colors['donor_specific'],
                                    },
                        ),
                SeqFeature(location=FeatureLocation(starts['HA_2'], starts['HA_2'] + lengths['HA_2'], strand=1),
                        id='HA_2',
                        type='misc_feature',
                        qualifiers={'label': 'HA_2',
                                    'ApEinfo_fwdcolor': colors['HA_2'],
                                    },
                        ),
            ]

            results['donor'] = Bio.SeqRecord.SeqRecord(Seq(possibly_flipped_donor_seq, generic_dna), name=donor_name, features=donor_features)

            target_features = [
                SeqFeature(location=FeatureLocation(donor_start, donor_start + lengths['HA_1'], strand=1),
                        id='HA_1',
                        type='misc_feature',
                        qualifiers={'label': 'HA_1',
                                    'ApEinfo_fwdcolor': colors['HA_1'],
                                    },
                        ),
                SeqFeature(location=FeatureLocation(donor_end - lengths['HA_2'], donor_end, strand=1),
                        id='HA_2',
                        type='misc_feature',
                        qualifiers={'label': 'HA_2',
                                    'ApEinfo_fwdcolor': colors['HA_2'],
                                    },
                        ),
            ]
        
        else:
            results['donor'] = None
            target_features = []

        target_features.extend([
            SeqFeature(location=FeatureLocation(fwd_start - offset, fwd_start - offset + len(final_fwd_primer), strand=1),
                       id='forward_primer',
                       type='misc_feature',
                       qualifiers={'label': 'forward_primer',
                                   'ApEinfo_fwdcolor': colors['forward_primer'],
                                  },
                      ),
            SeqFeature(location=FeatureLocation(rev_start - offset, rev_start - offset + len(final_rev_primer), strand=-1),
                       id='reverse_primer',
                       type='misc_feature',
                       qualifiers={'label': 'reverse_primer',
                                   'ApEinfo_fwdcolor': colors['reverse_primer'],
                                  },
                      ),
        ])

        for sgRNA_i, (ps_seq, ps_start, ps_strand) in enumerate(protospacer_locations):
            sgRNA_feature = SeqFeature(location=FeatureLocation(ps_start - offset, ps_start - offset + len(protospacer), strand=ps_strand),
                                       id=f'sgRNA_{sgRNA_i}',
                                       type='sgRNA_SpCas9',
                                       qualifiers={'label': f'sgRNA_{sgRNA_i}',
                                                   'ApEinfo_fwdcolor': colors['sgRNA'],
                                                   },
                                       )
            target_features.append(sgRNA_feature)

        results['target'] = SeqRecord(Seq(target_seq, generic_dna), name=name, features=target_features)

        return results
    
    good_candidates = []
    bad_candidates = []
    
    for al in perfect_als:
        results = evaluate_candidate(al)
        if 'failed' in results:
            bad_candidates.append(results)
        else:
            good_candidates.append(results)
    
    if len(good_candidates) == 0:
        if len(bad_candidates) == 0:
            print(f'Error building {name}: no perfect matches to sgRNA {protospacer} found in {genome}')
            return 

        else:
            print(f'Error building {name}: no valid genomic locations for {name}')

            for results in bad_candidates:
                print(f'\t{results["location"]}: {results["failed"]}')

            return 

    elif len(good_candidates) > 1:
        print(f'Warning: multiple valid genomic locations for {name}:')
        for results in good_candidates:
            print(f'\t{results["location"]}')
        best_candidate = good_candidates[0]
        print(f'Arbitrarily choosing {best_candidate["location"]}')
    else:
        best_candidate = good_candidates[0]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=BiopythonWarning)

        try:
            Bio.SeqIO.write(best_candidate['target'], target_gb_fn, 'genbank')

            if has_donor:
                Bio.SeqIO.write(best_candidate['donor'], donor_gb_fn, 'genbank')
        except ValueError:
            # locus line too long, can't write genbank file with BioPython
            pass

    manifest_fn = target_dir / 'manifest.yaml'
    manifest = {
        'sources': [name, donor_name] if has_donor else [name],
        'target': name,
    }
    if has_donor:
        manifest['donor_specific'] = 'donor_specific'

    manifest_fn.write_text(yaml.dump(manifest, default_flow_style=False))
        
    gb_records = [best_candidate['target']]
    if has_donor:
        gb_records.append(best_candidate['donor'])

    ti = TargetInfo(base_dir, name, gb_records=gb_records)
    ti.make_references()
    ti.identify_degenerate_indels()

def build_target_infos_from_csv(base_dir, genome='hg19'):
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'targets.csv'

    index_locations = locate_supplemental_indices(base_dir)
    if genome not in index_locations:
        print(f'Error: can\'t locate indices for {genome}')
        sys.exit(0)

    df = pd.read_csv(csv_fn).replace({np.nan: None})

    for _, row in df.iterrows():
        name = row['name']
        print(f'Building {name}...')
        build_target_info(base_dir, row, genome)

def build_indices(base_dir, name, num_threads=1):
    base_dir = Path(base_dir)

    print(f'Building indices for {name}')
    fasta_dir = base_dir / 'indices' / name / 'fasta'

    fasta_fns = genomes.get_all_fasta_file_names(fasta_dir)
    if len(fasta_fns) == 0:
        raise ValueError(f'No fasta files found in {fasta_dir}')
    elif len(fasta_fns) > 1:
        raise ValueError(f'Can only build minimap2 index from a single fasta file')

    print('Indexing fastas...')
    genomes.make_fais(fasta_dir)

    minimap2_dir = base_dir / 'indices' / name / 'minimap2'
    minimap2_dir.mkdir(exist_ok=True)

    fasta_fn = fasta_fns[0]

    print('Building STAR index...')
    STAR_dir = base_dir / 'indices' / name / 'STAR'
    STAR_dir.mkdir(exist_ok=True)
    mapping_tools.build_STAR_index([fasta_fn], STAR_dir, num_threads=num_threads)

    print('Building minimap2 index...')
    minimap2_index_fn = minimap2_dir / f'{name}.mmi'
    mapping_tools.build_minimap2_index(fasta_fn, minimap2_index_fn)

def download_and_build_hg38(base_dir, num_threads=8):
    base_dir = Path(base_dir)
    hg38_dir = base_dir / 'indices' / 'hg38'
    fasta_dir = hg38_dir / 'fasta'

    print('Downloading hg38 from ucsc...')
    wget_command = [
        'wget', 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
        '-P', str(fasta_dir),
    ]
    subprocess.run(wget_command, check=True)

    print('Uncompressing...')
    gunzip_command = [
        'gunzip',  str(fasta_dir / 'hg38.fa.gz'),
    ]
    subprocess.run(gunzip_command, check=True)

    build_indices(base_dir, 'hg38', num_threads=num_threads)

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