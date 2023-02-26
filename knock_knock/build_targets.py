import logging
import shutil
import subprocess
import sys
import warnings

from urllib.parse import urlparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import pysam
import yaml

import Bio.SeqIO
import Bio.SeqUtils
from Bio import BiopythonWarning
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from hits import fastq, genomes, interval, mapping_tools, sam, sw, utilities
from knock_knock import target_info, pegRNAs

def identify_homology_arms(donor_seq, donor_type, target_seq, cut_after, required_match_length=15):
    header = pysam.AlignmentHeader.from_references(['donor', 'target'], [len(donor_seq), len(target_seq)])
    mapper = sw.SeedAndExtender(donor_seq.encode(), 8, header, 'donor')
    
    target_bytes = target_seq.encode()
    
    alignments = {
        'before_cut': [],
        'after_cut': [],
    }

    seed_starts = {
        'before_cut': range(cut_after - required_match_length, 0, -1),
        'after_cut': range(cut_after, len(target_seq) - required_match_length),
    }

    for side in ['before_cut', 'after_cut']:
        for seed_start in seed_starts[side]:  
            alignments[side] = mapper.seed_and_extend(target_bytes, seed_start, seed_start + required_match_length, 'target')
            if alignments[side]:
                break

        else:
            results = {'failed': f'cannot locate homology arm on {side}'}
            return results
        
    possible_HA_boundaries = []
    
    for before_al in alignments['before_cut']:
        for after_al in alignments['after_cut']:
            if sam.get_strand(before_al) == sam.get_strand(after_al):
                strand = sam.get_strand(before_al)
                if strand == '+':
                    if before_al.reference_end < after_al.reference_start:
                        possible_HA_boundaries.append((donor_seq, before_al.reference_start, after_al.reference_end))
                elif strand == '-':
                    if before_al.reference_start > after_al.reference_end:
                        flipped_seq = utilities.reverse_complement(donor_seq)
                        start = len(donor_seq) - 1 - (before_al.reference_end - 1)
                        end = len(donor_seq) - 1 - after_al.reference_start + 1
                        possible_HA_boundaries.append((flipped_seq, start, end))

    possible_HAs = []
    for possibly_flipped_donor_seq, HA_start, HA_end in possible_HA_boundaries:
        donor_window = possibly_flipped_donor_seq[HA_start:HA_end]

        donor_prefix = donor_window[:required_match_length]

        donor_suffix = donor_window[-required_match_length:]

        # Try to be resilient against multiple occurrence of HA substrings in the target
        # by prioritizing matches closest to the cut site.
        target_HA_start = target_seq.rfind(donor_prefix, 0, cut_after + required_match_length)
        target_HA_end = target_seq.find(donor_suffix, cut_after - required_match_length) + len(donor_suffix)

        if target_HA_start == -1 or target_HA_end == -1 or target_HA_start >= target_HA_end:
            results = {'failed': f'cannot locate homology arms in target'}
            return results

        relevant_target_seq = target_seq[target_HA_start:target_HA_end]

        total_HA_length = target_HA_end - target_HA_start

        mismatches_before_deletion = np.cumsum([t != d for t, d in zip(relevant_target_seq, donor_window)])

        flipped_target = relevant_target_seq[::-1]
        flipped_donor = donor_window[::-1]
        mismatches_after_deletion = np.cumsum([0] + [t != d for t, d in zip(flipped_target, flipped_donor)][:-1])[::-1]

        total_mismatches = mismatches_before_deletion + mismatches_after_deletion

        last_index_in_HA_1 = int(np.argmin(total_mismatches))
        min_mismatches = total_mismatches[last_index_in_HA_1]

        lengths = {}
        lengths['HA_1'] = last_index_in_HA_1 + 1
        lengths['HA_2'] = total_HA_length - lengths['HA_1']
        lengths['donor_specific'] = len(donor_seq) - total_HA_length
        
        info = {
            'min_mismatches': min_mismatches,
            'possibly_flipped_donor_seq': possibly_flipped_donor_seq,
            'donor_HA_start': HA_start,
            'donor_HA_end': HA_end,
            'target_HA_start': target_HA_start,
            'target_HA_end': target_HA_end,
            'lengths': lengths,
        }
        possible_HAs.append((info))
        
    def priority(info):
        return info['min_mismatches'], -min(info['lengths']['HA_1'], info['lengths']['HA_2'])

    if not possible_HAs:
        results = {'failed': 'cannot locate homology arms'}
    else:
        results = min(possible_HAs, key=priority)

    lengths = results['lengths']

    donor_starts = {
        'HA_1': results['donor_HA_start'],
        'donor_specific': results['donor_HA_start'] + lengths['HA_1'],
        'HA_2': results['donor_HA_end'] - lengths['HA_2'],
    }
    donor_ends = {
        'HA_1': donor_starts['HA_1'] + lengths['HA_1'],
        'donor_specific': donor_starts['HA_2'],
        'HA_2': donor_starts['HA_2'] + lengths['HA_2'],
    }

    if donor_type == 'PCR':
        if donor_starts['HA_1'] != 0:
            donor_starts['PCR_adapter_1'] = 0
            donor_ends['PCR_adapter_1'] = donor_starts['HA_1']

        if donor_ends['HA_2'] != len(donor_seq):
            donor_starts['PCR_adapter_2'] = donor_ends['HA_2']
            donor_ends['PCR_adapter_2'] = len(donor_seq)

    target_starts = {
        'HA_1': results['target_HA_start'],
        'HA_2': results['target_HA_end'] - lengths['HA_2'],
    }
    target_ends = {key: target_starts[key] + lengths[key] for key in target_starts}

    donor_strand = 1
    target_strand = 1

    donor_features = [
        SeqFeature(location=FeatureLocation(donor_starts[feature_name], donor_ends[feature_name], strand=donor_strand),
                    id=feature_name,
                    type='misc_feature',
                    qualifiers={'label': feature_name,
                                'ApEinfo_fwdcolor': feature_colors[feature_name],
                               },
                  )
        for feature_name in donor_starts
    ]

    target_features = ([
        SeqFeature(location=FeatureLocation(target_starts[feature_name], target_ends[feature_name], strand=target_strand),
                    id=feature_name,
                    type='misc_feature',
                    qualifiers={'label': feature_name,
                                'ApEinfo_fwdcolor': feature_colors[feature_name],
                               },
                  )
        for feature_name in target_starts
    ])

    HA_info = {
        'possibly_flipped_donor_seq': results['possibly_flipped_donor_seq'],
        'donor_features': donor_features,
        'target_features': target_features,
    }

    return HA_info

feature_colors = {
    'HA_1': '#c7b0e3',
    'HA_RT': '#c7b0e3',
    'RTT': '#c7b0e3',
    'HA_2': '#85dae9',
    'HA_PBS': '#85dae9',
    'PBS': '#85dae9',
    'forward_primer': '#75C6A9',
    'reverse_primer': '#9eafd2',
    'sgRNA': '#c6c9d1',
    'donor_specific': '#b1ff67',
    'PCR_adapter_1': '#F8D3A9',
    'PCR_adapter_2': '#D59687',
    'protospacer': '#ff9ccd',
    'scaffold': '#b7e6d7',
}

def build_target_info(base_dir, info, all_index_locations,
                      defer_HA_identification=False,
                      offtargets=False,
                     ):
    ''' 
    Attempts to identify the genomic location where an sgRNA sequence
    is flanked by amplicon primers.
    
    info should have keys:
        genome
        amplicon_primers
        sgRNAs
    optional keys:
        donor_sequence
        nonhomologous_donor_sequence
        extra_sequences
    '''
    genome = info['genome']
    if info['genome'] not in all_index_locations:
        print(f'Error: can\'t locate indices for {genome}')
        sys.exit(1)
    else:
        index_locations = all_index_locations[genome]

    base_dir = Path(base_dir)

    ti_name = info['name']

    donor_info = info.get('donor_sequence')
    if donor_info is None:
        donor_name = None
        donor_seq = None
    else:
        donor_name, donor_seq = donor_info
        if donor_name is None:
            donor_name = f'{ti_name}_donor'

    if donor_seq is None:
        has_donor = False
    else:
        has_donor = True

    if info['donor_type'] is None:
        donor_type = None
    else:
        _, donor_type = info['donor_type']

    nh_donor_info = info.get('nonhomologous_donor_sequence')
    if nh_donor_info is None:
        nh_donor_name = None
        nh_donor_seq = None
    else:
        nh_donor_name, nh_donor_seq = nh_donor_info
        if nh_donor_name is None:
            nh_donor_name = f'{ti_name}_NH_donor'

    if nh_donor_seq is None:
        has_nh_donor = False
    else:
        has_nh_donor = True

    primers_name, primers = info['amplicon_primers']
    primers = primers.split(';')

    target_dir = base_dir / 'targets' / ti_name
    target_dir.mkdir(parents=True, exist_ok=True)

    primers_dir = target_dir / 'primer_alignment'
    primers_dir.mkdir(exist_ok=True)

    # Align the primers to the reference genome.

    fastq_fn = primers_dir / 'primers.fastq'

    with fastq_fn.open('w') as fh:
        for primer_i, primer_seq in enumerate(primers):
            quals = fastq.encode_sanger([40]*len(primer_seq))
            read = fastq.Read(f'primer_{primer_i}', primer_seq, quals)
            fh.write(str(read))

    STAR_prefix = primers_dir / 'primers_'
    bam_fn = primers_dir / 'primers.bam'

    mapping_tools.map_STAR(fastq_fn,
                           index_locations['STAR'],
                           STAR_prefix,
                           mode='permissive',
                           bam_fn=bam_fn,
                          )

    if primers_name is None:
        target_name = ti_name
    else:
        target_name = primers_name

    primer_alignments = defaultdict(list)

    # Retain only alignments that include the 3' end of the primer.

    with pysam.AlignmentFile(bam_fn, 'rb') as bam_fh:
        for al in bam_fh:
            covered = interval.get_covered(al)
            if len(al.query_sequence) - 1 in covered:
                primer_alignments[al.query_name].append(al)
                
    if len(primer_alignments) != 2:
        raise ValueError

    # Find pairs of alignments that are on the same chromosome and point towards each other.

    def in_correct_orientation(first_al, second_al):
        ''' Check if first_al and second_al point towards each other on the same chromosome. '''
        
        if first_al.reference_name != second_al.reference_name:
            return False
        
        first_interval = interval.get_covered_on_ref(first_al)
        second_interval = interval.get_covered_on_ref(second_al)
        if interval.are_overlapping(first_interval, second_interval):
            return False
        
        left_al, right_al = sorted([first_al, second_al], key=lambda al: al.reference_start)
        
        if sam.get_strand(left_al) != '+' or sam.get_strand(right_al) != '-':
            return False
        else:
            return left_al, right_al

    correct_orientation_pairs = []

    primer_names = sorted(primer_alignments)

    for first_al in primer_alignments[primer_names[0]]:
        for second_al in primer_alignments[primer_names[1]]:
            if in_correct_orientation(first_al, second_al):
                correct_orientation_pairs.append(in_correct_orientation(first_al, second_al))
                
    if len(correct_orientation_pairs) == 0:
        raise ValueError

    # Rank pairs in the correct orientation by (shortest) amplicon length.

    def amplicon_length(left_al, right_al):
        left = left_al.reference_start
        right = right_al.reference_end
        return right - left

    correct_orientation_pairs = sorted(correct_orientation_pairs, key=lambda pair: amplicon_length(*pair))

    if len(correct_orientation_pairs) > 1:
        logging.warning(f'Multiple primer alignments found.')
        for left_al, right_al in correct_orientation_pairs:
            logging.warning(f'Found {amplicon_length(left_al, right_al):,} nt amplicon on {left_al.reference_name} from {left_al.reference_start:,} to {right_al.reference_end - 1:,}')
        
    left_al, right_al = correct_orientation_pairs[0]

    region_fetcher = genomes.build_region_fetcher(index_locations['fasta'])

    ref_name = left_al.reference_name
    amplicon_start = left_al.reference_start
    amplicon_end = right_al.reference_end
    amplicon_sequence = region_fetcher(ref_name, amplicon_start, amplicon_end).upper()

    protospacer_features_in_amplicon = {}
    for sgRNA_name, components in sorted(info['sgRNAs']):
        # Note: identify_protospacer_in_target will raise an exception on failure.
        protospacer_feature = pegRNAs.identify_protospacer_in_target(amplicon_sequence, components['protospacer'], components['effector'])
        protospacer_features_in_amplicon[sgRNA_name] = protospacer_feature

    final_window_around = 500

    def amplicon_coords_to_target_coords(p):
        return p + final_window_around

    def genomic_coords_to_target_coords(p):
        return p - amplicon_start + final_window_around

    convert_strand = {
        '+': 1,
        '-': -1,
    }

    target_sequence = region_fetcher(ref_name, amplicon_start - final_window_around, amplicon_end + final_window_around).upper()
    
    left_primer_location = FeatureLocation(genomic_coords_to_target_coords(left_al.reference_start),
                                           genomic_coords_to_target_coords(left_al.reference_end),
                                           strand=convert_strand['+'],
                                          )

    right_primer_location = FeatureLocation(genomic_coords_to_target_coords(right_al.reference_start),
                                            genomic_coords_to_target_coords(right_al.reference_end),
                                            strand=convert_strand['-'],
                                           )
    
    target_features = [
        SeqFeature(location=left_primer_location,
                    id='forward_primer',
                    type='misc_feature',
                    qualifiers={
                        'label': 'forward_primer',
                        'ApEinfo_fwdcolor': feature_colors['forward_primer'],
                    },
                   ),
        SeqFeature(location=right_primer_location,
                    id='reverse_primer',
                    type='misc_feature',
                    qualifiers={
                        'label': 'reverse_primer',
                        'ApEinfo_fwdcolor': feature_colors['reverse_primer'],
                    },
                   ),
        SeqFeature(location=left_primer_location,
                   id='anchor',
                   type='misc_feature',
                    qualifiers={
                        'label': 'anchor',
                    },
                  ),
    ]

    # Note: the primer listed first is assumed to correspond to the expected
    # start of sequencing reads.
    if left_al.query_name == 'primer_0':
        sequencing_start_feature_name = 'forward_primer'
    else:
        sequencing_start_feature_name = 'reverse_primer'

    protospacer_features = []
    for sgRNA_name, feature in protospacer_features_in_amplicon.items():
        location = FeatureLocation(amplicon_coords_to_target_coords(feature.start),
                                   amplicon_coords_to_target_coords(feature.end), 
                                   strand=convert_strand[feature.strand],
                                  )
        protospacer_feature = SeqFeature(location=location,
                                         id=sgRNA_name,
                                         type=f'protospacer',
                                         qualifiers={
                                            'label': sgRNA_name,
                                            'ApEinfo_fwdcolor': feature_colors['sgRNA'],
                                         },
                                        )
        protospacer_features.append(protospacer_feature)

    gb_records = {}

    if has_donor:
        if not defer_HA_identification:
            if len(info['sgRNAs']) > 1:
                raise ValueError

            sgRNA_name, components = info['sgRNAs'][0]

            protospacer_feature = protospacer_features[0]
            effector = target_info.effectors[components['effector']]

            # TODO: untested code branch here
            cut_after_offset = [offset for offset in effector.cut_after_offset if offset is not None][0]

            if protospacer_feature.strand == 1:
                # protospacer_feature.end is the first nt of the PAM
                cut_after = protospacer_feature.location.end + cut_after_offset
            else:
                # protospacer_feature.start - 1 is the first nt of the PAM
                cut_after = protospacer_feature.location.start - 1 - cut_after_offset - 1

            HA_info = identify_homology_arms(donor_seq, donor_type, target_sequence, cut_after)

            if 'failed' in HA_info:
                raise ValueError

            donor_Seq = Seq(HA_info['possibly_flipped_donor_seq'])
            donor_features = HA_info['donor_features']
            target_features.extend(HA_info['target_features'])

        else:
            donor_Seq = Seq(donor_seq)
            donor_features = []

        donor_record = SeqRecord(donor_Seq,
                                 name=donor_name,
                                 features=donor_features,
                                 annotations={
                                    'molecule_type': 'DNA',
                                 },
                                )

        gb_records[donor_name] = donor_record

    sgRNAs_with_extensions = [(name, components) for name, components in info['sgRNAs'] if components['extension'] != '']
    if len(sgRNAs_with_extensions) > 0:
        for name, components in sgRNAs_with_extensions:
            pegRNA_features, new_target_features = pegRNAs.infer_features(name, components, target_name, target_sequence)

            pegRNA_SeqFeatures = [
                SeqFeature(id=feature_name,
                           location=FeatureLocation(feature.start, feature.end + 1, strand=convert_strand[feature.strand]),
                           type='misc_feature',
                           qualifiers={
                               'label': feature_name,
                               'ApEinfo_fwdcolor': feature.attribute['color'],
                           },
                          )
                for (_, feature_name), feature in pegRNA_features.items()
            ]

            pegRNA_Seq = Seq(components['full_sequence'])
            pegRNA_record = SeqRecord(pegRNA_Seq,
                                      name=name,
                                      features=pegRNA_SeqFeatures,
                                      annotations={
                                        'molecule_type': 'DNA',
                                      },
                                     )

            gb_records[name] = pegRNA_record
        
    if has_nh_donor:
        nh_donor_Seq = Seq(nh_donor_seq)
        nh_donor_record = SeqRecord(nh_donor_Seq, name=nh_donor_name, annotations={'molecule_type': 'DNA'})
        gb_records[nh_donor_name] = nh_donor_record

    target_Seq = Seq(target_sequence)
    target_record = SeqRecord(target_Seq,
                              name=target_name,
                              features=target_features,
                              annotations={
                                'molecule_type': 'DNA',
                              },
                             )
    gb_records[target_name] = target_record

    if info.get('extra_sequences') is not None:
        for extra_seq_name, extra_seq in info['extra_sequences']:
            record = SeqRecord(extra_seq, name=extra_seq_name, annotations={'molecule_type': 'DNA'})
            gb_records[extra_seq_name] = record

    if info.get('extra_genbanks') is not None:
        for gb_fn in info['extra_genbanks']:
            full_gb_fn = base_dir / 'targets' / gb_fn

            if not full_gb_fn.exists():
                raise ValueError(f'{full_gb_fn} does not exist')

            for record in Bio.SeqIO.parse(full_gb_fn, 'genbank'):
                gb_records[record.name] = record

    if len(sgRNAs_with_extensions) > 0:
        # Note: for debugging convenience, genbank files are written for pegRNAs,
        # but these are NOT supplied as genbank records to make the final TargetInfo,
        # since relevant features are either represented by the intial decomposition into
        # components or inferred on instantiation of the TargetInfo.
        pegRNA_names = [name for name, components in sgRNAs_with_extensions]
        non_pegRNA_records = {name: record for name, record in gb_records.items() if name not in pegRNA_names}
        gb_records_for_manifest = non_pegRNA_records
    else:
        gb_records_for_manifest = gb_records

    truncated_name_i = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=BiopythonWarning)

        for which_seq, record in gb_records.items(): 
            gb_fn = target_dir / f'{which_seq}.gb'
            try:
                Bio.SeqIO.write(record, gb_fn, 'genbank')
            except ValueError:
                # locus line too long, can't write genbank file with BioPython
                old_name = record.name

                truncated_name = f'{record.name[:11]}_{truncated_name_i}'
                record.name = truncated_name
                Bio.SeqIO.write(record, gb_fn, 'genbank')

                record.name = old_name

                truncated_name_i += 1

    manifest_fn = target_dir / 'manifest.yaml'

    sources = sorted(gb_records_for_manifest)
        
    manifest = {
        'sources': sources,
        'target': target_name,
        'sequencing_start_feature_name': sequencing_start_feature_name,
    }

    if has_donor:
        manifest['donor'] = donor_name
        manifest['donor_specific'] = 'donor_specific'
        if donor_type is not None:
            manifest['donor_type'] = donor_type

    if has_nh_donor:
        manifest['nonhomologous_donor'] = nh_donor_name

    manifest['features_to_show'] = [
        [target_name, 'forward_primer'],
        [target_name, 'reverse_primer'],
    ]

    if has_donor:
        manifest['features_to_show'].extend([
            [donor_name, 'HA_1'],
            [donor_name, 'HA_2'],
            [donor_name, 'donor_specific'],
            [donor_name, 'PCR_adapter_1'],
            [donor_name, 'PCR_adapter_2'],
            [target_name, 'HA_1'],
            [target_name, 'HA_2'],
        ])

    manifest['genome_source'] = genome

    manifest_fn.write_text(yaml.dump(manifest, default_flow_style=False))

    gb_records = list(gb_records_for_manifest.values())

    ti = target_info.TargetInfo(base_dir, ti_name, gb_records=gb_records)

    sgRNAs_df = load_sgRNAs(base_dir, process=False)
    sgRNA_names = sorted([name for name, _ in info['sgRNAs']])
    sgRNAs_df.loc[sgRNA_names].to_csv(ti.fns['sgRNAs'])

    ti.make_protospacer_fastas()
    ti.map_protospacers(genome)

    shutil.rmtree(primers_dir)

def load_sgRNAs(base_dir, process=True):
    '''
    If process == False, just pass along the DataFrame for subsetting.
    '''
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'sgRNAs.csv'

    if not csv_fn.exists():
        return None
    else:
        return pegRNAs.read_csv(csv_fn, process=process)

def build_component_registry(base_dir):
    registry = {}

    registry['sgRNAs'] = load_sgRNAs(base_dir)

    amplicon_primers_fn = base_dir / 'targets' / 'amplicon_primers.csv'

    if amplicon_primers_fn.exists():
        registry['amplicon_primers'] = pd.read_csv(amplicon_primers_fn, index_col='name').squeeze('columns')
    else:
        registry['amplicon_primers'] = {}

    extra_sequences_fn = base_dir / 'targets' / 'extra_sequences.csv'
    if extra_sequences_fn.exists():
        registry['extra_sequence'] = pd.read_csv(extra_sequences_fn, index_col='name').squeeze('columns')
    else:
        registry['extra_sequence'] = {}

    donors_fn = base_dir / 'targets' / 'donors.csv'

    if donors_fn.exists():
        donors = pd.read_csv(donors_fn, index_col='name')
        registry['donor_sequence'] = donors['donor_sequence']
        registry['donor_type'] = donors['donor_type']
    else:
        registry['donor_sequence'] = {}
        registry['donor_type'] = {}

    return registry

def build_target_infos_from_csv(base_dir, offtargets=False, defer_HA_identification=False):
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'targets.csv'

    indices = target_info.locate_supplemental_indices(base_dir)

    targets_df = pd.read_csv(csv_fn, comment='#', index_col='name').replace({np.nan: None})

    registry = build_component_registry(base_dir)

    def lookup(row, column_to_lookup, registry_column, validate_sequence=True, multiple_lookups=False):
        value_to_lookup = row.get(column_to_lookup)
        if value_to_lookup is None:
            return None

        if multiple_lookups:
            values_to_lookup = value_to_lookup.split(';')
        else:
            values_to_lookup = [value_to_lookup]

        registered_values = registry[registry_column]
        valid_chars = set('TCAGN;')

        looked_up = []
        for value_to_lookup in values_to_lookup:
            if value_to_lookup in registered_values:
                value_name = value_to_lookup
                seq = registered_values[value_to_lookup]
                possible_error_message = f'invalid char in {row.name} {column_to_lookup} registry entry {value_to_lookup}\n{seq}'
            else:
                value_name = None
                seq = value_to_lookup
                possible_error_message = f'Error: {row.name} value for {column_to_lookup} ({seq}) is not a registered name but also doesn\'t look like a valid sequence.\nRegistered names: {registered_values}'

            if seq is not None and validate_sequence:
                seq = seq.upper()
                invalid_chars = set(seq) - valid_chars
                if invalid_chars:
                    print(possible_error_message)
                    print(f'Valid sequence characters are {valid_chars}; {seq} contains {invalid_chars}')
                    sys.exit(1)

            looked_up.append((value_name, seq))

        if not multiple_lookups:
            looked_up = looked_up[0]
            
        return looked_up

    for target_name, row in targets_df.iterrows():
        info = {
            'name': target_name,
            'genome': row['genome'],
            'amplicon_primers': lookup(row, 'amplicon_primers', 'amplicon_primers'),
            'sgRNAs': lookup(row, 'sgRNAs', 'sgRNAs', multiple_lookups=True, validate_sequence=False),
            'donor_sequence': lookup(row, 'donor_sequence', 'donor_sequence'),
            'extra_sequences': lookup(row, 'extra_sequences', 'extra_sequence', multiple_lookups=True),
            'nonhomologous_donor_sequence': lookup(row, 'nonhomologous_donor_sequence', 'donor_sequence'),
            'donor_type': lookup(row, 'donor_sequence', 'donor_type', validate_sequence=False),
        }

        for sgRNA_name, sgRNA_components in info['sgRNAs']:
            if sgRNA_name is None:
                # Because of how lookup works, sgRNA_components will hold value of 
                # name that wasn't found.
                raise ValueError(f'{sgRNA_components} not found')

        if row.get('extra_genbanks') is not None:
            info['extra_genbanks'] = row['extra_genbanks'].split(';')

        logging.info(f'Building {target_name}...')

        build_target_info(base_dir, info, indices,
                          offtargets=offtargets,
                          defer_HA_identification=defer_HA_identification,
                         )

def build_indices(base_dir, name, num_threads=1, **STAR_index_kwargs):
    base_dir = Path(base_dir)

    logging.info(f'Building indices for {name}')
    fasta_dir = base_dir / 'indices' / name / 'fasta'

    fasta_fns = genomes.get_all_fasta_file_names(fasta_dir)
    if len(fasta_fns) == 0:
        raise ValueError(f'No fasta files found in {fasta_dir}')
    elif len(fasta_fns) > 1:
        raise ValueError(f'Can only build minimap2 index from a single fasta file')

    logging.info('Indexing fastas...')
    genomes.make_fais(fasta_dir)

    fasta_fn = fasta_fns[0]

    logging.info('Building STAR index...')
    STAR_dir = base_dir / 'indices' / name / 'STAR'
    STAR_dir.mkdir(exist_ok=True)
    mapping_tools.build_STAR_index([fasta_fn], STAR_dir,
                                   num_threads=num_threads,
                                   RAM_limit=int(4e10),
                                   **STAR_index_kwargs,
                                  )

    logging.info('Building minimap2 index...')
    minimap2_dir = base_dir / 'indices' / name / 'minimap2'
    minimap2_dir.mkdir(exist_ok=True)
    minimap2_index_fn = minimap2_dir / f'{name}.mmi'
    mapping_tools.build_minimap2_index(fasta_fn, minimap2_index_fn)

def download_genome_and_build_indices(base_dir, genome_name, num_threads=8):
    urls = {
        'hg38': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
        'hg19': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
        'bosTau7': 'http://hgdownload.cse.ucsc.edu/goldenPath/bosTau7/bigZips/bosTau7.fa.gz',
        'mm10': 'ftp://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.toplevel.fa.gz',
        'e_coli': 'ftp://ftp.ensemblgenomes.org/pub/bacteria/release-44/fasta/bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655/dna/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.chromosome.Chromosome.fa.gz',
    }

    if genome_name not in urls:
        print(f'No URL known for {genome_name}.')
        print('Valid options are:')
        for gn in sorted(urls):
            print(f'\t- {gn}')
        sys.exit(1)

    base_dir = Path(base_dir)
    genome_dir = base_dir / 'indices' / genome_name
    fasta_dir = genome_dir / 'fasta'

    logging.info(f'Downloading {genome_name}...')

    wget_command = [
        'wget',
        '--quiet',
        urls[genome_name],
        '-P', str(fasta_dir),
    ]
    subprocess.run(wget_command, check=True)

    logging.info('Uncompressing...')

    file_name = Path(urlparse(urls[genome_name]).path).name

    gunzip_command = [
        'gunzip', '--force',  str(fasta_dir / file_name),
    ]
    subprocess.run(gunzip_command, check=True)

    if genome_name == 'e_coli':
        STAR_index_kwargs = {
            'wonky_param': 4,
        }
    else:
        STAR_index_kwargs = {}

    build_indices(base_dir, genome_name, num_threads=num_threads, **STAR_index_kwargs)

def build_manual_target(base_dir, target_name):
    target_dir = base_dir / 'targets' / target_name

    gb_fns = sorted(target_dir.glob('*.gb'))

    if len(gb_fns) != 1:
        raise ValueError

    gb_fn = gb_fns[0]

    records = list(Bio.SeqIO.parse(str(gb_fn), 'genbank'))

    if len(records) != 1:
        raise ValueError

    record = records[0]

    manifest = {
        'sources': [gb_fn.stem],
        'target': record.id,
    }

    manifest_fn = target_dir / 'manifest.yaml'

    with manifest_fn.open('w') as fh:
        fh.write(yaml.dump(manifest, default_flow_style=False))

    ti = target_info.TargetInfo(base_dir, target_name)
    ti.make_references()    
    ti.identify_degenerate_indels()