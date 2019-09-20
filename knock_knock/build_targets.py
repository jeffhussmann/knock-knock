import subprocess
import sys
import warnings
from urllib.parse import urlparse
from pathlib import Path

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
from Bio.Alphabet import generic_dna

from hits import fastq, mapping_tools, sam, genomes, utilities, sw
from knock_knock import target_info

def design_amplicon_primers_from_csv(base_dir, genome='hg19'):
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'sgRNAs.csv'

    index_locations = target_info.locate_supplemental_indices(base_dir)
    if genome not in index_locations:
        print(f'Error: can\'t locate indices for {genome}')
        sys.exit(0)

    df = pd.read_csv(csv_fn).replace({np.nan: None})

    amplicon_primer_info = {}

    for _, row in df.iterrows():
        name = row['name']
        print(f'Designing {name}...')
        best_candidate = design_amplicon_primers(base_dir, row, genome)
        if best_candidate is not None:
            amplicon_primer_info[name] = {
                'flanking_sequence': best_candidate['target_seq'],
                'genome': genome,
                'ref_name': best_candidate['ref_name'],
                'min_cut_after': best_candidate['min_cut_after'],
                'max_cut_after': best_candidate['max_cut_after'],
            }

    amplicon_primer_info = pd.DataFrame(amplicon_primer_info).T
    amplicon_primer_info.index.name = 'name'

    column_order = ['flanking_sequence', 'genome', 'ref_name', 'min_cut_after', 'max_cut_after']
    amplicon_primer_info = amplicon_primer_info[column_order]

    final_csv_fn = base_dir / 'targets' / 'sgRNAs_flanking_sequence.csv'
    amplicon_primer_info.to_csv(final_csv_fn)

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

    index_locations = target_info.locate_supplemental_indices(base_dir)
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
            'ref_name': al.reference_name,
            'cut_afters': [],
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
                cut_after = al.reference_start - full_window_around + ps_start + PAM_offset - 3
            else:
                PAM_offset = -3
                PAM_transform = utilities.reverse_complement
                cut_after = al.reference_start - full_window_around + ps_start + 2

            results['cut_afters'].append(cut_after)

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

        results['min_cut_after'] = min(results['cut_afters'])
        results['max_cut_after'] = max(results['cut_afters'])
        
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

def identify_homology_arms(donor_seq, target_seq, cut_after):
    required_match_length = 15
    
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
                        flipped = utilities.reverse_complement(donor_seq)
                        start = len(donor_seq) - 1 - (before_al.reference_end - 1)
                        end = len(donor_seq) - 1 - after_al.reference_start + 1
                        possible_HA_boundaries.append((flipped, start, end))

         
    possible_HAs = []
    for possibly_flipped_donor_seq, HA_start, HA_end in possible_HA_boundaries:
        donor_window = possibly_flipped_donor_seq[HA_start:HA_end]

        donor_prefix = donor_window[:required_match_length]

        if donor_prefix not in target_seq:
            raise ValueError

        donor_suffix = donor_window[-required_match_length:]

        if donor_suffix not in target_seq:
            raise ValueError

        target_HA_start = target_seq.index(donor_prefix)
        target_HA_end = target_seq.index(donor_suffix) + len(donor_suffix)

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
            'possibly_flipped_donor_seq': possibly_flipped_donor_seq,
            'donor_HA_start': HA_start,
            'donor_HA_end': HA_end,
            'target_HA_start': target_HA_start,
            'target_HA_end': target_HA_end,
            'lengths': lengths,
        }
        possible_HAs.append((min_mismatches, info))
        
    if not possible_HAs:
        results = {'failed': 'cannot locate homology arms'}
    else:
        results = min(possible_HAs)[1]

    return results

def build_target_info(base_dir, info, all_index_locations):
    ''' info should have keys:
            sgRNA_sequence
            amplicon_primers
        optional keys:
            donor_sequence
            nonhomologous_donor_sequence
            extra_sequences
    '''
    genome = info['genome']
    if info['genome'] not in all_index_locations:
        print(f'Error: can\'t locate indices for {genome}')
        sys.exit(0)
    else:
        index_locations = all_index_locations[genome]

    base_dir = Path(base_dir)

    name = info['name']

    donor_info = info.get('donor_sequence')
    if donor_info is None:
        donor_name = None
        donor_seq = None
    else:
        donor_name, donor_seq = donor_info
        if donor_name is None:
            donor_name = f'{name}_donor'

    if donor_seq is None:
        has_donor = False
    else:
        has_donor = True

    nh_donor_info = info.get('nonhomologous_donor_sequence')
    if nh_donor_info is None:
        nh_donor_name = None
        nh_donor_seq = None
    else:
        nh_donor_name, nh_donor_seq = nh_donor_info
        if nh_donor_name is None:
            nh_donor_name = f'{name}_NH_donor'

    if nh_donor_seq is None:
        has_nh_donor = False
    else:
        has_nh_donor = True

    target_dir = base_dir / 'targets' / name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    protospacer, *other_protospacers = info['sgRNA_sequence']
    amplicon_primers = info['amplicon_primers'][1].split(';')

    protospacer_dir = target_dir / 'protospacer_alignment'
    protospacer_dir.mkdir(exist_ok=True)
    fastq_fn = protospacer_dir / 'protospacer.fastq'
    STAR_prefix = protospacer_dir / 'protospacer_'
    bam_fn = protospacer_dir / 'protospacer.bam'

    STAR_index = index_locations['STAR']

    gb_fns = {
        'target': target_dir / f'{name}.gb',
        'donor': target_dir / f'{donor_name}.gb',
        'nh_donor': target_dir / f'{nh_donor_name}.gb',
    }

    # Make a fastq file with a single read containing the protospacer sequence.
    protospacer_name, protospacer_seq = protospacer
    
    with fastq_fn.open('w') as fh:
        quals = fastq.encode_sanger([40]*len(protospacer_seq))
        read = fastq.Read('protospacer', protospacer_seq, quals)
        fh.write(str(read))
        
    # Align the protospacer to the reference genome.
    mapping_tools.map_STAR(fastq_fn, STAR_index, STAR_prefix, mode='guide_alignment', bam_fn=bam_fn, sort=False)

    with pysam.AlignmentFile(bam_fn) as bam_fh:
        perfect_als = [al for al in bam_fh if not al.is_unmapped and sam.total_edit_distance(al) == 0]
    
    region_fetcher = genomes.build_region_fetcher(index_locations['fasta'])
    
    def evaluate_candidate(al):
        results = {
            'location': f'{al.reference_name} {al.reference_start:,} {sam.get_strand(al)}',
        }

        full_window_around = 5000

        full_around = region_fetcher(al.reference_name, al.reference_start - full_window_around, al.reference_end + full_window_around).upper()

        if sam.get_strand(al) == '+':
            ps_seq = protospacer_seq
            ps_strand = 1
        else:
            ps_seq = utilities.reverse_complement(protospacer_seq)
            ps_strand  = -1

        ps_start = full_around.index(ps_seq)

        protospacer_locations = [(ps_seq, ps_start, ps_strand)]

        for other_protospacer_name, other_protospacer_seq in other_protospacers:
            if other_protospacer_seq in full_around:
                ps_seq = other_protospacer_seq
                ps_strand = 1
            else:
                ps_seq =  utilities.reverse_complement(other_protospacer_seq)
                if ps_seq not in full_around:
                    results['failed'] = f'protospacer {other_protospacer_seq} not present near protospacer {protospacer_seq}'
                    return results
                ps_strand = -1

            ps_start = full_around.index(ps_seq)
            protospacer_locations.append((ps_seq, ps_start, ps_strand))

        for ps_seq, ps_start, ps_strand in protospacer_locations:
            PAM_pattern = 'NGG'

            if ps_strand == 1:
                PAM_offset = len(ps_seq)
                PAM_transform = utilities.identity
            else:
                PAM_offset = -len(PAM_pattern)
                PAM_transform = utilities.reverse_complement

            PAM_start = ps_start + PAM_offset
            PAM = PAM_transform(full_around[PAM_start:PAM_start + len(PAM_pattern)])
            pattern, *matches = Bio.SeqUtils.nt_search(PAM, PAM_pattern)

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
            'PCR_adapter_1': '#F8D3A9',
            'PCR_adapter_2': '#D59687',
        }
        
        target_features = [
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
        ]

        sgRNA_features = []
        for sgRNA_i, (ps_seq, ps_start, ps_strand) in enumerate(protospacer_locations):
            sgRNA_feature = SeqFeature(location=FeatureLocation(ps_start - offset, ps_start - offset + len(ps_seq), strand=ps_strand),
                                       id=f'sgRNA_{sgRNA_i}',
                                       type='sgRNA_SpCas9',
                                       qualifiers={'label': f'sgRNA_{sgRNA_i}',
                                                   'ApEinfo_fwdcolor': colors['sgRNA'],
                                                   },
                                       )
            target_features.append(sgRNA_feature)
            sgRNA_features.append(sgRNA_feature)

        results['gb_Records'] = {}

        if has_donor:
            # Identify the homology arms.

            if len(sgRNA_features) > 1:
                results['failed'] = 'multiple sgRNAs and a donor are not supported'
            else:
                sgRNA_feature = sgRNA_features[0]
                if sgRNA_feature.strand == 1:
                    # sgRNA_feature.end is the first nt of the PAM
                    cut_after = sgRNA_feature.location.end + target_info.effectors['SpCas9'].cut_after_offset
                else:
                    # sgRNA_feature.start - 1 is the first nt of the PAM
                    cut_after = sgRNA_feature.location.start - 1 - target_info.effectors['SpCas9'].cut_after_offset - 1

            HA_info = identify_homology_arms(donor_seq, target_seq, cut_after)
            if 'failed' in HA_info:
                results['failed'] = HA_info['failed']
                return results

            lengths = HA_info['lengths']

            starts = {
                'HA_1': HA_info['donor_HA_start'],
                'donor_specific': HA_info['donor_HA_start'] + lengths['HA_1'],
                'HA_2': HA_info['donor_HA_end'] - lengths['HA_2'],
            }
            ends = {
                'HA_1': starts['HA_1'] + lengths['HA_1'],
                'donor_specific': starts['HA_2'],
                'HA_2': starts['HA_2'] + lengths['HA_2'],
            }

            if info['donor_type'] == 'PCR':
                if starts['HA_1'] != 0:
                    starts['PCR_adapter_1'] = 0
                    ends['PCR_adapter_1'] = starts['HA_1']

                if ends['HA_2'] != len(donor_seq):
                    starts['PCR_adapter_2'] = ends['HA_2']
                    ends['PCR_adapter_2'] = len(donor_seq)
        
            donor_features = [
                SeqFeature(location=FeatureLocation(starts[key], ends[key], strand=1),
                           id=key,
                           type='misc_feature',
                           qualifiers={'label': key,
                                       'ApEinfo_fwdcolor': colors[key],
                                      },
                        )
                for key in starts
            ]

            donor_Seq = Seq(HA_info['possibly_flipped_donor_seq'], generic_dna)
            donor_Record = SeqRecord(donor_Seq, name=donor_name, features=donor_features)
            results['gb_Records']['donor'] = donor_Record

            target_features.extend([
                SeqFeature(location=FeatureLocation(HA_info['target_HA_start'], HA_info['target_HA_start'] + lengths['HA_1'], strand=1),
                        id='HA_1',
                        type='misc_feature',
                        qualifiers={'label': 'HA_1',
                                    'ApEinfo_fwdcolor': colors['HA_1'],
                                    },
                        ),
                SeqFeature(location=FeatureLocation(HA_info['target_HA_end'] - lengths['HA_2'], HA_info['target_HA_end'], strand=1),
                        id='HA_2',
                        type='misc_feature',
                        qualifiers={'label': 'HA_2',
                                    'ApEinfo_fwdcolor': colors['HA_2'],
                                    },
                        ),
            ])

        target_Seq = Seq(target_seq, generic_dna)
        target_Record = SeqRecord(target_Seq, name=name, features=target_features)
        results['gb_Records']['target'] = target_Record
        
        if has_nh_donor:
            nh_donor_Seq = Seq(nh_donor_seq, generic_dna)
            nh_donor_Record = SeqRecord(nh_donor_Seq, name=nh_donor_name)
            results['gb_Records']['nh_donor'] = nh_donor_Record

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

    truncated_name_i = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=BiopythonWarning)

        for which_seq, Record in best_candidate['gb_Records'].items(): 
            try:
                Bio.SeqIO.write(Record, gb_fns[which_seq], 'genbank')
            except ValueError:
                # locus line too long, can't write genbank file with BioPython
                old_name = Record.name

                truncated_name = f'{Record.name[:11]}_{truncated_name_i}'
                Record.name = truncated_name
                Bio.SeqIO.write(Record, gb_fns[which_seq], 'genbank')

                Record.name = old_name

                truncated_name_i += 1

    manifest_fn = target_dir / 'manifest.yaml'

    sources = [name]
    if has_donor:
        sources.append(donor_name)

    extra_Records = []
    if info.get('extra_sequences') is not None:
        for extra_seq_name, extra_seq in info['extra_sequences']:
            sources.append(extra_seq_name)

            extra_Records.append(SeqRecord(extra_seq, name=extra_seq_name))
        
    manifest = {
        'sources': sources,
        'target': name,
    }
    if has_donor:
        manifest['donor'] = donor_name
        manifest['donor_specific'] = 'donor_specific'

    if has_nh_donor:
        manifest['nonhomologous_donor'] = nh_donor_name

    manifest_fn.write_text(yaml.dump(manifest, default_flow_style=False))
        
    gb_records = list(best_candidate['gb_Records'].values()) + extra_Records
    ti = target_info.TargetInfo(base_dir, name, gb_records=gb_records)
    ti.make_references()
    ti.identify_degenerate_indels()

def build_target_infos_from_csv(base_dir):
    base_dir = Path(base_dir)
    csv_fn = base_dir / 'targets' / 'targets.csv'

    indices = target_info.locate_supplemental_indices(base_dir)

    targets_df = pd.read_csv(csv_fn, comment='#', index_col='name').replace({np.nan: None})

    # Fill in values for sequences that are specified by name.

    registry = {}

    sgRNA_fn = base_dir / 'targets' / 'sgRNAs.csv'

    if sgRNA_fn.exists():
        registry['sgRNA_sequence'] = pd.read_csv(sgRNA_fn, index_col='sgRNA_name', squeeze=True)
    else:
        registry['sgRNA_sequence'] = {}

    amplicon_primers_fn = base_dir / 'targets' / 'amplicon_primers.csv'

    if amplicon_primers_fn.exists():
        registry['amplicon_primers'] = pd.read_csv(amplicon_primers_fn, index_col='amplicon_primers_name', squeeze=True)
    else:
        registry['amplicon_primers'] = {}

    extra_sequences_fn = base_dir / 'targets' / 'extra_sequences.csv'
    if extra_sequences_fn.exists():
        registry['extra_sequence'] = pd.read_csv(extra_sequences_fn, index_col='name', squeeze=True)
    else:
        registry['extra_sequence'] = {}

    donors_fn = base_dir / 'targets' / 'donor_sequences.csv'

    if donors_fn.exists():
        donors = pd.read_csv(donors_fn, index_col='donor_name')
        registry['donor_sequence'] = donors['donor_sequence']
        registry['donor_type'] = donors['donor_type']
    else:
        registry['donor_sequence'] = {}
        registry['donor_type'] = {}

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
                possible_error_message = f'invalid char in {row.name}: {seq} \n Registered names: {registered_values}'

            if seq is not None and validate_sequence:
                seq = seq.upper()
                invalid_chars = set(seq) - valid_chars
                if invalid_chars:
                    print(possible_error_message)
                    print(invalid_chars)
                    sys.exit(0)

            looked_up.append((value_name, seq))

        if not multiple_lookups:
            looked_up = looked_up[0]
            
        return looked_up

    for target_name, row in targets_df.iterrows():
        info = {
            'name': target_name,
            'donor_sequence': lookup(row, 'donor_sequence', 'donor_sequence'),
            'sgRNA_sequence': lookup(row, 'sgRNA_sequence', 'sgRNA_sequence', multiple_lookups=True),
            'extra_sequences': lookup(row, 'extra_sequences', 'extra_sequence', multiple_lookups=True),
            'amplicon_primers': lookup(row, 'amplicon_primers', 'amplicon_primers'),
            'nonhomologous_donor_sequence': lookup(row, 'nonhomologous_donor_sequence', 'donor_sequence'),
            'donor_type': lookup(row, 'donor_sequence', 'donor_type', validate_sequence=False),
            'genome': row['genome'],
        }

        print(f'Building {target_name}...')
        build_target_info(base_dir, info, indices)

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

def download_genome_and_build_indices(base_dir, genome_name, num_threads=8):
    urls = {
        'hg38': 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz',
        'e_coli': 'ftp://ftp.ensemblgenomes.org/pub/bacteria/release-44/fasta/bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655/dna/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.chromosome.Chromosome.fa.gz',
    }

    if genome_name not in urls:
        print(f'No URL known for {genome_name}. Options are:')
        for gn in sorted(urls):
            print(f'\t- {gn}')
        sys.exit(0)

    base_dir = Path(base_dir)
    genome_dir = base_dir / 'indices' / genome_name
    fasta_dir = genome_dir / 'fasta'

    print(f'Downloading {genome_name}...')
    wget_command = [
        'wget', urls[genome_name],
        '-P', str(fasta_dir),
    ]
    subprocess.run(wget_command, check=True)

    print('Uncompressing...')
    file_name = Path(urlparse(urls[genome_name]).path).name

    gunzip_command = [
        'gunzip',  str(fasta_dir / file_name),
    ]
    subprocess.run(gunzip_command, check=True)

    build_indices(base_dir, genome_name, num_threads=num_threads)
