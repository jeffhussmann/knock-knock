#!/usr/bin/env python3.6

import argparse
import shutil
import subprocess
import pysam
import pandas as pd
import functools
import numpy as np
import yaml
import nbconvert
import nbformat.v4 as nbf
import Bio.SeqIO

import Sequencing.blast as blast
import Sequencing.fastq as fastq
import Sequencing.fasta as fasta
import Sequencing.utilities as utilities
import Sequencing.sam as sam
import Sequencing.interval as interval

import ribosomes.gff as gff

from pathlib import Path
from collections import Counter, defaultdict

import visualize

base_dir = Path('/home/jah/projects/manu')

def get_target_dir(target):
    target_dir = base_dir / target
    if not target_dir.is_dir():
        raise ValueError(target)
    return target_dir

def make_fns(target, dataset=None, outcome=None):
    fns = make_target_fns(target)

    if dataset is not None:
        dataset_fns = make_dataset_fns(target, dataset)
        fns.update(dataset_fns)

        if  outcome is not None:
            outcome_fns = make_outcome_fns(target, dataset, outcome)
            fns.update(outcome_fns)

    return fns

def make_target_fns(target):
    target_dir = get_target_dir(target)
    target_fns = {
        'target_dir': target_dir,
        'ref_gff': target_dir / 'refs.gff',
        'ref_fasta': target_dir / 'refs.fasta',
        'manifest': target_dir / 'manifest.yaml',
        'bams_dir': target_dir / 'bams',
        'fastqs_dir': target_dir / 'fastqs',
    }

    return target_fns

def make_dataset_fns(target, dataset):
    target_dir = get_target_dir(target)
    outcomes_dir = target_dir / 'outcomes' / dataset
    figures_dir = target_dir / 'figures' / dataset

    dataset_fns = {
        'full_fastq': target_dir / 'fastqs' / (dataset + '.fastq'),
        'full_bam': target_dir / 'bams' / (dataset + '.bam'),
        'full_bam_by_name': target_dir / 'bams' / (dataset + '.by_name.bam'),
        'all_lengths': figures_dir / 'all_lengths.png',
        'outcomes_dir': outcomes_dir,
        'counts': outcomes_dir / 'counts.csv',
        'figures_dir': figures_dir,
    }

    return dataset_fns

def make_outcome_fns(target, dataset, outcome): 
    target_dir = get_target_dir(target)
    outcomes_dir = target_dir / 'outcomes' / dataset
    figures_dir = target_dir / 'figures' / dataset

    outcome_string = '_'.join(outcome)
    outcome_base = outcomes_dir / outcome_string

    outcome_fns = {
        'qnames': outcome_base.with_suffix('.txt'),
        'bam': outcome_base.with_suffix('.bam'),
        'bam_by_name': outcome_base.with_suffix('.by_name.bam'),
        'lengths': figures_dir / (outcome_string + '_lengths.png'),
        'first': figures_dir / (outcome_string + '_first.png'),
        'figure': figures_dir / (outcome_string + '.png'),
        'text': figures_dir / (outcome_string + '_alignments.txt'),
    }

    return outcome_fns

def get_datasets(target):
    fastqs_dir = make_fns(target)['fastqs_dir']
    fastq_fns = sorted(fastqs_dir.glob('*.fastq'))
    datasets = [fn.stem for fn in fastq_fns]
    return datasets

def get_outcomes(target, dataset):
    fns = make_fns(target, dataset)
    outcome_fns = sorted(fns['outcomes_dir'].glob('*.txt'))
    outcomes = [fn.stem.split('_') for fn in outcome_fns]
    return outcomes

def get_outcome_qnames(target, dataset, outcome):
    fns = make_fns(target, dataset, outcome)
    if outcome is None:
        qnames = []
    else:
        qnames = [l.strip() for l in open(fns['qnames'])]
    return qnames

def generate_processors(targets):
    for target in targets:
        make_references(target)
        datasets = get_datasets(target)
        for dataset in datasets[:1]:
            print(f'{target} {dataset}')

def generate_bams(target, dataset):
    fns = make_fns(target, dataset)
    fns['bams_dir'].mkdir(parents=True, exist_ok=True)
    blast.blast(fns['ref_fasta'], fns['full_fastq'], output=fns['full_bam'])
    sam.to_sorted_bam(fns['full_bam'], fns['full_bam_by_name'], by_name=True)

def overlaps_feature(alignment, feature):
    same_reference = alignment.reference_name == feature.seqname
    num_overlapping_bases = alignment.get_overlap(feature.start, feature.end)
    return same_reference and (num_overlapping_bases > 0)

def determine_layout(als, features, target, query_length):
    primer = {
        5: features[target, 'forward primer'],
        3: features[target, 'reverse primer'],
    }

    als_containing_primer = {
        side: [al for al in als if overlaps_feature(al, primer[side])]
        for side in [5, 3]
    }

    if len(als_containing_primer[5]) > 1 or len(als_containing_primer[3]) > 1:
        outcome = ('malformed layout', 'extra copies of primer')

    elif len(als_containing_primer[3]) == 0 or len(als_containing_primer[3]) == 0:
        outcome = ('malformed layout', 'missing a primer')

    else:
        has_primer = {side: als_containing_primer[side][0] for side in [5, 3]}

        if sam.get_strand(has_primer[5]) != sam.get_strand(has_primer[3]):
            outcome = ('malformed layout', 'primers not in same orientation')

        else:
            # How much of the read is covered by alignments containing the primers?
            covered = interval.get_disjoint_covered([has_primer[5], has_primer[3]])

            if covered.start > 10 or query_length - covered.end > 10:
                outcome = ('malformed layout', 'primer far from read edge')

            else:
                if len(covered) == 1:
                    # The number of disjoint intervals is 1 - i.e. it is a
                    # single connected interval.

                    # TODO: characterize scar status.

                    outcome = ['no integration', '']

                else:
                    outcome = 'integration'

    return outcome

def count_outcomes(target, dataset):
    fns = make_fns(target, dataset)
    
    manifest = yaml.load(fns['manifest'].open())
    donor = manifest['donor']
    target = manifest['target']
    
    total_reads = sum(1 for _ in fastq.reads(fns['full_fastq']))
   
    features = {
        (f.seqname, f.attribute['ID']): f
        for f in gff.get_all_features(fns['ref_gff'])
        if 'ID' in f.attribute
    }
    
    bam_fh = pysam.AlignmentFile(fns['full_bam_by_name'])
    alignment_groups = utilities.group_by(bam_fh, lambda r: r.query_name)

    outcomes = defaultdict(list)

    HAs = {}
    for source in [donor, target]:
        for side in [5, 3]:
            HAs[source, side] = features[source, "{0}' HA".format(side)]

    target_cut = {
        5: features[target, "5' cut"],
        3: features[target, "3' cut"],
    }
        
    for name, als in alignment_groups:
        query_length = als[0].query_length
        quals = als[0].query_qualities
        strand = sam.get_strand(has_primer[5])

        outcome = determine_layout(als, feature, target, query_length)

        if outcome != 'integration':
            outcomes[outcome].append(name)
            continue
        
        target_relative_to_cut = {
            5: has_primer[5].reference_end - 1 - target_cut[5].end,
            3: has_primer[3].reference_start - target_cut[3].start,
        }
        
        target_q = {
            5: sam.true_query_position(has_primer[5].query_alignment_end - 1, has_primer[5]),
            3: sam.true_query_position(has_primer[3].query_alignment_start, has_primer[3]),
        }
        
        target_blunt = {}
        for side in [5, 3]:
            blunt = False
            offset = target_relative_to_cut[side]
            if offset == 0:
                blunt = True
            elif abs(offset) <= 3:
                q = target_q[side]
                if min(quals[q - 3: q + 4]) <= 30:
                    blunt = True
                    
            target_blunt[side] = blunt
        
        target_relative_to_arm = {
            5: has_primer[5].reference_end - 1 - HAs[target, 5].end,
            3: HAs[target, 3].start - has_primer[3].reference_start, 
        }

        # Mask off the parts of the read explained by the primer-containing alignments.
        # If these alignments go to exactly the cut site, use the cut as the boundary.
        # Otherwise, use the homology arm edge.
        
        if target_blunt[5]:
            mask_end = target_cut[5].end
        else:
            mask_end = HAs[target, 5].end
            
        has_primer[5] = sam.restrict_alignment_to_reference_interval(has_primer[5], 0, mask_end)
        
        if target_blunt[3]:
            mask_start = target_cut[3].start
        else:
            mask_start = HAs[target, 3].start
            
        has_primer[3] = sam.restrict_alignment_to_reference_interval(has_primer[3], mask_start, np.inf)
        
        covered_from_primers = interval.get_disjoint_covered([has_primer[5], has_primer[3]])
        integration_interval = interval.Interval(covered_from_primers[0].end + 1, covered_from_primers[-1].start - 1)

        parsimonious = interval.make_parsimoninous(als)
        donor_als = [al for al in parsimonious if al.reference_name == manifest['donor']]

        if len(donor_als) == 0:
            e_coli_als = [al for al in parsimonious if al.reference_name == 'e_coli_K12']
            if len(e_coli_als) == 1:
                e_coli_al, = e_coli_als
                e_coli_covered = interval.get_covered(e_coli_al)
                if e_coli_covered.start - integration_interval.start <= 10 and integration_interval.end - e_coli_covered.end <= 10:
                    outcomes['non-GFP integration', 'e coli'].append(name)
                    continue
                else:
                    outcomes['non-GFP integration', 'e coli'].append(name)
                    continue
            else:
                outcomes['non-GFP integration', 'uncategorized'].append(name)
                continue

        left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
        right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)
        if strand == '+':
            five_most, three_most = left_most, right_most
        else:
            five_most, three_most = right_most, left_most
            
        target_contains_full_arm = {
            5: HAs[target, 5].end - has_primer[5].reference_end <= 10,
            3: has_primer[3].reference_start - HAs[target, 3].start <= 10,
        }
        
        target_external_edge = {
            5: sam.closest_query_position(HAs[target, 5].start, has_primer[5]),
            3: sam.closest_query_position(HAs[target, 3].end, has_primer[3]),
        }
        
        donor_contains_full_arm = {
            5: (five_most.reference_start - HAs[donor, 5].start <= 10) and (five_most.reference_end - 1 - HAs[donor, 5].end >= 20),
            3: (HAs[donor, 3].end - (three_most.reference_end - 1) <= 10) and (HAs[donor, 3].start - three_most.reference_start >= 20),
        }
        
        donor_external_edge = {
            5: sam.closest_query_position(HAs[donor, 5].start, five_most),
            3: sam.closest_query_position(HAs[donor, 3].end, three_most),
        }
        
        junction = {
            5: sam.restrict_alignment_to_reference_interval(five_most, HAs[donor, 5].end - 10, HAs[donor, 5].end + 10),
            3: sam.restrict_alignment_to_reference_interval(three_most, HAs[donor, 3].start - 10, HAs[donor, 3].start + 10),
        }
        max_indel_nearby = {side: sam.max_indel_length(junction[side]) for side in [5, 3]}
        
        clean_handoff = {
            side: target_contains_full_arm[side] and \
                  donor_contains_full_arm[side] and \
                  abs(target_external_edge[side] - donor_external_edge[side]) <= 10 and \
                  max_indel_nearby[side] <= 2
            for side in [5, 3]
        }
        
        five_most = sam.restrict_alignment_to_query_interval(five_most, integration_interval.start, integration_interval.end)
        three_most = sam.restrict_alignment_to_query_interval(three_most, integration_interval.start, integration_interval.end)

        # 5 and 3 are both positive if there is extra in the insert and negative if truncated
        donor_relative_to_arm_internal = {
            5: (HAs[donor, 5].end + 1) - five_most.reference_start,
            3: ((three_most.reference_end - 1) + 1) - HAs[donor, 3].start,
        }
        
        donor_relative_to_arm_external = {
            5: five_most.reference_start - HAs[donor, 5].start,
            3: three_most.reference_end - 1 - HAs[donor, 3].end,
        }
        
        donor_q = {
            5: sam.true_query_position(five_most.query_alignment_start, five_most),
            3: sam.true_query_position(three_most.query_alignment_end - 1, three_most),
        }
        
        donor_blunt = {}
        for side in [5, 3]:
            blunt = False
            offset = donor_relative_to_arm_external[side]
            
            if offset == 0:
                blunt = True
            elif abs(offset) <= 3:
                q = donor_q[side]
                if min(quals[q - 3:q + 4]) <= 30:
                    blunt = True
                    
            donor_blunt[side] = blunt
            
        junction_status = []
        
        for side in [5, 3]:
            if target_blunt[side] and donor_blunt[side]:
                junction_status.append('{0}\' blunt'.format(side))
            elif clean_handoff[side]:
                pass
            else:
                junction_status.append('{0}\' uncategorized'.format(side))
        
        if len(junction_status) == 0:
            junction_description = ' '
        else:
            junction_description = ', '.join(junction_status)
        
        if len(donor_als) == 1:
            if clean_handoff[5] and clean_handoff[3]:
                insert_description = 'expected'
            elif donor_relative_to_arm_internal[5] < 0 or donor_relative_to_arm_internal[3] < 0:
                insert_description = 'truncated'
            elif donor_relative_to_arm_internal[5] > 0 or donor_relative_to_arm_internal[3] > 0:
                insert_description = 'extended'
            else:
                insert_description = 'uncategorized insertion'
            
            if sam.get_strand(donor_als[0]) != strand:
                insert_description = 'flipped ' + insert_description
                
        else:
            # Concatamer?
            if strand == '+':
                key = lambda al: interval.get_covered(al).start
                reverse = False
            else:
                key = lambda al: interval.get_covered(al).end
                reverse = True

            five_to_three = sorted(donor_als, key=key, reverse=reverse)
            concatamer_junctions = []
            for before_junction, after_junction in zip(five_to_three[:-1], five_to_three[1:]):
                adjacent = interval.are_adjacent(interval.get_covered(before_junction), interval.get_covered(after_junction))
                missing_before = before_junction.reference_end - 1 - HAs[donor, 3].end
                missing_after = after_junction.reference_start - HAs[donor, 5].start
                clean = adjacent and (missing_before == 0) and (missing_after == 0)
                concatamer_junctions.append(clean)

            if all(concatamer_junctions):
                #insert_description += ', {0}-mer GFP'.format(len(concatamer_junctions) + 1)
                insert_description = 'concatamer'.format(len(concatamer_junctions) + 1)
            else:
                insert_description = 'uncategorized insertion'
        
            if any(sam.get_strand(donor_al) != strand for donor_al in donor_als):
                insert_description = 'flipped ' + insert_description

        outcomes[insert_description, junction_description].append(name)
        
    bam_fh.close()
        
    counts = {description: len(names) for description, names in outcomes.items()}
    accounted_for = sum(counts.values())
    counts['malformed layout', 'no alignments detected'] = total_reads - accounted_for
    
    return outcomes, counts

def parse_benchling_genbank(genbank_fn):
    convert_strand = {
        -1: '-',
        1: '+',
    }

    gb_record = Bio.SeqIO.read(str(genbank_fn), 'genbank')

    fasta_record = fasta.Read(gb_record.name, str(gb_record.seq))

    gff_features = []

    for gb_feature in gb_record.features:
        attribute = {k: v[0] for k, v in gb_feature.qualifiers.items()}

        if 'label' in attribute:
            attribute['ID'] = attribute.pop('label')

        if 'ApEinfo_fwdcolor' in attribute:
            attribute['color'] = attribute.pop('ApEinfo_fwdcolor')

        if 'ApEinfo_revcolor' in attribute:
            attribute.pop('ApEinfo_revcolor')


        feature = gff.Feature.from_fields(
            gb_record.id,
            '.',
            gb_feature.type,
            gb_feature.location.start,
            gb_feature.location.end - 1,
            '.',
            convert_strand[gb_feature.location.strand],
            '.',
            '.',
        )

        feature.attribute = attribute

        gff_features.append(feature)

    return fasta_record, gff_features

def make_references(target):
    fns = make_fns(target)
    manifest = yaml.load(fns['manifest'].open())
    gbs = [fns['target_dir'] / (source + '.gb') for source in manifest['sources']]
    
    fasta_records = []
    all_gff_features = []
    
    for gb in gbs:
        fasta_record, gff_features = parse_benchling_genbank(gb)
        fasta_records.append(fasta_record)
        all_gff_features.extend(gff_features)
        
    with fns['ref_fasta'].open('w') as fasta_fh:
        for record in fasta_records:
            fasta_fh.write(str(record))
            
    pysam.faidx(str(fns['ref_fasta']))
            
    with fns['ref_gff'].open('w') as gff_fh:
        gff_fh.write('##gff-version 3')
        for feature in sorted(all_gff_features):
            gff_fh.write(str(feature) + '\n')

def load_counts(target):
    counts = {}
    for dataset in get_datasets(target):
        fns = make_fns(target, dataset)
        counts[dataset] = pd.read_csv(fns['counts'], index_col=(0, 1), header=None, squeeze=True)

    df = pd.DataFrame(counts).fillna(0)
    totals = df.sum(axis=0)
    df.loc[(' ', 'Total reads'), :] = totals
    df = df.sort_index().astype(int)
    df.index.names = (None, None)

    return df

def generate_html(target, must_contain=None):
    if must_contain is None:
        must_contain = []
    
    if not isinstance(must_contain, (list, tuple)):
        must_contain = [must_contain]

    nb = nbf.new_notebook()

    cell_contents = '''\
import table

table.make_table('{0}', must_contain={1})
'''.format(target, must_contain)

    nb['cells'] = [nbf.new_code_cell(cell_contents)]

    title = target
    if len(must_contain) > 0:
        title += '_' + '_'.join(must_contain)

    nb['metadata'] = {'title': title}

    exporter = nbconvert.HTMLExporter()
    exporter.template_file = 'modal_template.tpl'

    ep = nbconvert.preprocessors.ExecutePreprocessor(kernel_name='python3.6')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open('table_{0}.html'.format(title), 'w') as fh:
        fh.write(body)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--process', nargs=2)
    group.add_argument('--generate_processors', nargs='*')

    args = parser.parse_args()

    if args.generate_processors is not None:
        generate_processors(args.generate_processors)

    elif args.process is not None:
        target, dataset = args.process
        #fns = make_fns(target, dataset)
#
        #generate_bams(target, dataset)
#
        #if fns['outcomes_dir'].is_dir():
        #    shutil.rmtree(fns['outcomes_dir'])
#
        #outcomes, counts = count_outcomes(target, dataset)
        #
        #fns['outcomes_dir'].mkdir()
#
        #qname_to_outcome = {}
#
        #bam_fns = {
        #    'by_name': {},
        #    'by_pos': {},
        #}
        #
        #for outcome, qnames in outcomes.items():
        #    outcome_fns = make_fns(target, dataset, outcome)
        #    bam_fns['by_name'][outcome] = outcome_fns['bam_by_name']
        #    bam_fns['by_pos'][outcome] = outcome_fns['bam']
        #    
        #    with outcome_fns['qnames'].open('w') as fh:
        #        for qname in qnames:
        #            qname_to_outcome[qname] = outcome
        #            fh.write(qname + '\n')
        #        
        #full_bam_fh = pysam.AlignmentFile(fns['full_bam'])
        #
        #def make_sorter(outcome, order):
        #    sorter = sam.AlignmentSorter(full_bam_fh.references,
        #                                 full_bam_fh.lengths,
        #                                 bam_fns[order][outcome],
        #                                 by_name=(order == 'by_name'),
        #                                )
        #    return sorter
        #
        #flat = []
        #bam_fhs = {}
        #for order in bam_fns:
        #    bam_fhs[order] = {}
        #    for outcome in bam_fns[order]:
        #        sorter = make_sorter(outcome, order)
        #        flat.append(sorter)
        #        bam_fhs[order][outcome] = sorter
        #        
        #with sam.multiple_AlignmentSorters(flat):
        #    for al in full_bam_fh:
        #        outcome = qname_to_outcome[al.query_name]
        #        for order in bam_fhs:
        #            bam_fhs[order][outcome].write(al)
#
        #series = pd.Series(counts)
        #series.to_csv(fns['counts'])
#
        #visualize.make_outcome_plots(target, dataset)
        visualize.make_outcome_text_alignments(target, dataset)
