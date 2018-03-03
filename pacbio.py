import shutil
import subprocess
import pysam
import pandas as pd
import functools
import numpy as np
import nbconvert
import nbformat.v4 as nbf

import Sequencing.fastq as fastq
import Sequencing.fasta as fasta
import Sequencing.utilities as utilities
import Sequencing.sam as sam
import Sequencing.interval as interval

from pathlib import Path
from collections import Counter, defaultdict

import visualize

base_dir = Path('/home/jah/projects/manu')

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
#
        #visualize.make_outcome_plots(target, dataset)
        #visualize.make_outcome_text_alignments(target, dataset)
