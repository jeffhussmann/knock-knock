import shutil
from collections import Counter, defaultdict
from pathlib import Path

import scipy.sparse
import pandas as pd
import numpy as np
import yaml
import ipywidgets

from sequencing import utilities, sam, fastq, mapping_tools
from knockin import experiment, target_info, collapse, coherence, pooled_layout, visualize

memoized_property = utilities.memoized_property

class SingleGuideExperiment(experiment.Experiment):
    def __init__(self, *args, **kwargs):
        super(SingleGuideExperiment, self).__init__(*args, **kwargs)
        
        self.fns.update({
            'R2': self.data_dir / 'by_guide' / '{}_R2.fastq.gz'.format(self.name),
            'collapsed_R2': self.dir / 'collapsed_R2.fastq',
            'guide_mismatch_rates': self.dir / 'guide_mismatch_rates.txt',

            'supplemental_STAR_prefix': lambda name: self.dir / '{}_alignments_STAR.'.format(name),
            'supplemental_bam': lambda name: self.dir / '{}_alignments.bam'.format(name),
            'supplemental_bam_by_name': lambda name: self.dir / '{}_alignments.by_name.bam'.format(name),

            'combined_bam': self.dir / 'combined.bam',
            'combined_bam_by_name': self.dir / 'combined.by_name.bam',

            'collapsed_UMI_outcomes': self.dir / 'collapsed_UMI_outcomes.txt',
            'cell_outcomes': self.dir / 'cell_outcomes.txt',
            'filtered_cell_outcomes': self.dir / 'filtered_cell_outcomes.txt',
        })
        
        self.layout_module = pooled_layout
        self.max_insertion_length = 4
        self.max_qual = 41

        self.supplemental_indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'bosTau7': '/nvme/indices/bosTau7',
        }

        self.min_reads_per_cluster = 2

        self.pool = PooledScreen(self.base_dir, self.group)

    @property
    def reads(self):
        reads = fastq.reads(self.fns['R2'], up_to_space=True)
        reads = self.progress(reads)

        return reads
    
    def collapse_UMI_reads(self):
        ''' Takes R2_fn sorted by UMI and collapses reads with the same UMI and
        sufficiently similar sequence.
        '''

        def UMI_key(read):
            return collapse.Annotations['UMI_protospacer'].from_identifier(read.name)['UMI']

        def num_reads_key(read):
            return collapse.Annotations['collapsed_UMI'].from_identifier(read.name)['num_reads']

        R1_read_length = 45

        mismatch_counts = np.zeros(R1_read_length)
        total = 0

        expected_seq = self.pool.guides_df.loc[self.name, 'full_seq'][:R1_read_length]

        with self.fns['collapsed_R2'].open('w') as collapsed_fh:
            groups = utilities.group_by(self.reads, UMI_key)
            for UMI, UMI_group in groups:
                clusters = collapse.form_clusters(UMI_group, max_read_length=None, max_hq_mismatches=0)
                clusters = sorted(clusters, key=num_reads_key, reverse=True)

                for i, cluster in enumerate(clusters):
                    annotation = collapse.Annotations['collapsed_UMI'].from_identifier(cluster.name)
                    annotation['UMI'] = UMI
                    annotation['cluster_id'] = i

                    if annotation['num_reads'] >= self.min_reads_per_cluster:
                        total += 1
                        guide = annotation['guide']
                        if guide == expected_seq:
                            mismatch = -1
                        else:
                            qs = fastq.decode_sanger(annotation['guide_qual'])
                            mismatches = []
                            for i, (seen, expected, q) in enumerate(zip(guide, expected_seq, qs)):
                                if seen != expected and q >= 30:
                                    mismatches.append(i)

                            if len(mismatches) == 0:
                                mismatch = -1
                            elif len(mismatches) == 1:
                                mismatch = mismatches[0]
                            elif len(mismatches) > 1:
                                continue

                            mismatch_counts[mismatch] += 1

                        mismatch_annotation = collapse.Annotations['collapsed_UMI_mismatch'](annotation)
                        mismatch_annotation['mismatch'] = mismatch

                        cluster.name = str(mismatch_annotation)

                        collapsed_fh.write(str(cluster))

        mismatch_rates = mismatch_counts / total
        np.savetxt(self.fns['guide_mismatch_rates'], mismatch_rates)
    
    @property
    def collapsed_reads(self):
        reads = fastq.reads(self.fns['collapsed_R2'])
        reads = self.progress(reads)

        return reads

    def generate_supplemental_alignments(self):
        ''' Use STAR to produce local alignments, post-filtering spurious alignmnents.
        '''

        for index_name, index in self.supplemental_indices.items():
            bam_fn = mapping_tools.map_STAR(self.fns['collapsed_R2'],
                                            index,
                                            self.fns['supplemental_STAR_prefix'](index_name),
                                            sort=False,
                                            mode='permissive',
                                           )

            all_mappings = pysam.AlignmentFile(bam_fn)
            header = all_mappings.header
            new_references = ['{}_{}'.format(index_name, ref) for ref in header.references]
            new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)
            filtered_fn = str(self.fns['supplemental_bam_by_name'](index_name))

            with pysam.AlignmentFile(filtered_fn, 'wb', header=new_header) as fh:
                for al in all_mappings:
                    if al.query_alignment_length <= 20:
                        continue

                    if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        continue

                    fh.write(al)

            sam.sort_bam(self.fns['supplemental_bam_by_name'](index_name),
                         self.fns['supplemental_bam'](index_name),
                        )

    def combine_alignments(self):
        supplemental_fns = [self.fns['supplemental_bam'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam']] + supplemental_fns,
                                   self.fns['combined_bam'],
                                  )

        supplemental_fns = [self.fns['supplemental_bam_by_name'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam_by_name']] + supplemental_fns,
                                   self.fns['combined_bam_by_name'],
                                   by_name=True,
                                  )
        
    def categorize_outcomes(self):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        total = 0
        required_sw = 0

        with self.fns['outcome_list'].open('w') as fh:
            for name, als in self.alignment_groups('combined_bam_by_name'):
                layout = self.layout_module.Layout(als, self.target_info)
                total += 1
                try:
                    category, subcategory, details = layout.categorize()
                except:
                    print()
                    print(self.name, name)
                    raise
                
                if layout.required_sw:
                    required_sw += 1
                
                outcomes[category, subcategory].append(name)

                annotation = collapse.Annotations['collapsed_UMI_mismatch'].from_identifier(name)
                UMI_outcome = coherence.Pooled_UMI_Outcome(annotation['UMI'],
                                                           annotation['guide_mismatch'],
                                                           annotation['cluster_id'],
                                                           annotation['num_reads'],
                                                           category,
                                                           subcategory,
                                                           details,
                                                           name,
                                                          )
                fh.write(str(UMI_outcome) + '\n')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        full_bam_fh = pysam.AlignmentFile(str(self.fns['combined_bam_by_name']))
        
        for outcome, qnames in outcomes.items():
            outcome_fns = self.outcome_fns(outcome)
            outcome_fns['dir'].mkdir()
            bam_fhs[outcome] = pysam.AlignmentFile(str(outcome_fns['bam_by_name']), 'w', template=full_bam_fh)
            
            with outcome_fns['query_names'].open('w') as fh:
                for qname in qnames:
                    qname_to_outcome[qname] = outcome
                    fh.write(qname + '\n')
        
        for al in full_bam_fh:
            outcome = qname_to_outcome[al.query_name]
            bam_fhs[outcome].write(al)

        full_bam_fh.close()
        for outcome, fh in bam_fhs.items():
            fh.close()

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_table(self.fns['outcome_counts'],
                               header=None,
                               index_col=[0, 1, 2, 3],
                               squeeze=True,
                               na_filter=False,
                              )
        counts.index.names = ['perfect_guide', 'category', 'subcategory', 'details']
        return counts
    
    @memoized_property
    def perfect_guide_outcome_counts(self):
        return self.outcome_counts.xs(True)
    
    def collapse_UMI_outcomes(self):
        all_collapsed_outcomes, most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in all_collapsed_outcomes:
                fh.write(str(outcome) + '\n')
        
        with self.fns['cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')
        
        counts = Counter()
        with self.fns['filtered_cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                if outcome.num_reads >= 10:
                    fh.write(str(outcome) + '\n')
                    perfect = (outcome.guide_mismatch == -1)
                    counts[perfect, outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts).sort_values(ascending=False)
        counts.to_csv(self.fns['outcome_counts'], sep='\t')

    @memoized_property
    def filtered_cell_outcomes(self):
        df = pd.read_table(self.fns['filtered_cell_outcomes'], header=None, names=coherence.Pooled_UMI_Outcome.columns)
        return df

    def process(self):
        try:
            #self.collapse_UMI_reads()
            #self.generate_alignments()
            #self.generate_supplemental_alignments()
            #self.combine_alignments()
            #self.categorize_outcomes()
            self.collapse_UMI_outcomes()
            #self.make_outcome_plots(num_examples=3)
        except:
            print(self.name)
            raise
    
def collapse_categories(df):
    supplemental_names = {s for c, s, v in df.index.values if c == 'genomic insertion'}
    supplemental_counts = {name: df.xs(('genomic insertion', name)).sum() for name in supplemental_names}
        
    donor_insertions = df.xs('donor insertion').sum()

    df = df.drop('donor insertion', level=0)
    df = df.drop('genomic insertion', level=0)

    df.loc['donor insertion', 'donor insertion', 'n/a'] = donor_insertions
    for name in supplemental_names:
        df.loc['genomic insertion', name, 'n/a'] = supplemental_counts[name]

    return df
    
class PooledScreen(object):
    def __init__(self, base_dir, group, progress=None):
        self.base_dir = Path(base_dir)
        self.group = group

        if progress is None:
            progress = utilities.identity

        self.progress = progress

        sample_sheet_fn = self.base_dir / 'data' / group / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        self.target_name = sample_sheet['target_info_prefix']
        self.target_info = target_info.TargetInfo(self.base_dir, self.target_name)

        self.fns = {
            'guides': self.base_dir / 'guides' / 'guides.txt',
            'outcome_counts': self.base_dir / 'results' / group / 'outcome_counts.npz',
            'total_outcome_counts': self.base_dir / 'results' / group / 'total_outcome_counts.txt',
            'mismatch_outcome_counts': self.base_dir / 'results' / group / 'mismatch_outcome_counts.npz',
            'mismatch_total_outcome_counts': self.base_dir / 'results' / group / 'mismatch_total_outcome_counts.txt',
            'quantiles': self.base_dir / 'results' / group / 'quantiles.hdf5',
        }

    @memoized_property
    def guides_df(self):
        guides_df = pd.read_table(self.base_dir / 'guides' / 'guides.txt', index_col='short_name')
        return guides_df

    @memoized_property
    def guides(self):
        guides = self.guides_df.index.values
        return guides

    @memoized_property
    def non_targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' in g]

    @memoized_property
    def targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' not in g]

    @memoized_property
    def genes(self):
        return sorted(set(self.guides_df['gene']))

    def gene_guides(self, gene):
        return self.guides_df.query('gene == @gene').index

    def guide_to_gene(self, guide):
        return self.guides_df.loc[guide]['gene']

    def make_outcome_counts(self):
        all_counts = {}
        guides = self.guides

        for guide in self.progress(guides):
            exp = SingleGuideExperiment(self.base_dir, self.group, guide)
            if len(exp.outcome_counts) > 0:
                all_counts[guide] = exp.outcome_counts

        all_outcomes = set()

        for guide in all_counts:
            all_outcomes.update(all_counts[guide].index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(guides)), dtype=int)

        for g, guide in enumerate(self.progress(guides)):
            for outcome, count in all_counts[guide].items():
                o = outcome_to_index[outcome]
                counts[o, g] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.todense(),
                            columns=guides,
                            index=pd.MultiIndex.from_tuples(outcome_order),
                            )

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'])

    @memoized_property
    def total_outcome_counts(self):
        return pd.read_csv(self.fns['total_outcome_counts'], header=None, index_col=[0, 1, 2, 3], na_filter=False)
    
    @memoized_property
    def mismatch_total_outcome_counts(self):
        return pd.read_csv(self.fns['mismatch_total_outcome_counts'], header=None, index_col=[0, 1, 2], na_filter=False)

    @memoized_property
    def outcome_counts_df(self):
        guides = self.guides
        sparse_counts = scipy.sparse.load_npz(self.fns['outcome_counts'])
        df = pd.DataFrame(sparse_counts.todense(),
                          index=self.total_outcome_counts.index,
                          columns=guides,
                         )
        df.index.names = ('perfect_guide', 'category', 'subcategory', 'details')

        return df

    @memoized_property
    def perfect_guide_outcome_counts(self):
        perfect_guide = self.outcome_counts_df.xs(True)
        collapsed = collapse_categories(perfect_guide)
        return collapsed

    @memoized_property
    def mismatch_guide_outcome_counts(self):
        mismatch_guide = self.outcome_counts_df.xs(False)
        collapsed = collapse_categories(mismatch_guide)
        return collapsed

    @memoized_property
    def all_outcome_counts(self):
        return self.outcome_counts_df.sum(level=[1, 2, 3])
    
    @memoized_property
    def perfect_guide_UMI_counts(self):
        return self.perfect_guide_outcome_counts.sum()
    
    @memoized_property
    def mismatch_guide_UMI_counts(self):
        return self.mismatch_guide_outcome_counts.sum()
    
    @memoized_property
    def all_UMI_counts(self):
        return self.all_outcome_counts.sum()
    
    @memoized_property
    def non_targeting_outcomes(self):
        guide_outcomes = {}
        for nt_guide in self.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, nt_guide)
            fn = exp.fns['filtered_cell_outcomes']

            outcomes = [coherence.Pooled_UMI_Outcome.from_line(line) for line in fn.open()]

            for outcome in outcomes:
                if outcome.category == 'genomic insertion':
                    outcome.details = 'n/a'
                
                if outcome.category == 'donor insertion':
                    outcome.details = 'n/a'

            guide_outcomes[nt_guide] = outcomes

        return guide_outcomes

    @memoized_property
    def all_non_targeting_counts(self):
        return self.outcome_counts[self.non_targeting_guides].sum(axis=1).sort_values(ascending=False)
    
    @memoized_property
    def all_non_targeting_fractions(self):
        return self.all_non_targeting_counts / self.all_non_targeting_counts.sum()

    @memoized_property
    def most_frequent_outcomes(self):
        return self.all_non_targeting_counts.index.values[:200]

    @memoized_property
    def common_non_targeting_fractions(self):
        counts = self.common_counts[self.non_targeting_guides].sum(axis=1)
        return counts / counts.sum()
    
    @memoized_property
    def common_counts(self):
        frequent_counts = self.outcome_counts.loc[self.most_frequent_outcomes] 
        leftover = self.UMI_counts - frequent_counts.sum()
        leftover_row = pd.DataFrame.from_dict({('uncommon', 'uncommon', 'n/a'): leftover}, orient='index')
        everything = pd.concat([frequent_counts, leftover_row])
        return everything

    @memoized_property
    def common_fractions(self):
        return self.common_counts / self.UMI_counts

    @memoized_property
    def fold_changes(self):
        return self.common_fractions.div(self.common_non_targeting_fractions, axis=0)

    @memoized_property
    def log2_fold_changes(self):
        fc = self.fold_changes
        smallest_nonzero = fc[fc > 0].min().min()
        floored = np.maximum(fc, smallest_nonzero)
        return np.log2(floored)

    def rational_outcome_order(self):
        def get_deletion_info(details):
            _, starts, length = pooled_layout.string_to_indels(details)[0]
            return {'num_MH_nts': len(starts) - 1,
                    'start': min(starts),
                    'length': length,
                }

        def has_MH(details):
            info = get_deletion_info(details)
            return info['num_MH_nts'] >= 2 and info['length'] > 1

        conditions = {
            'insertions': lambda c, sc, d: sc == 'insertion',
            'no_MH_deletions': lambda c, sc, d: sc == 'deletion' and not has_MH(d),
            'MH_deletions': lambda c, sc, d: sc == 'deletion' and has_MH(d),
            'donor': lambda c, sc, d: sc == 'donor' or sc == 'other',
            'wt': lambda c, sc, d: sc == 'wild type',
            'uncat': lambda c, sc, d: c == 'uncategorized',
            'genomic': lambda c, sc, d: c == 'genomic insertion',
            'donor insertion': lambda c, sc, d: c == 'donor insertion',
            'uncommon': [('uncommon', 'uncommon', 'n/a')],
        }

        group_order = [
            'uncat',
            'genomic',
            'donor insertion',
            'wt',
            'donor',
            'insertions',
            'no_MH_deletions',
            'MH_deletions',
            'uncommon',
        ]

        donor_order = [
                ('no indel', 'donor', 'ACGAGTTT'),
                ('no indel', 'other', '___AGTTT'),
                ('no indel', 'other', '____GTTT'),
                ('no indel', 'other', '___AGTT_'),
                ('no indel', 'other', '____GTT_'),
                ('no indel', 'other', '____GT__'),
                ('no indel', 'other', '____G___'),
                ('no indel', 'other', 'ACGAGTT_'),
                ('no indel', 'other', 'ACGAG___'),
                ('no indel', 'other', 'ACG_GTTT'),
                ('no indel', 'other', 'ambiguou'),
        ]

        groups = {
            name: [o for o in self.most_frequent_outcomes if condition(*o)] if name != 'uncommon' else condition
            for name, condition in conditions.items()
        }

        groups['donor'] = sorted(groups['donor'], key=donor_order.index)

        ordered = []
        for name in group_order:
            ordered.extend(groups[name])

        sizes = [len(groups[name]) for name in group_order]
        return ordered, sizes

    def explore(self, **kwargs):
        return explore(self.base_dir, self.group, **kwargs)

def explore(base_dir, group, initial_guide=None, by_outcome=False, **kwargs):
    pool = PooledScreen(base_dir, group)

    guides = pool.guides
    if initial_guide is None:
        initial_guide = guides[0]

    widgets = {
        'guide': ipywidgets.Select(options=guides, value=initial_guide, layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='600px')),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
        'zoom_in': ipywidgets.FloatRangeSlider(value=[-0.02, 1.02], min=-0.02, max=1.02, step=0.001, continuous_update=False, layout=ipywidgets.Layout(width='1200px')),
        'save': ipywidgets.Button(description='Save'),
        'file_name': ipywidgets.Text(value=str(base_dir / 'figures')),
    }

    toggles = [
        'parsimonious',
        'relevant',
        'ref_centric',
        'draw_sequence',
        'draw_qualities',
        'draw_mismatches',
        'highlight_SNPs',
        'highlight_around_cut',
    ]
    for toggle in toggles:
        widgets[toggle] = ipywidgets.ToggleButton(value=kwargs.get(toggle, False))

    def save(change):
        fig = interactive.result
        fn = widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

    widgets['save'].on_click(save)

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    output = ipywidgets.Output()

    def get_exp():
        guide = widgets['guide'].value
        exp = SingleGuideExperiment(base_dir, group, guide)
        return exp

    @output.capture()
    def populate_outcomes(change):
        previous_value = widgets['outcome'].value

        exp = get_exp()

        outcomes = {(c, sc) for c, sc, d in exp.perfect_guide_outcome_counts.index.values}

        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in sorted(outcomes)]
        if outcomes:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    @output.capture()
    def populate_read_ids(change):
        exp = get_exp()

        df = exp.filtered_cell_outcomes

        if exp is None:
            return

        if by_outcome:
            outcome = widgets['outcome'].value
            if outcome is None:
                qnames = []
            else:
                category, subcategory = outcome
                right_outcome = df.query('category == @category and subcategory == @subcategory')
                qnames = right_outcome['original_name'].values[:200]
        else:
            qnames = df['original_name'].values[:200]

        widgets['read_id'].options = qnames

        if len(qnames) > 0:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    if by_outcome:
        populate_outcomes({'name': 'initial'})

    populate_read_ids({'name': 'initial'})

    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
        widgets['guide'].observe(populate_outcomes, names='value')
    else:
        widgets['guide'].observe(populate_read_ids, names='value')

    @output.capture(clear_output=False)
    def plot(guide, read_id, **plot_kwargs):
        exp = get_exp()

        if exp is None:
            return

        if by_outcome:
            als = exp.get_read_alignments(read_id, outcome=plot_kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        l = pooled_layout.Layout(als, exp.target_info)
        info = l.categorize()
        if widgets['relevant'].value:
            als = l.relevant_alignments

        diagram = visualize.ReadDiagram(als, exp.target_info,
                                        size_multiple=kwargs.get('size_multiple', 1),
                                        max_qual=exp.max_qual,
                                        flip_target=True,
                                        **plot_kwargs)
        fig = diagram.fig

        fig.axes[0].set_title(' '.join((l.name,) + info))

        if widgets['draw_sequence'].value:
            print(als[0].query_name)
            print(als[0].get_forward_sequence())

        return diagram.fig

    # Make a version of the widgets dictionary that excludes non-plot arguments.
    most_widgets = widgets.copy()
    most_widgets.pop('save')
    most_widgets.pop('file_name')

    interactive = ipywidgets.interactive(plot, **most_widgets)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['guide', 'outcome', 'read_id']
    else:
        top_row_keys = ['guide', 'read_id']

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row(toggles),
         make_row(['save',
                   'file_name',
                  ]),
         #widgets['zoom_in'],
         interactive.children[-1],
         output,
        ],
    )

    return layout