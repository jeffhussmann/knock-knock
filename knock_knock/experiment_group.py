import bisect
import datetime
import logging
import multiprocessing
import os

import numpy as np
import pandas as pd
import pysam
import scipy.sparse
import seaborn as sns

import hits.visualize
from hits import utilities
memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

import knock_knock.visualize.stacked
import knock_knock.visualize.rejoining_boundaries
import knock_knock.parallel

import knock_knock.common_sequences

def run_stage(GroupClass, group_args, sample_name, stage):
    group = GroupClass(*group_args)

    if sample_name in group.sample_names:
        # Important to do this branch first, since preprocessing happens before common sequence collection.
        exp = group.sample_name_to_experiment(sample_name, no_progress=True)
    elif sample_name in group.common_sequence_chunk_exp_names:
        exp = group.common_sequence_chunk_exp_from_name(sample_name)
    else:
        raise ValueError(sample_name)

    logging.info(f'Starting {sample_name} {stage}')
    exp.process(stage=stage)
    logging.info(f'Finished {sample_name} {stage}')

class ExperimentGroup:
    def __init__(self):
        self.fns = {
            'common_sequences_dir': self.results_dir / 'common_sequences',
            'common_sequence_outcomes': self.results_dir / 'common_sequences' / 'common_sequence_outcomes.txt',
            'common_sequence_special_alignments': self.results_dir / 'common_sequences' / 'all_special_alignments.bam',

            'total_outcome_counts': self.results_dir / 'total_outcome_counts.txt',
            'outcome_counts': self.results_dir  / 'outcome_counts.npz',

            'genomic_insertion_length_distributions': self.results_dir / 'genomic_insertion_length_distribution.txt',
             
            'partial_incorporation_figure': self.results_dir / 'partial_incorporation.pdf',
            'deletion_boundaries_figure': self.results_dir / 'deletion_boundaries.pdf',
        }

    def process(self, generate_figures=False, num_processes=18, verbose=True, use_logger_thread=False):
        self.results_dir.mkdir(exist_ok=True, parents=True)
        log_fn = self.results_dir / f'log_{datetime.datetime.now():%y%m%d-%H%M%S}.out'

        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_fn)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S',
                                     )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        if verbose:
            print(f'Logging in {log_fn}')

        if use_logger_thread:
            pool = knock_knock.parallel.PoolWithLoggerThread(num_processes, logger)
        else:
            NICENESS = 3
            pool = multiprocessing.Pool(num_processes, maxtasksperchild=1, initializer=os.nice, initargs=(NICENESS,))

        with pool:
            logger.info('Preprocessing')

            args = [(type(self), self.group_args, sample_name, 'preprocess') for sample_name in self.sample_names]
            pool.starmap(run_stage, args)

            self.make_common_sequences()

            for stage in [
                'align',
                'categorize',
            ]:

                logger.info(f'Processing common sequences, stage {stage}')
                args = [(type(self), self.group_args, chunk_exp_name, stage) for chunk_exp_name in self.common_sequence_chunk_exp_names]
                pool.starmap(run_stage, args)

            self.merge_common_sequence_outcomes()

            stages = [
                'align',
                'categorize',
            ]

            if generate_figures:
                stages.append('generate_figures')

            for stage in stages:
                logger.info(f'Processing unique sequences, stage {stage}')
                args = [(type(self), self.group_args, sample_name, stage) for sample_name in self.sample_names]
                pool.starmap(run_stage, args)

        logger.info('Collecting outcome counts')
        self.make_outcome_counts()

        if generate_figures:
            self.make_group_figures()

        logger.info('Done!')

        logger.removeHandler(file_handler)
        file_handler.close()

    def make_common_sequences(self):
        ''' Identify all sequences that occur more than once across preprocessed
        reads for all experiments in the group and write them into common sequences
        experiments to be categorized.
        '''
        splitter = knock_knock.common_sequences.CommonSequenceSplitter(self, max_sequences=20000)

        description = 'Collecting common sequences'
        exps = self.experiments(no_progress=True)
        for exp in self.progress(exps, desc=description, total=self.num_experiments):
            reads = exp.reads_by_type(self.preprocessed_read_type)
            splitter.update_counts((read.seq for read in reads))

        splitter.write_files()

    @memoized_property
    def common_sequence_chunk_exp_names(self):
        ''' Names of all common sequence chunk experiments. '''
        return sorted([d.name for d in self.fns['common_sequences_dir'].iterdir() if d.is_dir()])

    def common_sequence_chunk_exps(self):
        ''' Iterator over common sequence chunk experiments. ''' 
        for chunk_name in self.common_sequence_chunk_exp_names:
            yield self.common_sequence_chunk_exp_from_name(chunk_name)

    @memoized_property
    def common_names(self):
        ''' List of all names assigned to common sequence artificial reads. '''
        return [outcome.query_name for outcome in self.common_sequence_outcomes]

    @memoized_property
    def common_sequence_outcomes(self):
        outcomes = []
        for exp in self.common_sequence_chunk_exps():
            for outcome in exp.outcome_iter():
                outcomes.append(outcome)

        return outcomes

    @memoized_property
    def common_name_to_common_sequence(self):
        name_to_seq = {}
        for outcome in self.common_sequence_outcomes:
            name_to_seq[outcome.query_name] = outcome.seq

        return name_to_seq

    @memoized_property
    def common_sequence_to_common_name(self):
        return utilities.reverse_dictionary(self.common_name_to_common_sequence)

    @memoized_property
    def common_name_to_special_alignment(self):
        name_to_al = {}

        if self.fns['common_sequence_special_alignments'].exists():
            for al in pysam.AlignmentFile(self.fns['common_sequence_special_alignments']):
                name_to_al[al.query_name] = al

        return name_to_al

    @memoized_property
    def common_sequence_to_outcome(self):
        common_sequence_to_outcome = {}
        for outcome in self.common_sequence_outcomes:
            common_sequence_to_outcome[outcome.seq] = outcome

        return common_sequence_to_outcome

    def merge_common_sequence_outcomes(self):
        with self.fns['common_sequence_outcomes'].open('w') as fh:
            for outcome in self.common_sequence_outcomes:
                fh.write(f'{outcome}\n')

    @memoized_property
    def name_to_chunk(self):
        chunks = list(self.common_sequence_chunk_exps())
        starts = [int(chunk.sample_name.split('-')[0]) for chunk in chunks]

        def name_to_chunk(name):
            number = int(name.split('_')[0])
            start_index = bisect.bisect(starts, number) - 1 
            chunk = chunks[start_index]
            return chunk

        return name_to_chunk

    def get_common_seq_alignments(self, seq):
        name = self.common_sequence_to_common_name[seq]
        als = self.get_read_alignments(name)
        return als

    @memoized_property
    def common_sequence_to_alignments(self):
        common_sequence_to_alignments = {}
        for chunk_exp in self.common_sequence_chunk_exps():
            for common_name, als in chunk_exp.alignment_groups():
                seq = self.common_name_to_common_sequence[common_name]
                common_sequence_to_alignments[seq] = als
        return common_sequence_to_alignments

    def get_read_alignments(self, name):
        if isinstance(name, int):
            name = self.common_names[name]

        chunk = self.name_to_chunk(name)

        als = chunk.get_read_alignments(name)

        return als

    def get_read_layout(self, name, **kwargs):
        als = self.get_read_alignments(name)
        layout = self.categorizer(als, self.target_info, mode=self.layout_mode, error_corrected=False, **kwargs)
        return layout

    def get_read_diagram(self, read_id, relevant=True, **diagram_kwargs):
        layout = self.get_read_layout(read_id)

        if relevant:
            layout.categorize()
            
        diagram = layout.plot(relevant=relevant, **diagram_kwargs)

        return diagram

    def make_group_figures(self):
        self.make_partial_incorporation_figure()
        self.make_deletion_boundaries_figure()

    def make_partial_incorporation_figure(self,
                                          conditions=None,
                                          frequency_cutoff=1e-3,
                                          show_log_scale=True,
                                          manual_window=None,
                                          min_reads=None,
                                          unique_colors=False,
                                          **diagram_kwargs,
                                         ):
        ti = self.target_info

        if len(ti.pegRNA_names) == 0:
            return

        if unique_colors:
            colors = {condition: f'C{i}' for i, condition in enumerate(self.full_conditions)}
        else:
            colors = self.condition_colors()

        if conditions is None:
            conditions = self.full_conditions

        if min_reads is not None:
            conditions = [c for c in conditions if self.total_valid_reads.loc[c] >= min_reads]

        color_overrides = {primer_name: 'grey' for primer_name in ti.primers}

        for pegRNA_name in ti.pegRNA_names:
            ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
            PAM_name = f'{pegRNA_name}_PAM'
            color = ti.pegRNA_name_to_color[pegRNA_name]
            light_color = hits.visualize.apply_alpha(color, 0.5)
            color_overrides[ps_name] = light_color
            color_overrides[PAM_name] = color

        if manual_window is not None:
            window = manual_window
        elif len(ti.pegRNA_names) == 2:
            window = (-20, ti.nick_offset + 20 - 1)
        elif len(ti.pegRNA_names) == 1:
            window = (-20, len(ti.sgRNA_components[ti.pegRNA_names[0]]['RTT']) + 4)
        else:
            raise ValueError

        if ti.protospacer_feature.strand == '+':
            window_interval = hits.interval.Interval(ti.cut_after + window[0], ti.cut_after + window[1])
        else:
            window_interval = hits.interval.Interval(ti.cut_after - window[1], ti.cut_after - window[0])

        def mismatch_in_window(d):
            if d == 'n/a':
                return False
            else:
                SNVs = knock_knock.outcome.MismatchOutcome.from_string(d).undo_anchor_shift(ti.anchor).snvs
                return any(p in window_interval for p in SNVs.positions)

        fs = self.outcome_fractions[conditions]

        outcomes = [(c, s, d) for (c, s, d), f_row in fs.iterrows()
                    if (((c, s) in {('wild type', 'mismatches')} and mismatch_in_window(d))
                        or c in {'intended edit', 'partial replacement', 'partial edit'}
                       )
                    and max(f_row) > frequency_cutoff
                   ]
        outcomes = fs.loc[outcomes].mean(axis=1).sort_values(ascending=False).index

        if len(outcomes) > 0:
            grid = knock_knock.visualize.stacked.DiagramGrid(outcomes, 
                                                             ti,
                                                             draw_wild_type_on_top=True,
                                                             window=window,
                                                             block_alpha=0.1,
                                                             color_overrides=color_overrides,
                                                             draw_all_sequence=0.1,
                                                             **diagram_kwargs,
                                                            )

            grid.add_ax('fractions', width_multiple=12, title='% of reads')

            if show_log_scale:
                grid.add_ax('log10_fractions', width_multiple=12, gap_multiple=2, title='% of reads (log scale)')

            ax_params = [
                ('fractions', 'percentage'),
                ('log10_fractions', 'log10'),
            ]

            for ax, transform in ax_params:
                for condition in fs:

                    sample_name = self.full_condition_to_sample_name[condition]
                    label = f'{" ".join(condition)} ({sample_name}, {self.total_valid_reads[condition]:,} total reads)'

                    grid.plot_on_ax(ax,
                                    fs[condition],
                                    transform=transform,
                                    color=colors.get(condition, 'black'),
                                    line_alpha=0.75,
                                    linewidth=1.5,
                                    markersize=7,
                                    fill=0,
                                    label=label,
                                   )

            grid.style_frequency_ax('fractions')

            grid.set_xlim('fractions', (0,))
            x_max = (grid.axs_by_name['fractions'].get_xlim()[1] / 100) * 1.01

            grid.set_xlim('log10_fractions', (np.log10(0.49 * frequency_cutoff), np.log10(x_max)))
            grid.style_log10_frequency_ax('log10_fractions')

            for pegRNA_i, pegRNA_name in enumerate(ti.pegRNA_names):
                grid.diagrams.draw_pegRNA(ti.name, pegRNA_name, y_offset=pegRNA_i + 1, label_features=False)

            grid.plot_pegRNA_conversion_fractions_above(self.target_info,
                                                        self.pegRNA_conversion_fractions,
                                                        conditions=conditions,
                                                        condition_colors=colors,
                                                       )

            grid.style_pegRNA_conversion_plot('pegRNA_conversion_fractions')

            grid.axs_by_name['pegRNA_conversion_fractions'].set_title(self.group_name, y=1.2)

            grid.ordered_axs[-1].legend(bbox_to_anchor=(1, 1))

            grid.fig.savefig(self.fns['partial_incorporation_figure'], bbox_inches='tight')

            return grid

    def make_deletion_boundaries_figure(self,
                                        frequency_cutoff=5e-4,
                                        conditions=None,
                                        include_simple_deletions=True,
                                        include_edit_plus_deletions=True,
                                        include_insertions=False,
                                        include_multiple_indels=False,
                                        min_reads=None,
                                        show_log_scale=False,
                                        unique_colors=False,
                                       ):
        ti = self.target_info
        if ti.primary_protospacer is None:
            return

        if unique_colors:
            colors = {condition: f'C{i}' for i, condition in enumerate(self.full_conditions)}
        else:
            colors = self.condition_colors()

        if conditions is None:
            conditions = self.full_conditions

        if min_reads is not None:
            conditions = [c for c in conditions if self.total_valid_reads.loc[c] >= min_reads]

        fs = self.outcome_fractions[conditions]

        if 'deletion' in fs.index.levels[0]:
            deletions = fs.xs('deletion', drop_level=False).groupby('details').sum()
            deletions.index = pd.MultiIndex.from_tuples([('deletion', 'collapsed', details) for details in deletions.index])
        else:
            deletions = None

        if 'insertion' in fs.index.levels[0]:
            insertions = fs.xs('insertion', drop_level=False).groupby('details').sum()
            insertions.index = pd.MultiIndex.from_tuples([('insertion', 'collapsed', details) for details in insertions.index])
        else:
            insertions = None

        multiple_indels = fs.xs('multiple indels', level=1, drop_level=False)

        edit_plus_deletions = [(c, s, d) for c, s, d in fs.index
                               if (c, s) == ('edit + indel', 'deletion')
                              ] 

        to_concat = []

        if include_simple_deletions and deletions is not None:
            to_concat.append(deletions)

        if include_edit_plus_deletions:
            to_concat.append(fs.loc[edit_plus_deletions])

        if include_insertions and insertions is not None:
            to_concat.append(insertions)

        fs = pd.concat(to_concat)

        outcomes = [(c, s, d) for (c, s, d), f_row in fs.iterrows()
                    if max(f_row) > frequency_cutoff
                   ]
        outcomes = fs.loc[outcomes].mean(axis=1).sort_values(ascending=False).index

        color_overrides = {primer_name: 'grey' for primer_name in ti.primers}

        for pegRNA_name in ti.pegRNA_names:
            ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
            PAM_name = f'{pegRNA_name}_PAM'
            color = ti.pegRNA_name_to_color[pegRNA_name]
            light_color = hits.visualize.apply_alpha(color, 0.5)
            color_overrides[ps_name] = light_color
            color_overrides[PAM_name] = color

        flip = (ti.features[ti.target, ti.primary_protospacer].strand == '-')

        if flip:
            window = (ti.cut_after - ti.amplicon_interval.end, ti.cut_after - ti.amplicon_interval.start)
        else:
            window = (ti.amplicon_interval.start - ti.cut_after, ti.amplicon_interval.end - ti.cut_after)
            
        window = (window[0] - 5, window[1] + 5)
            
        grid = knock_knock.visualize.stacked.DiagramGrid(outcomes, 
                                                         ti,
                                                         draw_wild_type_on_top=True,
                                                         window=window,
                                                         block_alpha=0.1,
                                                         color_overrides=color_overrides,
                                                         features_to_draw=sorted(ti.primers), 
                                                        )

        grid.add_ax('fractions', width_multiple=12, title='% of reads')

        if show_log_scale:
            grid.add_ax('log10_fractions', width_multiple=12, gap_multiple=2, title='% of reads (log scale)')

        for ax, transform in [('fractions', 'percentage'),
                              ('log10_fractions', 'log10'),
                             ]:

            for condition in fs:
                label = f'{" ".join(condition)} ({self.total_valid_reads[condition]:,} total reads)'

                grid.plot_on_ax(ax,
                                fs[condition],
                                transform=transform,
                                color=colors.get(condition, 'black'),
                                line_alpha=0.75,
                                linewidth=1.5,
                                markersize=7,
                                label=label,
                                clip_on=True,
                               )

        grid.style_frequency_ax('fractions')
        grid.ordered_axs[-1].legend(bbox_to_anchor=(1, 1), loc='upper left')

        grid.set_xlim('fractions', (0,))
        x_max = (grid.axs_by_name['fractions'].get_xlim()[1] / 100) * 1.01

        grid.set_xlim('log10_fractions', (np.log10(0.49 * frequency_cutoff), np.log10(x_max)))
        grid.style_log10_frequency_ax('log10_fractions')

        if flip:
            panel_order = [
                'fraction_removed',
                'stops',
                'starts',
            ]
        else:
            panel_order = [
                'fraction_removed',
                'starts',
                'stops',
            ]

        deletion_boundaries = self.deletion_boundaries(include_simple_deletions=include_simple_deletions,
                                                       include_edit_plus_deletions=include_edit_plus_deletions,
                                                      )

        for quantity in panel_order:
            grid.add_ax_above(quantity,
                              gap=6 if quantity == panel_order[0] else 2,
                              height_multiple=15 if quantity == 'fraction_removed' else 7,
                             )

            for condition in conditions:
                series = deletion_boundaries[quantity][condition].copy()
                series.index = series.index.values - grid.diagrams.offsets[ti.name]
                grid.plot_on_ax_above(quantity,
                                      series.index,
                                      series * 100,
                                      markersize=3 if quantity != 'fraction_removed' else 0,
                                      linewidth=1 if quantity != 'fraction_removed' else 2,
                                      color=colors.get(condition, 'black'),
                                     )

            for cut_after in ti.cut_afters.values():
                grid.axs_by_name[quantity].axvline(cut_after + 0.5 - grid.diagrams.offsets[ti.name], linestyle='--', color=grid.diagrams.cut_color, linewidth=grid.diagrams.line_widths)

        for pegRNA_i, pegRNA_name in enumerate(ti.pegRNA_names):
            grid.diagrams.draw_pegRNA(ti.name, pegRNA_name, y_offset=pegRNA_i + 1, label_features=False)

        grid.axs_by_name['fraction_removed'].set_ylabel('% of reads with\nposition deleted', size=14)
        grid.axs_by_name[panel_order[1]].set_ylabel('% of reads\nwith deletion\nstarting at', size=14)
        grid.axs_by_name[panel_order[2]].set_ylabel('% of reads\nwith deletion\nending at', size=14)

        for panel in panel_order:
            grid.axs_by_name[panel].set_ylim(0)

        grid.fig.savefig(self.fns['deletion_boundaries_figure'], bbox_inches='tight')

        return grid

    def make_single_flap_extension_chain_edge_figure(self,
                                                     palette='inferno',
                                                     conditions=None,
                                                     aggregate_replicates=True,
                                                     **plot_kwargs,
                                                    ):

        if aggregate_replicates:
            if conditions is None:
                conditions = self.conditions

            exp_sets = {}

            for condition in conditions:
                sample_names = self.condition_to_sample_names[condition]
                
                boundaries = knock_knock.visualize.rejoining_boundaries.BoundaryProperties()
                for sample_name in sample_names:
                    exp = self.sample_name_to_experiment(sample_name)
                    boundaries.count_single_flap_boundaries(exp)

                exp_sets[condition] = {
                    'color': self.condition_colors(palette=palette)[condition],
                    'results': boundaries,
                }
        else:
            if conditions is None:
                conditions = self.full_conditions

            exp_sets = {}

            for condition in conditions:
                exp = self.full_condition_to_experiment[condition]
                
                boundaries = knock_knock.visualize.rejoining_boundaries.BoundaryProperties()
                boundaries.count_single_flap_boundaries(exp)

                key = f'{",".join(condition)} ({exp.sample_name})'

                exp_sets[key] = {
                    'color': self.condition_colors(palette=palette)[condition],
                    'results': boundaries,
                }

        fig, axs = knock_knock.visualize.rejoining_boundaries.plot_single_flap_extension_chain_edges(exp.target_info,
                                                                                                     exp_sets,
                                                                                                     **plot_kwargs,
                                                                                                    ) 
        
        fig.suptitle(f'{self.target_info.target} - {",".join(self.target_info.sgRNAs)}')

        return fig, axs

    def make_outcome_counts(self):
        all_counts = {}

        description = 'Loading outcome counts'
        items = self.progress(self.full_condition_to_experiment.items(), desc=description)
        for condition, exp in items:
            try:
                all_counts[condition] = exp.outcome_counts
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        all_outcomes = set()

        for counts in all_counts.values():
            all_outcomes.update(counts.index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(self.full_conditions)), dtype=int)

        description = 'Combining outcome counts'
        full_conditions = self.progress(self.full_conditions, desc=description)

        for c_i, condition in enumerate(full_conditions):
            if condition in all_counts:
                for outcome, count in all_counts[condition].items():
                    o_i = outcome_to_index[outcome]
                    counts[o_i, c_i] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.toarray(),
                          columns=self.full_conditions,
                          index=pd.MultiIndex.from_tuples(outcome_order),
                         )

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'], header=False)

        ## Collapse potentially equivalent outcomes together.
        #collapsed = pd.concat({pg: collapse_categories(df.loc[pg]) for pg in [True, False] if pg in df.index.levels[0]})

        #coo = scipy.sparse.coo_matrix(np.array(collapsed))
        #scipy.sparse.save_npz(self.fns['collapsed_outcome_counts'], coo)

        #collapsed.sum(axis=1).to_csv(self.fns['collapsed_total_outcome_counts'], header=False)

    @memoized_with_args
    def outcome_counts_df(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'outcome_counts'

        sparse_counts = scipy.sparse.load_npz(self.fns[key])
        df = pd.DataFrame(sparse_counts.toarray(),
                          index=self.total_outcome_counts(collapsed).index,
                          columns=pd.MultiIndex.from_tuples(self.full_conditions),
                         )

        df.index.names = self.outcome_index_levels
        df.columns.names = self.outcome_column_levels

        return df

    @memoized_with_args
    def total_outcome_counts(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'total_outcome_counts'

        return pd.read_csv(self.fns[key], header=None, index_col=list(range(len(self.outcome_index_levels))), na_filter=False)

    @memoized_property
    def genomic_insertion_length_distributions(self):
        df = pd.read_csv(self.fns['genomic_insertion_length_distributions'], index_col=[0, 1, 2])
        df.columns = [int(c) for c in df.columns]
        return df