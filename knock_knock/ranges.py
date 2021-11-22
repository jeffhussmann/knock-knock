import copy
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hits.utilities
import hits.visualize
import knock_knock.target_info
import knock_knock.outcome_record
import knock_knock.outcome

import repair_seq.pooled_layout
import repair_seq.prime_editing_layout

memoized_property = hits.utilities.memoized_property
idx = pd.IndexSlice

def combine_ranges(ranges_list):
    starts = itertools.chain.from_iterable([r.starts for r in ranges_list])
    ends = itertools.chain.from_iterable([r.ends for r in ranges_list])
    total_reads = sum(r.total_reads for r in ranges_list)

    empty_read_info = ('', '', '')
    range_iter = ((empty_read_info, start, end) for start, end in zip(starts, ends))
    return Ranges(ranges_list[0].target_info, ranges_list[0].sequence_name, range_iter, total_reads)

class Ranges:
    def __init__(self, target_info, sequence_name, range_iter, total_reads, exps=None):
        self.target_info = target_info

        if isinstance(sequence_name, str):
            sequence_name = {
                'start': sequence_name,
                'end': sequence_name,
            }

        self.sequence_name = sequence_name
        self.sequence = {edge: target_info.reference_sequences[n] for edge, n in self.sequence_name.items()}
        self.sequence_length = {edge: len(s) for edge, s in self.sequence.items()}

        if exps is None:
            exps = []
        self.exps = {(exp.batch, exp.sample_name): exp for exp in exps}

        self.total_reads = total_reads
        
        self.edge_counts = {edge: np.zeros(length, int) for edge, length in self.sequence_length.items()}

        self.edge_pairs = []
        self.starts = []
        self.ends = []

        self.read_ids = []
        
        if sequence_name['start'] == sequence_name['end']:
            self.positions_involved = np.zeros(self.sequence_length['start'])
        
        for read_id, start, end in range_iter:
            self.edge_counts['start'][start] += 1
            self.edge_counts['end'][end] += 1
            
            self.starts.append(start)
            self.ends.append(end)
            self.edge_pairs.append((start, end))

            self.read_ids.append(read_id)

            if end < start:
                # Deletion junctions flip the meaning of start and end.
                # For positions_involved to be useful, flip them back
                start, end = end + 1, start - 1

            if sequence_name['start'] == sequence_name['end']:
                self.positions_involved[start:end + 1] += 1
            
        self.cumulative_counts = {edge: {} for edge in self.edge_counts}
        
        for edge in ['start', 'end']:
            self.cumulative_counts[edge]['from_left'] = np.cumsum(self.edge_counts[edge])
            self.cumulative_counts[edge]['from_right'] = np.cumsum(self.edge_counts[edge][::-1])[::-1]

    @memoized_property
    def joint_counts(self):
        counts = np.zeros((self.sequence_length['start'], self.sequence_length['end']))

        for start, end in self.edge_pairs:
            counts[start, end] += 1

        return counts

    @memoized_property
    def joint_percentages(self):
        return self.joint_counts / self.total_reads * 100

    @memoized_property
    def edge_percentages(self):
        return {edge: self.edge_counts[edge] / self.total_reads * 100 for edge in self.edge_counts}

    @memoized_property
    def total_percentage(self):
        return len(self.edge_pairs) / self.total_reads * 100

    @memoized_property
    def percentages_involved(self):
        return self.positions_involved / self.total_reads * 100

    @memoized_property
    def differences(self):
        counts = np.zeros(self.sequence_length['start'] + 1)

        for start, end in self.edge_pairs:
            counts[end - start + 1] += 1

        return counts

    @memoized_property
    def df(self):
        df = pd.DataFrame({
            'group': [g for g, n, q in self.read_ids],
            'name': [n for g, n, q in self.read_ids],
            'query_name': [q for g, n, q in self.read_ids],
            'start': [s for s, e in self.edge_pairs],
            'end': [e for s, e in self.edge_pairs],
        })
        return df    
            
    def downsampled_edge_pairs(self, n):
        return hits.utilities.reservoir_sample(self.edge_pairs, n)

    def most_common_edge_pairs(self, start_p=None, end_p=None):
        if start_p is not None:
            if isinstance(start_p, int):
                start_p = (start_p, start_p)
            start_p_min, start_p_max = start_p
        else:
            start_p_min, start_p_max = -np.inf, np.inf

        if end_p is not None:
            if isinstance(end_p, int):
                end_p = (end_p, end_p)
            end_p_min, end_p_max = end_p    
        else:
            end_p_min, end_p_max = -np.inf, np.inf

        edge_pairs = Counter(self.edge_pairs).most_common()

        edge_pairs = [((start, end), count) for (start, end), count in edge_pairs
                      if start_p_min <= start <= start_p_max
                      and end_p_min <= end <= end_p_max
                     ]

        return edge_pairs

    def edge_pair_percentages(self):
        return pd.Series(Counter(self.edge_pairs)) / self.total_reads * 100

    def get_example_diagram(self, start_p=None, end_p=None, example_num=1, **diagram_kwargs):
        constraints = []

        if start_p is not None:
            if isinstance(start_p, int):
                start_p = (start_p, start_p)
            start_p_min, start_p_max = start_p

            constraints.append('@start_p_min <= start <= @start_p_max')

        if end_p is not None:
            if isinstance(end_p, int):
                end_p = (end_p, end_p)
            end_p_min, end_p_max = end_p    

            constraints.append('@end_p_min <= end <= @end_p_max')

        if constraints:
            query = ' and '.join(constraints)
            rows = self.df.query(query)
        else:
            rows = self.df

        if example_num <= len(rows):
            row = rows.iloc[example_num - 1]
            print(row)
            exp = self.exps[row['group'], row['name']]
            
            # Hack for now
            exp.target_info.features_to_show.update(set(self.target_info.PAM_features))
            
            exp_diagram_kwargs = copy.deepcopy(exp.diagram_kwargs)
            exp_diagram_kwargs.update(diagram_kwargs)

            return exp.get_read_diagram(row['query_name'], **exp_diagram_kwargs)
        else:
            return None

    def most_common_example_diagram(self, pair_rank=1, example_num=1, **diagram_kwargs):
        (start_p, end_p), count = self.most_common_edge_pairs()[pair_rank - 1]
        return self.get_example_diagram(start_p, end_p, example_num=example_num, **diagram_kwargs)
            
    @classmethod
    def deletion_ranges(cls, exps,
                        as_junctions=False,
                        without_edit=True,
                        with_edit=False,
                        exclude_buffer_around_primers=None,
                       ):
        ranges = []
        total_reads = 0

        with_edit_subcategories = [('edit + indel', 'edit + deletion')]
        def get_with_edit_deletion(outcome):
            full_Outcome = repair_seq.prime_editing_layout.HDRPlusDeletionOutcome.from_string(outcome.details)
            deletion = full_Outcome.deletion_outcome.deletion
            return deletion

        without_edit_subcategories = [('deletion', 'clean'), ('deletion', 'mismatches')]
        def get_without_edit_deletion(outcome):
            deletion = knock_knock.target_info.DegenerateDeletion.from_string(outcome.details)
            return deletion

        for exp in exps:
            target_info = exp.target_info

            if exclude_buffer_around_primers is None:
                primer_intervals = hits.interval.Interval.empty()
            else:
                primer_intervals = []
                for name, feature in target_info.primers.items():
                    interval = hits.interval.Interval(feature.start - exclude_buffer_around_primers, feature.end + exclude_buffer_around_primers)
                    primer_intervals.append(interval)

                primer_intervals = hits.interval.DisjointIntervals(primer_intervals)

            for outcome in exp.outcome_iter():
                if outcome.category != 'nonspecific amplification':
                    total_reads += 1

                cs = (outcome.category, outcome.subcategory)
                if with_edit and cs in with_edit_subcategories:
                    deletion = get_with_edit_deletion(outcome)
                elif without_edit and cs in without_edit_subcategories:
                    deletion = get_without_edit_deletion(outcome)
                else:
                    continue

                start, end = deletion.starts_ats[0], deletion.ends_ats[0]
                start += exp.target_info.anchor
                end += exp.target_info.anchor

                if start in primer_intervals or end in primer_intervals:
                    continue

                if as_junctions:
                    # Treat as end/start of remaining sequence, rather than start/end of removed.
                    start, end = end + 1, start - 1

                read_id = (exp.group, exp.sample_name, outcome.query_name)
                ranges.append((read_id, start, end))
                
        return cls(target_info, target_info.target, ranges, total_reads, exps)

    @classmethod
    def insertion_ranges(cls, exps):
        ranges = []
        total_reads = 0
        
        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1
                if outcome.category == 'insertion':
                    insertion = knock_knock.target_info.DegenerateInsertion.from_string(outcome.details)

                    start = insertion.starts_afters[0] + exp.target_info.anchor
                    end = start + 1

                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id, start, end))
                
        return cls(exp.target_info, exp.target_info.target, ranges, total_reads, exps)

    @classmethod
    def deletion_with_edit_ranges(cls, exps):
        ranges = []
        total_reads = 0
        
        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1
                if outcome.category == 'edit + indel' and outcome.subcategory == 'edit + deletion':

                    full_Outcome = repair_seq.prime_editing_layout.HDRPlusDeletionOutcome.from_string(outcome.details)
                    deletion = full_Outcome.deletion_outcome.deletion

                    start, end = deletion.starts_ats[0], deletion.ends_ats[0]
                    start += exp.target_info.anchor
                    end += exp.target_info.anchor

                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id, start, end))
                
        return cls(exp.target_info, exp.target_info.target, ranges, total_reads, exps)
    
    @classmethod
    def duplication_ranges(cls, exps):
        ranges = []
        total_reads = 0
        
        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1
                if outcome.category == 'duplication' and outcome.subcategory == 'simple':
                    dup = repair_seq.prime_editing_layout.DuplicationOutcome.from_string(outcome.details)

                    start, end = sorted(dup.ref_junctions[0])
                    start += exp.target_info.anchor
                    end += exp.target_info.anchor

                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id, start, end))
        
        return cls(exp.target_info, {'start': exp.target_info.target, 'end': exp.target_info.target}, ranges, total_reads, exps)

    @classmethod
    def duplication_junctions(cls, exps, has_edit=False):
        ranges = []
        total_reads = 0

        if has_edit:
            category = 'edit + duplication'
        else:
            category = 'duplication'

        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1
                if outcome.category == category and outcome.subcategory == 'simple':
                    duplication = knock_knock.outcome.DuplicationOutcome.from_string(outcome.details).undo_anchor_shift(exp.target_info.anchor)

                    (starts, ends), = duplication.ref_junctions
                    start = starts[0]
                    end = ends[0]

                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id, start, end))

        return cls(exp.target_info, {'start': exp.target_info.target, 'end': exp.target_info.target}, ranges, total_reads, exps)
                    
    @classmethod
    def donor_ranges(cls, exps, component='insertion', subcategories=None, strand=None):
        ranges = []
        total_reads = 0

        for exp in exps:
            sequencing_direction = exp.target_info.sequencing_direction
            sgRNA_strand = exp.target_info.sgRNA_feature.strand

            left_primer = exp.target_info.primers_by_side_of_read['left']
            right_primer = exp.target_info.primers_by_side_of_read['right']

            if component == 'insertion':
                sequence_name = exp.target_info.donor
            elif component == 'target':
                sequence_name = exp.target_info.target
            else:
                raise ValueError
        
            for outcome in exp.outcome_iter():
                if outcome.category != 'nonspecific amplification':
                    total_reads += 1

                if (outcome.category, outcome.subcategory) in subcategories:
                    lti = repair_seq.pooled_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)

                    if component == 'insertion':
                        left = lti.left_insertion_ref_bound
                        right = lti.right_insertion_ref_bound

                        if lti.strand != strand:
                            continue

                        start, end = sorted([right, left])

                    elif component == 'target':
                        if sequencing_direction == '+' and sgRNA_strand == '-':
                            start = lti.right_target_ref_bound
                            end = lti.left_target_ref_bound

                        elif sequencing_direction == '-' and sgRNA_strand == '+':
                            start = lti.left_target_ref_bound
                            end = lti.right_target_ref_bound

                        elif sequencing_direction == '+' and sgRNA_strand == '+':
                            start = lti.right_target_ref_bound
                            end = lti.left_target_ref_bound

                        else:
                            raise NotImplementedError
                    else:
                        raise ValueError

                    if start == knock_knock.outcome.NAN_INT or end == knock_knock.outcome.NAN_INT:
                        continue
                 
                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id, start, end))
                                            
        return cls(exp.target_info, sequence_name, ranges, total_reads, exps)

    @classmethod
    def genomic_insertion_boundaries(cls, exps, subcategories=None, min_length=0, max_length=np.inf):
        if subcategories is None:
            subcategories = [('genomic insertion', 'hg19')]

        ranges = []
        total_reads = 0

        for exp in exps:
            for outcome in exp.outcome_iter():
                if outcome.category != 'nonspecific amplification':
                    total_reads += 1

                if (outcome.category, outcome.subcategory) in subcategories:
                    lti = repair_seq.pooled_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)

                    length = lti.insertion_length()
                    if min_length <= length <= max_length:
                        start, end = sorted([lti.right_target_ref_bound, lti.left_target_ref_bound])
                        if start == knock_knock.outcome.NAN_INT:
                            start = 0
                        if end == knock_knock.outcome.NAN_INT:
                            end = len(exp.target_info.target_sequence)
                    
                        read_id = (exp.group, exp.sample_name, outcome.query_name)
                        ranges.append((read_id, start, end))
                                            
        return cls(exp.target_info, exp.target_info.target, ranges, total_reads, exps)

    @classmethod
    def donor_junctions(cls, exps, require_SNV=True):
        ranges = []
        total_reads = 0

        for exp in exps:
            sequencing_direction = exp.target_info.sequencing_direction
            sgRNA_strand = exp.target_info.sgRNA_feature.strand

            for outcome in exp.outcome_iter():
                total_reads += 1

                if outcome.category == 'unintended annealing of RT\'ed sequence' \
                    and 'with deletion' not in outcome.subcategory:

                    if require_SNV and outcome.subcategory == 'no scaffold, no SNV':
                        continue

                    lti = repair_seq.pooled_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)

                    if sequencing_direction == '-' and sgRNA_strand == '+':
                        donor_end = lti.left_insertion_ref_bound
                        target_start = lti.left_target_ref_bound

                    elif sequencing_direction == '-' and sgRNA_strand == '-':
                        donor_end = lti.right_insertion_ref_bound
                        target_start = lti.right_target_ref_bound

                    elif sequencing_direction == '+' and sgRNA_strand == '+':
                        donor_end = lti.right_insertion_ref_bound
                        target_start = lti.right_target_ref_bound

                    elif sequencing_direction == '+' and sgRNA_strand == '-':
                        donor_end = lti.left_insertion_ref_bound
                        target_start = lti.left_target_ref_bound

                    else:
                        raise ValueError

                    if target_start == knock_knock.outcome.NAN_INT or donor_end == knock_knock.outcome.NAN_INT:
                        continue
                    else:
                        read_id = (exp.group, exp.sample_name, outcome.query_name)
                        ranges.append((read_id, target_start, donor_end))

        return cls(exp.target_info, {'start': exp.target_info.target, 'end': exp.target_info.donor}, ranges, total_reads, exps)

    @classmethod
    def DSB_donor_junctions(cls, exps, subcategory):
        ranges = []
        total_reads = 0

        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1

                if outcome.category == 'donor misintegration' and outcome.subcategory == subcategory:
                    lti = repair_seq.pooled_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)

                    if lti.left_gap == 0:
                        if 'left unintended' in subcategory:
                            target_start = lti.left_target_ref_bound
                            donor_end = lti.left_insertion_ref_bound
                        else:
                            target_start = lti.right_target_ref_bound
                            donor_end = lti.right_insertion_ref_bound

                        if target_start == knock_knock.outcome.NAN_INT or donor_end == knock_knock.outcome.NAN_INT:
                            continue
                        else:
                            read_id = (exp.group, exp.sample_name, outcome.query_name)
                            ranges.append((read_id, target_start, donor_end))

        return cls(exp.target_info, {'start': exp.target_info.target, 'end': exp.target_info.donor}, ranges, total_reads, exps)

    @classmethod
    def integration_junctions(cls, exps, perfect_side=3):
        ranges = []
        total_reads = 0

        if perfect_side == 3:
            relevant_cats = [
                ('incomplete HDR', '5\' imperfect, 3\' HDR'),
                #('blunt misintegration', '5\' blunt, 3\' HDR'), 
            ]

            start = 'target_edge_before'
            end = 'donor_start'
        else:
            relevant_cats = [
                ('incomplete HDR', '5\' HDR, 3\' imperfect'),
                #('blunt misintegration', '5\' HDR, 3\' blunt'), 
            ]

            start = 'target_edge_after'
            end = 'donor_end'

        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1

                if (outcome.category, outcome.subcategory) in relevant_cats:
                    if outcome.details == 'n/a':
                        continue

                    integration = knock_knock.outcome_record.Integration.from_string(outcome.details)

                    ranges.append(((exp.group, exp.sample_name, outcome.query_name), getattr(integration, start), getattr(integration, end)))

        return cls(exp.target_info, {'start': exp.target_info.target, 'end': exp.target_info.donor}, ranges, total_reads, exps)

    @classmethod
    def donor_fragments(cls, exps):
        ranges = []
        total_reads = 0

        for exp in exps:
            for outcome in exp.outcome_iter():
                total_reads += 1

                if outcome.category == 'donor fragment':
                    integration = knock_knock.outcome_record.Integration.from_string(outcome.details)
                    start, end = sorted([integration.donor_start, integration.donor_end])

                    read_id = (exp.group, exp.sample_name, outcome.query_name)
                    ranges.append((read_id,  start, end))

        return cls(exp.target_info, {'start': exp.target_info.donor, 'end': exp.target_info.donor}, ranges, total_reads, exps)

def plot_ranges(all_ranges,
                names=None,
                primary_name='negative_control',
                features_to_draw=None,
                landmark=0,
                x_lims=None,
                y_lims=None,
                num_examples=100,
                invert=False,
                panels=None,
                sort_kwargs=None,
               ):
    if names is None:
        names = sorted(all_ranges)

    if panels is None:
        panels = [
            'start',
            'end',
        ]

    if y_lims is None:
        y_lims = {}

    primary_ranges = all_ranges[primary_name]

    if primary_ranges.sequence_name['start'] != primary_ranges.sequence_name['end']:
        raise ValueError(primary_ranges.sequence_name)

    sequence_length = primary_ranges.sequence_length['start'] 
    sequence_name = primary_ranges.sequence_name['start'] 

    if sort_kwargs is None:
        sort_kwargs = dict(key=lambda d: (d[1] - d[0], d[0]), reverse=True)
    
    sampled = all_ranges[primary_name].downsampled_edge_pairs(num_examples)
    
    xs = np.arange(sequence_length) - landmark
    
    ordered = sorted(sampled, **sort_kwargs)
    ordered = np.array(ordered)
    ordered = ordered - landmark
    
    fig, panel_axs = plt.subplots(len(panels), 1, figsize=(12, 2 * len(panels)), squeeze=False, sharex=True)
    panel_axs = {name: ax for name, ax in zip(panels, panel_axs[:, 0])}
    bottom_ax = panel_axs[panels[-1]]

    ax_p = bottom_ax.get_position()

    pair_ax_height = ax_p.height * (len(ordered) / 25)
    pair_ax_y0 = ax_p.y0 - ax_p.height * 0.5 - pair_ax_height
    pair_ax = fig.add_axes((ax_p.x0, pair_ax_y0, ax_p.width, pair_ax_height), sharex=bottom_ax)

    pair_ax.set_ylabel(f'random sample of {num_examples} examples')

    for i, (start, end) in enumerate(ordered):
        pair_ax.plot([start - 0.5, end + 0.5], [i, i], linewidth=2, color='black')

    if all_ranges[primary_name].sequence_name == all_ranges[primary_name].target_info.target:
        for name, cut_after in all_ranges[primary_name].target_info.cut_afters.items():
            x = cut_after - landmark
            
            pair_ax.axvline(x, color='black', linestyle='--')

    pair_ax.set_ylim(-0.03 * len(ordered), len(ordered) * 1.03)

    pair_ax.set_yticks([])

    for spine in ['left', 'right', 'top']:
        pair_ax.spines[spine].set_visible(False)
        
    pair_ax.set_xlabel('position')
    
    for panel, ax in panel_axs.items():
        if panel == 'start':
            get_ys = lambda ranges: ranges.edge_counts['start'] / ranges.total_reads * 100
            y_label = 'start (percentage)'

        elif panel == 'end':
            get_ys = lambda ranges: ranges.edge_counts['end'] / ranges.total_reads * 100
            y_label = 'end (percentage)'

        elif panel == 'involved':
            get_ys = lambda ranges: ranges.positions_involved / ranges.total_reads * 100
            y_label = 'involved\n(percentage)'

        elif panel == 'relative involved':
            get_ys = lambda ranges: ranges.positions_involved / len(ranges.edge_pairs) * 100
            y_label = 'involved\n(percentage of all deletions)'

        elif isinstance(panel, tuple):
            if panel[0] == 'log2_fc':
                if len(panel) == 3:
                    edge, side = panel[1:]
                    def get_ys(ranges):
                        numerator = ranges.cumulative_counts[edge][side] / ranges.total_reads * 100
                        denominator = all_ranges[primary_name].cumulative_counts[edge][side] / all_ranges[primary_name].total_reads * 100
                        ys = np.log2(numerator / denominator)
                        return ys

                    y_label = f'log2 fold change, {edge}, {side}'

                else:
                    edge = panel[1]
                    def get_ys(ranges):
                        numerator = ranges.edge_counts[edge] / ranges.total_reads * 100
                        denominator = all_ranges[primary_name].edge_counts[edge] / all_ranges[primary_name].total_reads * 100
                        ys = np.log2(numerator / denominator)
                        return ys

                    y_label = f'log2 fold change, {edge}'

            elif panel[0] == 'diff':
                if len(panel) == 3:
                    edge, side = panel[1:]
                    def get_ys(ranges):
                        numerator = ranges.cumulative_counts[edge][side] / ranges.total_reads * 100
                        denominator = all_ranges[primary_name].cumulative_counts[edge][side] / all_ranges[primary_name].total_reads * 100
                        ys = numerator - denominator
                        return ys

                    y_label = f'diff from {primary_name}, {edge} {side}'

                else:
                    edge = panel[1]
                    def get_ys(ranges):
                        numerator = ranges.edge_counts[edge] / ranges.total_reads * 100
                        denominator = all_ranges[primary_name].edge_counts[edge] / all_ranges[primary_name].total_reads * 100
                        ys = numerator - denominator
                        return ys

                    y_label = f'diff from {primary_name}, {edge}'

            elif panel[0] == 'log10' and panel[1] == 'involved':
                get_ys = lambda ranges: np.log10(ranges.positions_involved / ranges.total_reads * 100)
                y_label = 'involved (percentage)'

            else:
                edge, side = panel
                get_ys = lambda ranges: ranges.cumulative_counts[edge][side] / ranges.total_reads * 100
                y_label = f'{edge} {side}'

        color_i = 0
        for name in names:
            ranges = all_ranges[name]
            if name == primary_name:
                color = 'black'
            else:
                color = f'C{color_i}'
                color_i += 1

            ax.plot(xs, get_ys(ranges), marker='o', markersize=2, color=color, label=name)

        ax.grid(axis='y', alpha=0.3)

        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
            
        ax.set_ylabel(y_label)

        panel_y_lim = y_lims.get(panel)
        if panel_y_lim is not None:
            ax.set_ylim(*panel_y_lim)

    panel_axs[panels[0]].legend()
    
    if x_lims is None:
        x_lims = (0, sequence_length)

    ax.set_xlim(*x_lims)

    for panel_name, panel_ax in panel_axs.items():
        draw_features(panel_ax,
                      primary_ranges.target_info,
                      sequence_name,
                      x_lims[0], x_lims[1],
                      draw_labels=(panel_name == panels[0]),
                      features_to_draw=features_to_draw,
                     )

    if invert:
        ax.invert_xaxis()
    
    return fig

def plot_joint(all_ranges,
               nt_name='negative_control',
               other_name=None,
               target_min_p=None,
               target_max_p=None,
               donor_min_p=0,
               donor_max_p=None,
               v_max=None,
               width=12,
               invert_x=False,
               invert_y=False,
               title='',
               end_label='position in donor where integration ends',
               start_label='position in target where outcome resumes',
               label_offsets=None,
              ):
    ranges = all_ranges[nt_name]
    counts = ranges.joint_percentages

    total_percentage = all_ranges[nt_name].total_percentage

    ti = ranges.target_info

    if other_name is not None:
        other_ranges = all_ranges[other_name]
        other_counts = other_ranges.joint_percentages
        total_other_percentage = all_ranges[other_name].total_percentage

    if donor_max_p is None:
        donor_max_p = ranges.sequence_length['end']

    if target_min_p is None:
        target_min_p = ti.cut_after - 100

    if target_max_p is None:
        target_max_p = ti.cut_after + 100

    matrix_slice = idx[target_min_p:target_max_p + 1, donor_min_p:donor_max_p + 1]

    # bottom/top of extent is confusing
    extent = (donor_min_p - 0.5, donor_max_p + 0.5, target_max_p + 0.5, target_min_p - 0.5)

    height = width * (target_max_p - target_min_p + 1) / (donor_max_p - donor_min_p + 1)

    fig, joint_ax = plt.subplots(figsize=(width, height))

    if other_name is None:
        if v_max is None:
            v_max = counts.max() * 1.01
        cmap = hits.visualize.reds
        #cmap = copy.copy(plt.get_cmap('Oranges'))
        #cmap.set_over('black')
        im = joint_ax.imshow(counts[matrix_slice], extent=extent, cmap=cmap, vmax=v_max, interpolation='none')

    else:
        cmap = copy.copy(plt.get_cmap('bwr'))
        cmap.set_over('black')
        diffs = other_counts - counts

        if v_max is None:
            v_max = np.abs(diffs).max() * 1.01

        im = joint_ax.imshow((other_counts - counts)[matrix_slice], extent=extent, cmap=cmap, vmin=-v_max, vmax=v_max, interpolation='none')

    joint_ax.set_ylim(target_max_p + 0.5, target_min_p - 0.5)
    joint_ax.set_xlim(donor_min_p - 0.5, donor_max_p + 0.5)

    ax_p = joint_ax.get_position()

    ax_donor_height = ax_p.height / height * 2
    gap = ax_donor_height * 0.25
    ax_donor = fig.add_axes([ax_p.x0, ax_p.y1 + gap, ax_p.width, ax_donor_height], sharex=joint_ax)

    plot_marginal(all_ranges, nt_name, ax_donor, 'vertical')

    if other_name is not None:
        plot_marginal(all_ranges, other_name, ax_donor, 'vertical', color='C0')

    ax_target_width = ax_p.width / width * 2
    gap = ax_target_width * 0.25
    ax_target = fig.add_axes([ax_p.x1 + gap, ax_p.y0, ax_target_width, ax_p.height], sharey=joint_ax)

    plot_marginal(all_ranges, nt_name, ax_target, 'right')

    if other_name is not None:
        plot_marginal(all_ranges, other_name, ax_target, 'right', color='C0')

    #ax_target.set_xticks([])
    ax_target.xaxis.tick_top()
    ax_target.set_xlabel('percentage of reads')
    ax_target.xaxis.set_label_position('top') 
    plt.setp(ax_target.get_yticklabels(), visible=False)

    #ax_donor.set_yticks([])
    ax_donor.set_ylabel('percentage of reads')
    plt.setp(ax_donor.get_xticklabels(), visible=False)

    draw_features(ax_target, ti, ranges.sequence_name['start'], target_min_p, target_max_p,
                  orientation='right',
                  label_offsets=label_offsets,
                 )
    draw_features(ax_donor, ti, ranges.sequence_name['end'], donor_min_p, donor_max_p,
                  orientation='vertical',
                  label_offsets=label_offsets,
                 )
                
    for ax in [ax_donor, ax_target]:
        plt.setp(ax.spines.values(), visible=False)
        
    ax_donor.spines['bottom'].set_visible(True)
    ax_target.spines['left'].set_visible(True)
                
    for _, p in ti.cut_afters.items():
        lines = []

        if ranges.sequence_name['start'] == ti.target:
            lines.append(joint_ax.axhline)

        if ranges.sequence_name['end'] == ti.target:
            lines.append(joint_ax.axvline)

        for line in lines:
            line(p + 0.5, color='black', linestyle='--', alpha=0.5)
                
    plt.setp(joint_ax.spines.values(), alpha=0.5)

    if invert_x:
        joint_ax.invert_xaxis()

    if invert_y:
        joint_ax.invert_yaxis()

    joint_ax.annotate(end_label,
                      xy=(0.5, 1),
                      xycoords='axes fraction',
                      xytext=(0, 5),
                      textcoords='offset points',
                      ha='center',
                      va='bottom',
                      size=14,
                     )

    joint_ax.annotate(start_label,
                      xy=(1, 0.5),
                      xycoords='axes fraction',
                      xytext=(5, 0),
                      textcoords='offset points',
                      ha='left',
                      va='center',
                      rotation=-90,
                      size=14,
                     )

    cax = fig.add_axes([ax_p.x0 - ax_p.width * 0.2, ax_p.y0 + ax_p.height * (0.5 - 0.1), ax_p.width * 0.03, ax_p.height * 0.2])
    plt.colorbar(mappable=im, cax=cax)

    cax.yaxis.tick_left()

    if other_name is None:
        if title is None:
            title = nt_name
        colorbar_title = 'percentage of reads'
    else:
        if title is None:
            title = f'{other_name} vs. {nt_name}'
        colorbar_title = 'change in\npercentage of reads'

    ax_donor.set_title(title, y=1.2)
    cax.set_title(colorbar_title, y=1.1)

    ax_donor.annotate(f'{str(nt_name):>8}: {total_percentage:2.2f}%',
                      xy=(1, 0.9),
                      xycoords='axes fraction',
                      xytext=(2, 0),
                      textcoords='offset points',
                      color='black',
                      size=14,
                      family='monospace',
                     )

    if other_name is not None:
        ax_donor.annotate(f'{str(other_name):>8}: {total_other_percentage:2.2f}%',
                          xy=(1, 0.9),
                          xycoords='axes fraction',
                          xytext=(2, -16),
                          textcoords='offset points',
                          color='C0',
                          size=14,
                          family='monospace',
                         )

    return fig

def plot_marginal(all_ranges, name, ax, orientation, color='black'):
    if orientation == 'vertical':
        ys = all_ranges[name].edge_percentages['end']
        xs = np.arange(len(ys))
    else:
        xs = all_ranges[name].edge_percentages['start']
        ys = np.arange(len(xs))

    ax.plot(xs, ys, 'o-', markersize=1, color=color, label=name)

def draw_features(ax, ti, seq_name, min_p, max_p,
                  orientation='vertical',
                  label_offsets=None,
                  alpha=0.5,
                  draw_labels=True,
                  label_PAMs=False,
                  features_to_draw=None,
                  relative_to_cut_after=False,
                  label_overrides=None,
                  above=False,
                 ):
    if label_offsets is None:
        label_offsets = {}

    if label_overrides is None:
        label_overrides = {}

    if features_to_draw is None:
        features_to_draw = ti.features_to_show

    if relative_to_cut_after:
        x_offset = ti.cut_after + 0.5
    else:
        x_offset = 0

    features = [(s_name, f_name) for s_name, f_name in features_to_draw if s_name == seq_name]

    if orientation == 'right':
        span = ax.axhspan
        line = ax.axhline
    else:
        span = ax.axvspan
        line = ax.axvline

    for s_name, f_name in features:
        if (s_name, f_name) not in ti.annotated_and_inferred_features:
            continue
        feature = ti.annotated_and_inferred_features[s_name, f_name]

        if min_p <= feature.start <= max_p or min_p <= feature.end <= max_p:
            color = feature.attribute['color']

            if above:
                args = (0.95, 1)
            else:
                args = (0, 1)
            span(feature.start - 0.5 - x_offset, feature.end + 0.5 - x_offset, *args, facecolor=color, alpha=alpha, edgecolor='none', clip_on=False)

            if draw_labels:
                label = feature.attribute.get('short_name', f_name)

                label = label_overrides.get(label, label)

                if label is None:
                    continue

                if 'PAM' in label and not label_PAMs:
                    continue

                offset_points = 4
                if f_name in label_offsets:
                    extra_offsets = 12 * label_offsets[f_name]
                    offset_points += extra_offsets

                center = np.mean([feature.start, feature.end]) - x_offset
                if orientation == 'right':
                    annotate_kwargs = dict(
                        xy = (1, center),
                        xycoords = ('axes fraction', 'data'),
                        xytext = (offset_points, 0),
                        ha='left',
                        va='center',
                        rotation=-90,
                    )
                else:
                    annotate_kwargs = dict(
                        xy = (center, 1),
                        xycoords = ('data', 'axes fraction'),
                        xytext = (0, offset_points),
                        ha='center',
                        va='bottom',
                    )
                    
                ax.annotate(label,
                            textcoords='offset points',
                            color=color,
                            weight='bold',
                            **annotate_kwargs,
                           )

    if seq_name == ti.target:
        for _, p in ti.cut_afters.items():
            line(p + 0.5 - x_offset, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

def load_pool_ranges(pool, keys, RangesClass, progress=None, **kwargs):
    ''' keys: a list of genes or guides
        RangesClass
    '''
    ranges = {}

    for key in keys:
        if key in pool.variable_guide_library.guides:
            guides = [key]
        else:
            guides = pool.variable_guide_library.gene_guides(key, only_best_promoter=True)

        exps = [pool.single_guide_experiment('none', guide) for guide in guides]

        if progress is not None:
            exps = progress(exps)

        ranges[key] = RangesClass(exps, **kwargs)

    return ranges