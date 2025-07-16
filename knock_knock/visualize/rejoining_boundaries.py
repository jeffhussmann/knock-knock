from collections import defaultdict, Counter
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hits.visualize
import hits.utilities

import knock_knock.outcome
import knock_knock.pegRNAs
import knock_knock.editing_strategy

memoized_property = hits.utilities.memoized_property

class BoundaryProperties:
    def __init__(self, specific_subcategories=None):
        ''' edge_distributions: dict of Counters, outer dict keyed by (side, description), Counter keyed by position
            joint_distribution: Counter keyed by (subcategory, left position, right position)
            MH_nts_distribution: Counter keyed by number of nts of perfect microhomology at junction
            total_outcomes: number of eligibile outcomes counted to get these counts
        '''
        self.edge_distributions = defaultdict(Counter)
        self.joint_distribution = Counter()
        self.MH_nts_distribution = Counter()
        self.total_outcomes = 0
        self.specific_subcategories = specific_subcategories

    def make_joint_array(self):
        counts = Counter({(left, right): c for (key, left, right), c in self.joint_distribution.most_common()
                         if key == "left RT'ed, right RT'ed"
                        })

        self.joint_array = hits.utilities.counts_to_array(counts, dim=2) / self.total_outcomes

    def count_single_flap_boundaries(self, exp, include_intended_edit=False, max_reads=None):
        strat = exp.editing_strategy

        if include_intended_edit:
            pegRNA_name = strat.pegRNA.name
            pegRNA_HA_RT = strat.features[pegRNA_name, f'HA_RT_{pegRNA_name}']
            pegRNA_PBS = strat.features[pegRNA_name, 'PBS']
            last_HA_RT_nt_in_pegRNA = pegRNA_PBS.end - pegRNA_HA_RT.start

            target_PBS_name = strat.PBS_names_by_side_of_read[strat.pegRNA_side]
            target_PBS = strat.features[strat.target, target_PBS_name]
            target_HA_RT = strat.features[strat.target, f'HA_RT_{pegRNA_name}']

            # By definition, the nt on the PAM-distal side of the nick
            # is zero in the coordinate system, and postive values go towards
            # the PAM.

            if target_PBS.strand == '+':
                first_nt_after_HA_RT_in_genome = target_HA_RT.end + 1 - (target_PBS.end + 1)
            else:
                # TODO: confirm that there are no off-by-one errors here.
                first_nt_after_HA_RT_in_genome = (target_PBS.start - 1) - (target_HA_RT.start - 1)

        outcomes = exp.outcome_iter()
        if max_reads is not None:
            outcomes = islice(outcomes, max_reads)

        for outcome in outcomes:
            if outcome.category != 'nonspecific amplification':
                self.total_outcomes += 1

            if outcome.category.startswith('unintended rejoining') or outcome.category == 'RTed sequence':
                if self.specific_subcategories is not None:
                    if outcome.subcategory not in self.specific_subcategories:
                        continue

                ur_outcome = knock_knock.prime_editing_layout.UnintendedRejoiningOutcome.from_string(outcome.details)

                self.MH_nts_distribution[ur_outcome.MH_nts] += 1

                self.edge_distributions['pegRNA', 'RT\'ed'][ur_outcome.edges[strat.pegRNA_side]] += 1
                if ur_outcome.edges[strat.non_pegRNA_side] is not None:
                    self.edge_distributions['target', 'not RT\'ed'][ur_outcome.edges[strat.non_pegRNA_side] + ur_outcome.MH_nts] += 1

                    joint_key = (outcome.subcategory, ur_outcome.edges['left'], ur_outcome.edges['right'])
                    self.joint_distribution[joint_key] += 1

            elif include_intended_edit and outcome.category == 'intended edit':
                self.edge_distributions['pegRNA', 'RT\'ed'][last_HA_RT_nt_in_pegRNA] += 1
                self.edge_distributions['target', 'not RT\'ed'][first_nt_after_HA_RT_in_genome] += 1

        return self

    def count_dual_flap_boundaries(self, exp):
        strat = exp.editing_strategy

        for outcome in exp.outcome_iter():
            if outcome.category != 'nonspecific amplification':
                self.total_outcomes += 1

            if outcome.category.startswith('unintended rejoining'):
                if self.specific_subcategories is not None:
                    if outcome.subcategory not in self.specific_subcategories:
                        continue

                ur_outcome = knock_knock.prime_editing_layout.UnintendedRejoiningOutcome.from_string(outcome.details)

                for side_description in outcome.subcategory.split(', '):
                    side, description = side_description.split(' ', 1)
                    self.edge_distributions[side, description][ur_outcome.edges[side]] += 1

                self.MH_nts_distribution[ur_outcome.MH_nts] += 1

                joint_key = (outcome.subcategory, ur_outcome.edges['left'], ur_outcome.edges['right'])
                self.joint_distribution[joint_key] += 1

        return self

    def get_xs_and_ys(self, side, subcategory_key, cumulative=False, normalize=False, from_right=False):
        counts = self.edge_distributions[side, subcategory_key]

        if len(counts) == 0:
            return None, None

        xs = np.arange(min(counts), max(counts) + 1)
        ys = np.array([counts[x] for x in xs]) / max(self.total_outcomes, 1) * 100

        if normalize:
            denominator = ys.sum()
            if denominator == 0:
                denominator = 1
            ys = ys / denominator * 100

        if cumulative:
            if from_right:
                ys = np.cumsum(ys[::-1])[::-1]
            else:
                ys = np.cumsum(ys)

        return xs, ys

class EfficientBoundaryProperties:
    def __init__(self, editing_strategy, counts, aggregate_conditions=None, include_intended_edit=False):
        if 'nonspecific amplification' in counts.index:
            counts = counts.drop('nonspecific amplification')
        
        self.editing_strategy = editing_strategy

        if isinstance(counts, pd.Series):
            counts = counts.to_frame()

        self.counts = counts

        self.include_intended_edit = include_intended_edit

        if aggregate_conditions is not None:
            self.counts = self.counts.T.groupby(aggregate_conditions).sum().T

        cats = [
            "unintended rejoining of RT'ed sequence",
            "RTed sequence",
        ]

        if include_intended_edit:
            cats.append('intended edit')

        self.rejoining_counts = self.counts.loc[[c for c in cats if c in counts.index]]

    @memoized_property
    def pegRNA_coords(self):
        if self.include_intended_edit:
            pegRNA_name = self.editing_strategy.pegRNA.name
            pegRNA_HA_RT = self.editing_strategy.features[pegRNA_name, f'HA_RT_{pegRNA_name}']
            pegRNA_PBS = self.editing_strategy.features[pegRNA_name, 'PBS']
            last_HA_RT_nt_in_pegRNA = pegRNA_PBS.end - pegRNA_HA_RT.start

        def csd_to_pegRNA_coords(csd):
            c, s, d = csd

            details = knock_knock.outcome.Details.from_string(d)

            if c == 'intended edit':
                pegRNA_coord = last_HA_RT_nt_in_pegRNA
            elif c == 'RTed sequence':
                pegRNA_coord = details.pegRNA_edge
            else:
                pegRNA_coord = details[f'{self.editing_strategy.pegRNA_side}_rejoining_edge']

            return pegRNA_coord

        return self.rejoining_counts.groupby(by=csd_to_pegRNA_coords).sum()

    @memoized_property
    def target_coords(self):
        if self.include_intended_edit:
            target_PBS_name = self.editing_strategy.PBS_names_by_side_of_read[self.editing_strategy.pegRNA_side]
            target_PBS = self.editing_strategy.features[self.editing_strategy.target, target_PBS_name]
            target_HA_RT = self.editing_strategy.features[self.editing_strategy.target, f'HA_RT_{self.editing_strategy.pegRNA.name}']

            # By definition, the nt on the PAM-distal side of the nick
            # is zero in the coordinate system, and postive values go towards
            # the PAM.

            if target_PBS.strand == '+':
                first_nt_after_HA_RT_in_genome = target_HA_RT.end + 1 - (target_PBS.end + 1)
            else:
                # TODO: confirm that there are no off-by-one errors here.
                first_nt_after_HA_RT_in_genome = (target_PBS.start - 1) - (target_HA_RT.start - 1)

        def csd_to_target_coords(csd):
            c, s, d = csd

            details = knock_knock.outcome.Details.from_string(d)

            if c == 'intended edit':
                target_coord = first_nt_after_HA_RT_in_genome
            elif c == 'RTed sequence':
                target_coord = 0
            else:
                target_coord = details[f'{self.editing_strategy.non_pegRNA_side}_rejoining_edge']

            return target_coord

        return self.rejoining_counts.groupby(by=csd_to_target_coords).sum()

    def to_exp_sets(self, columns_to_extract=None):
        if columns_to_extract is None:
            columns_to_extract = [
                ('all', pd.IndexSlice[:], 'black'),
            ]

        exp_sets = {}

        for name, columns, color in columns_to_extract:
            bps = BoundaryProperties()
            bps.total_outcomes = self.counts[columns].sum().sum()

            bps.edge_distributions['pegRNA', 'RT\'ed'].update(self.pegRNA_coords[columns].sum(axis=1).to_dict())
            bps.edge_distributions['target', 'not RT\'ed'].update(self.target_coords[columns].sum(axis=1).to_dict())

            exp_sets[name] = {
                'color': color,
                'results': bps,
            }

        return exp_sets

class EfficientDualFlapBoundaryProperties:
    def __init__(self, editing_strategy, counts, aggregate_conditions=None):
        if 'nonspecific amplification' in counts.index:
            counts = counts.drop('nonspecific amplification')
        
        self.editing_strategy = editing_strategy
        self.counts = counts

        if aggregate_conditions is not None:
            self.counts = self.counts.T.groupby(aggregate_conditions).sum().T

        left_cats = [
            ("unintended rejoining of RT'ed sequence", "left RT'ed, right RT'ed"),
            ("unintended rejoining of RT'ed sequence", "left RT'ed, right not RT'ed"),
        ]

        self.left_rejoining_counts = self.counts.loc[[(c, s, d) for c, s, d in self.counts.index if (c, s) in left_cats]]

        right_cats = [
            ("unintended rejoining of RT'ed sequence", "left RT'ed, right RT'ed"),
            ("unintended rejoining of RT'ed sequence", "left not RT'ed, right RT'ed"),
        ]

        self.right_rejoining_counts = self.counts.loc[[(c, s, d) for c, s, d in self.counts.index if (c, s) in right_cats]]

    @memoized_property
    def left_pegRNA_coords(self):

        def csd_to_left_coords(csd):
            c, s, d = csd

            details = knock_knock.outcome.Details.from_string(d)
            pegRNA_coord = details.left_rejoining_edge

            return pegRNA_coord

        return self.left_rejoining_counts.groupby(by=csd_to_left_coords).sum()

    @memoized_property
    def right_pegRNA_coords(self):

        def csd_to_right_coords(csd):
            c, s, d = csd

            details = knock_knock.outcome.Details.from_string(d)
            pegRNA_coord = details.right_rejoining_edge

            return pegRNA_coord

        return self.right_rejoining_counts.groupby(by=csd_to_right_coords).sum()

    def to_exp_sets(self, columns_to_extract):
        exp_sets = {}

        for name, columns, color in columns_to_extract:
            bps = BoundaryProperties()
            bps.total_outcomes = self.counts[columns].sum().sum()

            bps.edge_distributions['left', 'RT\'ed'].update(self.left_pegRNA_coords[columns].sum(axis=1).to_dict())
            bps.edge_distributions['right', 'RT\'ed'].update(self.right_pegRNA_coords[columns].sum(axis=1).to_dict())

            exp_sets[name] = {
                'color': color,
                'results': bps,
            }

        return exp_sets

def plot_single_flap_extension_chain_edges(editing_strategy,
                                           exp_sets,
                                           normalize=False,
                                           pegRNA_x_lims=None,
                                           target_x_lims=None,
                                           marker_size=2,
                                           line_width=1,
                                           pegRNA_from_right=True,
                                           draw_sequence=False,
                                           include_genome=True,
                                           side_and_subcategories=None,
                                           annotate_structure=False,
                                           draw_genomic_homology=False,
                                           manual_title=None,
                                           manual_title_color=None,
                                          ):
    strat = editing_strategy

    if side_and_subcategories is None:
        side_and_subcategories = [
            ('pegRNA', 'RT\'ed'),
            ('target', 'not RT\'ed'),
        ]

    # Common parameters.
    ref_bar_height = 0.05
    feature_height = 0.04

    y_start = -0.35

    figsize = (16, 6)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    if len(strat.pegRNAs) != 1:
        return fig, axs

    for ax_col, (side, subcategory_key) in zip(axs.T, side_and_subcategories):
        for ax, cumulative in zip(ax_col, [True, False]):
            for set_name, set_details in exp_sets.items():
                color = set_details['color']

                xs, ys = set_details['results'].get_xs_and_ys(side,
                                                              subcategory_key,
                                                              cumulative=cumulative,
                                                              normalize=normalize,
                                                              from_right=pegRNA_from_right and (side == 'pegRNA' or subcategory_key == 'RT\'ed'),
                                                             )

                if xs is None:
                    continue

                ax.plot(xs,
                        ys,
                        'o-',
                        label=set_name,
                        linewidth=line_width,
                        markersize=marker_size,
                        color=color,
                       )

        ax = ax_col[1]

        if subcategory_key == 'RT\'ed':

            if len(strat.pegRNAs) > 1:
                pegRNA_name = strat.pegRNA_names_by_side_of_read[side]
            else:
                pegRNA_name = strat.pegRNA.name

            # By definition, the end of the PBS on this side's pegRNA 
            # is zero in the coordinate system.
            PBS_end = strat.features[pegRNA_name, 'PBS'].end

            pegRNA_length = len(strat.reference_sequences[pegRNA_name])

            start = PBS_end - pegRNA_length - 0.5
            end = PBS_end + 0.5

            if pegRNA_x_lims is not None:
                pegRNA_x_min, pegRNA_x_max = pegRNA_x_lims
            else:
                pegRNA_x_min, pegRNA_x_max = start - 5, end + 5

            start = max(start, pegRNA_x_min)
            end = min(end, pegRNA_x_max)

            ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                       facecolor=strat.pegRNA_name_to_color[pegRNA_name],
                       clip_on=False,
                      )

            if draw_sequence:
                pegRNA_sequence = strat.reference_sequences[pegRNA_name]
                for x in range(int(np.ceil(start)), int(np.ceil(end))):
                    p = PBS_end - x
                    if 0 <= p < len(pegRNA_sequence):
                        ax.annotate(pegRNA_sequence[p],
                                    xy=(x, y_start + ref_bar_height * 0.5),
                                    xycoords=('data', 'axes fraction'),
                                    annotation_clip=False,
                                    size=7,
                                    family='monospace',
                                    va='center',
                                    ha='center',
                                   )

                if p > 0:
                    ax.annotate('...',
                                xy=(x + 1 - 0.5, y_start),
                                xycoords=('data', 'axes fraction'),
                                annotation_clip=False,
                                size=7,
                                family='monospace',
                               )

            features_to_annotate = [
                'protospacer',
                'scaffold',
                'PBS',
            ]

            feature_aliases = {}

            if strat.pegRNA_programmed_insertion_features:
                for insertion_feature in strat.pegRNA_programmed_insertion_features:
                    features_to_annotate.append(insertion_feature.ID)
                    feature_aliases[insertion_feature.ID] = 'insertion'

                features_to_annotate.append(f'HA_RT_{strat.pegRNA.name}')
                feature_aliases[f'HA_RT_{strat.pegRNA.name}'] = 'homology\narm'

            else:
                features_to_annotate.append('RTT')

            for feature_name in features_to_annotate:
                feature = strat.features[pegRNA_name, feature_name]
                color = feature.attribute['color']

                if feature_name == 'protospacer':
                    color = strat.pegRNA_name_to_color[pegRNA_name]

                # Moving back from the PBS end is moving
                # forward in the coordinate system.
                start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

                if end < pegRNA_x_min or start > pegRNA_x_max: 
                    continue

                start = max(start, pegRNA_x_min)
                end = min(end, pegRNA_x_max)

                ax.axvspan(start,
                           end,
                           y_start + ref_bar_height,
                           y_start + ref_bar_height + feature_height,
                           facecolor=color,
                           alpha=0.75,
                           clip_on=False,
                          )

                for data_ax in ax_col:
                    data_ax.axvspan(start,
                                    end,
                                    facecolor=color,
                                    alpha=0.5,
                                    clip_on=False,
                                )

                label = feature_aliases.get(feature_name, feature_name)

                if len(label) > 3 and end - start < 10:
                    y_offset = -15
                else:
                    y_offset = -5

                ax.annotate(label,
                            xy=(np.mean([start, end]), y_start),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, y_offset),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=color,
                            annotation_clip=False,
                            weight='bold',
                           )

            new_zero = len(strat.sgRNA_components[pegRNA_name]['PBS']) - 1
            new_ticks = [t for t in np.arange(-100, 300, 10) + new_zero if pegRNA_x_min <= t <= pegRNA_x_max]
            new_labels = [str(t - new_zero) for t in new_ticks]

            for ax in ax_col:
                ax.set_xlim(pegRNA_x_min, pegRNA_x_max)
                ax.set_ylim(0)
                ax.set_xticks(new_ticks)
                ax.set_xticklabels(new_labels)

            if manual_title is None:
                title = f'pegRNA ({pegRNA_name})'
            else:
                title = manual_title

            if manual_title_color is None:
                title_color = strat.pegRNA_name_to_color[pegRNA_name]
            else:
                title_color = manual_title_color

            ax_col[0].set_title(title, color=title_color)
            ax_col[0].set_xticklabels([])

            ax_col[1].set_xlabel('End of reverse transcribed sequence')

            if annotate_structure:
                flipped_total_bpps, flipped_propensity = strat.pegRNA.RTT_structure

                components = strat.sgRNA_components[pegRNA_name]
                xs = np.arange(len(components['PBS']), len(components['PBS'] + components['RTT']))

                bpps_ax = ax.twinx()
                bpps_ax.plot(xs,
                             flipped_total_bpps,
                             'o-',
                             markersize=2,
                             color='black',
                             clip_on=False,
                             alpha=0.75,
                            )
                bpps_ax.set_ylim(0, 1)

                bpps_ax.set_ylabel('Total probability paired', size=12, rotation=270, labelpad=12)

                for x, c in zip(xs, flipped_propensity):
                    ax.annotate(c,
                                xy=(x, 1),
                                xycoords=('data', 'axes fraction'),
                                xytext=(0, 2),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                family='monospace',
                               )
                  
                ax.annotate('RTT MFE:',
                            xy=(xs[0], 1),
                            xycoords=('data', 'axes fraction'),
                            xytext=(-5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom',
                            family='monospace',
                           )

        elif subcategory_key == 'not RT\'ed':
            ax = ax_col[1]

            colors = {}

            for pegRNA_name in strat.pegRNA_names:
                color = strat.pegRNA_name_to_color[pegRNA_name]
                light_color = hits.visualize.apply_alpha(color, 0.5)
                ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
                colors[ps_name] = light_color

                PAM_name = f'{ps_name}_PAM'
                colors[PAM_name] = color

            for primer_name in strat.primer_names:
                colors[primer_name] = 'grey'

            if len(strat.pegRNA_names) == 1:
                pegRNA_name = strat.pegRNA_names[0]
            else:
                pegRNA_name = strat.pegRNA_names_by_side_of_read[side]

            PBS = strat.features[strat.target, knock_knock.pegRNAs.PBS_name(pegRNA_name)]

            # By definition, the nt on the PAM-distal side of the nick
            # is zero in the coordinate system, and postive values go towards
            # the PAM.

            # 24.12.05: having trouble reconciling comment above with code. 
            # Should it be "PAM-proximal side of the nick"?

            feature_names = strat.protospacer_names + list(strat.PAM_features) + strat.primer_names

            if draw_genomic_homology:
                feature_names.append(f'HA_RT_{pegRNA_name}')
                feature_names.append(f'HA_PBS_{pegRNA_name}')

            def target_to_nick_coords(target_x):
                if PBS.strand == '+':
                    return target_x - (PBS.end + 1)
                else:
                    return (PBS.start - 1) - target_x

            def target_bounds_to_xs(start, end):
                left, right = sorted([target_to_nick_coords(start), target_to_nick_coords(end)])
                return left - 0.5, right + 0.5

            if target_x_lims is not None:
                target_x_min, target_x_max = target_x_lims
            else:
                target_x_min, target_x_max = target_bounds_to_xs(strat.amplicon_interval.start, strat.amplicon_interval.end)

            for feature_name in feature_names:
                if (strat.target, feature_name) not in strat.features:
                    # 3b spacer
                    continue

                feature = strat.features[strat.target, feature_name]
                
                start, end = target_bounds_to_xs(feature.start, feature.end)

                if end < target_x_min or start > target_x_max: 
                    continue

                if 'HA_PBS' in feature_name or 'HA_RT' in feature_name:
                    height = feature_height * 0.75
                else:
                    height = feature_height * 1.5
                
                color = colors.get(feature_name, feature.attribute['color'])

                ax.axvspan(start, end,
                           y_start, y_start - height,
                           facecolor=color,
                           clip_on=False,
                          )

                if 'PAM' in feature_name:
                    label = None
                elif feature_name.endswith('protospacer'):
                    label = feature_name[:-len('_protospacer')]
                elif 'HA_RT' in feature_name or 'HA_PBS' in feature_name:
                    feature_name = ''
                else:
                    label = feature_name

                ax.annotate(label,
                            xy=(np.mean([start, end]), y_start),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, -15),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            color=color,
                            annotation_clip=False,
                            weight='bold',
                        )

            ax.axvspan(target_x_min, target_x_max, y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

            for cut_after_name, cut_after in strat.cut_afters.items():
                # Potentially confusing - if PBS is on the plus strand, cut_after is -1 in nick coords,
                # but if PBS is on the minus strand, cut_after is 0 in nick coords. In either case,
                # just take actual cut position (cut_after + 0.5) in target coords and convert it.
                x = target_to_nick_coords(cut_after + 0.5)

                if target_x_min <= x <= target_x_max:

                    ax.axvline(x, color='black', alpha=0.75)

                    name, strand = cut_after_name.rsplit('_', 1)
                    ref_y = y_start + 0.5 * ref_bar_height
                    cut_y_bottom = ref_y - feature_height
                    cut_y_middle = ref_y
                    cut_y_top = ref_y + feature_height

                    if (strand == '+' and strat.sequencing_direction == '+') or (strand == '-' and strat.sequencing_direction == '-'):
                        ys = [cut_y_middle, cut_y_top]
                    elif (strand == '-' and strat.sequencing_direction == '+') or (strand == '+' and strat.sequencing_direction == '-'):
                        ys = [cut_y_bottom, cut_y_middle]
                    else:
                        ys = [cut_y_bottom, cut_y_top]

                    ax.plot([x, x],
                            ys,
                            '-',
                            linewidth=1,
                            color='black',
                            solid_capstyle='butt',
                            zorder=10,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                           )

            for ax in ax_col:
                ax.set_xlim(target_x_min, target_x_max)
                ax.set_ylim(0)

            ax_col[0].set_title('genome', color='C0')
            ax_col[0].set_xticklabels([])

            ax_col[1].set_xlabel('Rejoining position in genome')
        
        #if side == 'right':
        #    for ax in ax_col:
        #        ax.invert_xaxis()

    if len(exp_sets) > 1:
        if include_genome:
            ax = axs[0, 1]
        else:
            ax = axs[0, 0]

        ax.legend(bbox_to_anchor=(1, 1),
                  loc='upper left',
                  ncol=len(exp_sets) // 20 + (1 if len(exp_sets) % 20 != 0 else 0),
                  prop=dict(
                      family='monospace',
                  ),
                 )

    if normalize:
        ylabel = 'Normalized cumulative\npercentage of\nrelevant reads'
        y_lims = (0, 100.5)
    else:
        ylabel = 'Cumulative\npercentage of all reads'
        y_lims = (0,)

    axs[0, 0].set_ylabel(ylabel, size=12)
    axs[0, 0].set_ylim(*y_lims)

    if normalize:
        ylabel = 'Normalized percentage\nof relevant reads'
    else:
        ylabel = 'Percentage of all reads'

    axs[1, 0].set_ylabel(ylabel, size=12)

    if not include_genome:
        for ax in axs[:, 1]:
            fig.delaxes(ax)

    return fig, axs

def plot_dual_flap_extension_chain_edges(editing_strategy,
                                         exp_sets,
                                         cumulative=False,
                                         normalize=False,
                                         x_lims=(-100, 150),
                                        ):

    strat = editing_strategy

    # Common parameters.
    ref_bar_height = 0.02
    feature_height = 0.03
    gap_between_refs = 0.01

    if cumulative:
        marker_size = 2
    else:
        marker_size = 3

    figsize = (16, 3)

    figs = {}

    # not RT'ed and deletion
    for subcategory_key in [
        "not RT'ed",
    ]:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        figs[subcategory_key] = fig

        for ax, side in zip(axs, ['left', 'right']):
            for set_name, set_details in exp_sets.items():
                color = set_details['color']

                xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

                if xs is None:
                    continue

                ax.plot(xs, ys, 'o-', label=set_name, markersize=marker_size, color=color, alpha=0.5)

            pegRNA_name = strat.pegRNA_names_by_side_of_read[side]
            protospacer_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)

            PBS_name = strat.PBS_names_by_side_of_read[side]
            PBS = strat.features[strat.target, PBS_name]

            other_pegRNA_name = strat.pegRNA_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]
            other_PBS_name = strat.PBS_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]
            other_protospacer_name = knock_knock.pegRNAs.protospacer_name(other_pegRNA_name)
            PAM_name = f'{protospacer_name}_PAM'
            other_PAM_name = f'{other_protospacer_name}_PAM'

            colors = {
                protospacer_name: hits.visualize.apply_alpha(strat.pegRNA_name_to_color[pegRNA_name], 0.5),
                other_protospacer_name: hits.visualize.apply_alpha(strat.pegRNA_name_to_color[other_pegRNA_name], 0.5),
                PAM_name: strat.pegRNA_name_to_color[pegRNA_name],
                other_PAM_name: strat.pegRNA_name_to_color[other_pegRNA_name],
            }

            for primer_name in strat.primer_names:
                colors[primer_name] = 'lightgrey'

            # By definition, the nt on the PAM-distal side of the nick
            # is zero in the coordinate system, and postive values go towards
            # the PAM.

            y_start = -0.1

            feature_names = [
                protospacer_name,
                other_protospacer_name,
                PBS_name, other_PBS_name,
                PAM_name,
                other_PAM_name
            ] + strat.primer_names

            for feature_name in feature_names:
                feature = strat.features[strat.target, feature_name]
                
                # Moving towards the other nick is moving
                # forward in the coordinate system.
                if PBS.strand == '+':
                    start, end = feature.start - PBS.end - 0.5, feature.end - PBS.end + 0.5
                else:
                    start, end = PBS.start - feature.end - 0.5, PBS.start - feature.start + 0.5

                if 'PBS' in feature_name:
                    height = 0.015
                else:
                    height = 0.03

                if end < x_lims[0] or start > x_lims[1]: 
                    continue
                
                ax.axvspan(start, end,
                           y_start, y_start - height * 1.5,
                           facecolor=colors.get(feature_name, feature.attribute['color']),
                           clip_on=False,
                          )

            ax.axvspan(x_lims[0], x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

            for cut_after_name, cut_after in strat.cut_afters.items():
                if PBS.strand == '+':
                    x = cut_after - PBS.end
                else:
                    x = PBS.start - cut_after

                name, strand = cut_after_name.rsplit('_', 1)

                ref_y = y_start + 0.5 * ref_bar_height
                cut_y_bottom = ref_y - feature_height
                cut_y_middle = ref_y
                cut_y_top = ref_y + feature_height

                if (strand == '+' and strat.sequencing_direction == '+') or (strand == '-' and strat.sequencing_direction == '-'):
                    ys = [cut_y_middle, cut_y_top]
                elif (strand == '-' and strat.sequencing_direction == '+') or (strand == '+' and strat.sequencing_direction == '-'):
                    ys = [cut_y_bottom, cut_y_middle]
                else:
                    raise ValueError

                ax.plot([x, x],
                        ys,
                        '-',
                        linewidth=1,
                        color='black',
                        solid_capstyle='butt',
                        zorder=10,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False,
                       )

            ax.set_xlim(*x_lims)
            ax.set_ylim(0)

            ax.set_title(f'{side} {subcategory_key}')

            ax.set_xticklabels([])

            if side == 'right':
                ax.invert_xaxis()

        if len(exp_sets) > 1:
            axs[1].legend(bbox_to_anchor=(1, 1), loc='upper left')

        if cumulative:
            axs[0].set_ylabel('Cumulative percentage of reads', size=12)
        else:
            axs[0].set_ylabel('Percentage of reads', size=12)

    # just RT'ed
    subcategory_key = "RT\'ed"

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    figs[subcategory_key] = fig

    for ax, side in zip(axs, ['left', 'right']):
        for set_name, set_details in exp_sets.items():
            color = set_details['color']

            xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

            if xs is None:
                continue

            ax.plot(xs, ys, 'o-', label=set_name, markersize=marker_size, color=color, alpha=0.5)

        pegRNA_name = strat.pegRNA_names_by_side_of_read[side]

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = strat.features[pegRNA_name, 'PBS'].end

        y_start = -0.1

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = strat.features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                       facecolor=strat.pegRNA_name_to_color[pegRNA_name],
                       clip_on=False,
                      )
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                       facecolor=feature.attribute['color'],
                       alpha=0.75,
                       clip_on=False,
                      )

            if feature_name == 'overlap':
                for x in [start, end]:
                    ax.axvline(x, color=feature.attribute['color'])

        ax.set_xlim(*x_lims)
        ax.set_ylim(0)

        ax.set_title(f'{side} {subcategory_key}')

        ax.set_xticklabels([])
            
        if side == 'right':
            ax.invert_xaxis()

    if len(exp_sets) > 1:
        axs[1].legend(bbox_to_anchor=(1, 1), loc='upper left')

    if cumulative:
        axs[0].set_ylabel('Cumulative percentage of reads', size=12)
    else:
        axs[0].set_ylabel('Percentage of reads', size=12)

    # overlap-extended
    subcategory_key = "RT'ed + overlap-extended"

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    figs[subcategory_key] = fig

    for ax, side in zip(axs, ['left', 'right']):
        for set_name, set_details in exp_sets.items():
            color = set_details['color']

            xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

            if xs is None:
                continue

            ax.plot(xs, ys, '.-', label=set_name, color=color, alpha=0.5)

        pegRNA_name = strat.pegRNA_names_by_side_of_read[side]
        other_pegRNA_name = strat.pegRNA_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = strat.features[pegRNA_name, 'PBS'].end

        y_start = -0.2

        for feature_name in ['PBS', 'RTT', 'overlap']:
            feature = strat.features[pegRNA_name, feature_name]
            
            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=strat.pegRNA_name_to_color[pegRNA_name], clip_on=False)
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
            
        # The left side of the pegRNA overlap in the coordinate system is the 
        # end of the overlap feature on this side's pegRNA.
        overlap_start = PBS_end - strat.features[pegRNA_name, 'overlap'].end

        other_overlap = strat.features[other_pegRNA_name, 'overlap']

        overlap_start_offset = overlap_start - other_overlap.start

        y_start = y_start + ref_bar_height + feature_height + gap_between_refs

        for feature_name in ['PBS', 'RTT', 'overlap']:
            feature = strat.features[other_pegRNA_name, feature_name]
            
            start, end = overlap_start_offset + feature.start - 0.5, overlap_start_offset + feature.end + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=strat.pegRNA_name_to_color[other_pegRNA_name], clip_on=False)
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
            
        other_PBS_name = strat.PBS_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]
        other_protospacer_name = knock_knock.pegRNAs.protospacer_name(other_pegRNA_name)
        other_PBS_target = strat.features[strat.target, other_PBS_name]
            
        other_PBS_start_offset = overlap_start_offset + strat.features[other_pegRNA_name, 'PBS'].start

        y_start = y_start + ref_bar_height + feature_height + gap_between_refs

        for feature_name in [other_protospacer_name,
                             other_PBS_name,
                             strat.primers_by_side_of_read[knock_knock.editing_strategy.other_side[side]].ID,
                            ]:
            feature = strat.features[strat.target, feature_name]
            
            if other_PBS_target.strand == '+':
                start, end = other_PBS_start_offset + (other_PBS_target.end - feature.end) - 0.5, other_PBS_start_offset + (other_PBS_target.end - feature.start) + 0.5
            else:
                start, end = other_PBS_start_offset + (feature.start - other_PBS_target.start) - 0.5, other_PBS_start_offset + (feature.end - other_PBS_target.start) + 0.5
                
            start = max(start, other_PBS_start_offset - 0.5)
            
            if feature_name == other_PBS_name:
                height = 0.015
            else:
                height = 0.03
            
            ax.axvspan(start, end,
                        y_start + ref_bar_height,
                        y_start + ref_bar_height + height,
                        facecolor=colors.get(feature_name, feature.attribute['color']),
                        clip_on=False,
                        )

        ax.axvspan(other_PBS_start_offset - 0.5, x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

        ax.set_xlim(*x_lims)
        ax.set_ylim(0)

        ax.set_title(f'{side} {subcategory_key}')

        ax.set_xticklabels([])
            
        if side == 'right':
            ax.invert_xaxis()

    if len(exp_sets) > 1:
        axs[0].legend()

    if cumulative:
        axs[0].set_ylabel('Cumulative percentage of reads', size=12)
    else:
        axs[0].set_ylabel('Percentage of reads', size=12)

    return figs

def plot_joint_RT_edges(editing_strategy, exp_sets, v_max=0.4):
    strat = editing_strategy

    for set_name in exp_sets:
        results = exp_sets[set_name]['results']
        results.make_joint_array()
        percentages_array = results.joint_array * 100
        
        max_marginal = max(percentages_array.sum(axis=0).max(), percentages_array.sum(axis=1).max())

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(percentages_array, cmap=hits.visualize.blues, vmin=0, vmax=v_max)
        fig.suptitle(set_name)
        
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        
        ax_p = ax.get_position()
        
        top_ax = fig.add_axes((ax_p.x0, ax_p.y1, ax_p.width, ax_p.height * 0.25), sharex=ax)
        
        xs = np.arange(percentages_array.shape[1])
        ys = percentages_array.sum(axis=0)
        top_ax.plot(xs, ys, 'o-', markersize=2)
        plt.setp(top_ax.get_xticklabels(), visible=False)
        top_ax.spines[['top', 'right']].set_visible(False)
        top_ax.tick_params(axis='x', length=0)
        
        top_ax.set_ylim(0, max_marginal * 1.1)
        
        right_ax = fig.add_axes((ax_p.x1, ax_p.y0, ax_p.width * 0.25, ax_p.height), sharey=ax)
        
        ys = np.arange(percentages_array.shape[0])
        xs = percentages_array.sum(axis=1)
        right_ax.plot(xs, ys, 'o-', markersize=2)
        plt.setp(right_ax.get_yticklabels(), visible=False)
        right_ax.spines[['top', 'right']].set_visible(False)
        right_ax.set_xlim(0, max_marginal * 1.1)
        right_ax.tick_params(axis='y', length=0)
        
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 90)
        
        ref_bar_height = 0.025
        feature_height = 0.025
        
        pegRNA_name = strat.pegRNA_names_by_side_of_read['right']

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = strat.features[pegRNA_name, 'PBS'].end

        y_start = -3 * (ref_bar_height + feature_height)

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = strat.features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

            ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                    facecolor=strat.pegRNA_name_to_color[pegRNA_name],
                    clip_on=False,
                    )
            
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                    facecolor=feature.attribute['color'],
                    alpha=0.75,
                    clip_on=False,
                    )
            
            if feature_name == 'overlap':
                ax.axvline(start, color=feature.attribute['color'])
                ax.axvline(end, color=feature.attribute['color'])
            
        pegRNA_name = strat.pegRNA_names_by_side_of_read['left']

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        
        PBS_end = strat.features[pegRNA_name, 'PBS'].end

        x_start = -2 * (ref_bar_height + feature_height)

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = strat.features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

            ax.axhspan(start, end, x_start, x_start - ref_bar_height, 
                       facecolor=strat.pegRNA_name_to_color[pegRNA_name],
                       clip_on=False,
                      )
            
            ax.axhspan(start, end, x_start - ref_bar_height, x_start - ref_bar_height - feature_height, 
                       facecolor=feature.attribute['color'],
                       alpha=0.75,
                       clip_on=False,
                      )

            if feature_name == 'overlap':
                ax.axhline(start, color=feature.attribute['color'])
                ax.axhline(end, color=feature.attribute['color'])
