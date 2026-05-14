from collections import Counter, OrderedDict

import bokeh.palettes

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd

import hits.utilities

import knock_knock.editing_strategy
import knock_knock.pegRNAs

top_color = '#FF00FF'
bottom_color = '#00FF00'

def round_1_condition_to_color(condition):
    palette = bokeh.palettes.Dark2_8

    if condition['time_point'] == 'T1 (4h post nucleofection)' and condition['cell_line'] == 'Parental RPE1':
        color = palette[0]
    elif condition['time_point'] == 'T2 (7h post nucleofection)' and condition['cell_line'] == 'Parental RPE1':
        color = palette[2]
    elif condition['time_point'] == 'T1 (4h post nucleofection)' and condition['cell_line'] == 'MLH1KO RPE1':
        color = palette[1]
    elif condition['time_point'] == 'T2 (7h post nucleofection)' and condition['cell_line'] == 'MLH1KO RPE1':
        color = palette[3]
    else:
        color = 'black'

    return color

def round_2_condition_to_color(condition):
    palette = bokeh.palettes.Dark2_8

    if condition['arrest_status'] == 'Nonarrested':
        color = palette[0]
    elif condition['arrest_status'] == 'Nonswapped':
        color = palette[2]
    elif condition['arrest_status'] == 'Swapped':
        color = palette[1]
    else:
        color = 'black'

    return color

def round_3_condition_to_color(condition):
    palette = bokeh.palettes.Category20b_20

    if condition['cell_line'] == 'MLH1ko':
        index = 3
    elif condition['cell_line'] == 'parental':
        index = 7
    else:
        raise ValueError

    if condition['time_point'] == '-PE':
        offset = 0
    elif condition['time_point'] == '05h':
        offset = -1
    elif condition['time_point'] == '10h':
        offset = -3
    else:
        raise ValueError

    color = palette[index + offset]

    return color

def condition_to_label(condition, condition_keys_to_label):
    label = ', '.join([f'{key}={value}' for key, value in condition.items() if key in condition_keys_to_label])
    return label

class StrandsGrid:
    def __init__(self,
                 base_dir,
                 pegRNA_name=None,
                 buffer=10,
                 y_max=100,
                ):

        target_name = 'FANCF'

        sgRNAs = [
            'oWY407',
            'oWY408',
            'oWY427',
        ]

        if pegRNA_name is None:
            self.anchor_sgRNA_name = 'pWY090'
        else:
            self.anchor_sgRNA_name = pegRNA_name

        sgRNAs.append(self.anchor_sgRNA_name)

        self.pegRNA_name = pegRNA_name

        self.editing_strategy = knock_knock.editing_strategy.EditingStrategy(base_dir, target_name, sgRNAs=sgRNAs)

        self.anchor = self.editing_strategy.cut_afters[f'{self.anchor_sgRNA_name}_protospacer_+']

        left_primer_name = 'oWY388'
        right_primer_name = 'oWY390'

        self.left_primer = self.editing_strategy.features[self.editing_strategy.target, left_primer_name]
        self.right_primer = self.editing_strategy.features[self.editing_strategy.target, right_primer_name]

        self.x_min = self.left_primer.start - buffer - self.anchor
        self.x_max = self.right_primer.end + buffer - self.anchor

        self.hspace = 0.35

        self.y_max = y_max

        self.axs = {}
        self.fig, (self.axs['top'], self.axs['bottom']) = plt.subplots(2, 1,
                                                                       figsize=((self.x_max - self.x_min) * 0.05, 8),
                                                                       sharex=True,
                                                                       gridspec_kw=dict(hspace=self.hspace),
                                                                      )

        if self.pegRNA_name is not None:
            ax_p = self.axs['top'].get_position()

            self.axs['pegRNA'] = self.fig.add_axes((ax_p.x0, ax_p.y1 + ax_p.height * 0.15, ax_p.width, ax_p.height),
                                                   sharex=self.axs['top'],
                                                  )

        self.axs['top'].set_xlim(self.x_min, self.x_max)

        for ax in [self.axs['top'], self.axs['bottom']]:
            ax.axvline(0.5, color='black', alpha=0.25)

        self.draw_genomic_sequence()
        self.annotate_genomic_features()

        if self.pegRNA_name is not None:
            self.draw_pegRNA_sequence()

        self.format_axes()

    def draw_genomic_sequence(self):
        sequence = self.editing_strategy.reference_sequences[self.editing_strategy.target]

        for x in range(self.x_min, self.x_max):
            b = sequence[x + self.anchor]

            common_kwargs = dict(
                xycoords=('data', 'axes fraction'),
                family='monospace',
                size=4,
                ha='center',
            )

            self.axs['top'].annotate(b,
                                     xy=(x, -0.5 * self.hspace + 0.01),
                                     va='bottom',
                                     **common_kwargs,
                                    )

            self.axs['top'].annotate(hits.utilities.complement(b),
                                     xy=(x, -0.5 * self.hspace - 0.01),
                                     va='top',
                                     **common_kwargs,
                                    )

    def annotate_genomic_features(self):
        ax = self.axs['top']

        sgRNA_info = [
            (self.anchor_sgRNA_name, 'pegRNA', 'tab:red'),
            ('oWY408', 'oWY408', bottom_color),
            ('oWY427', 'oWY427', bottom_color),
            ('oWY407', 'oWY407', top_color),
        ]

        transform = ax.get_xaxis_transform()

        for ps_feature_name, label, color in sgRNA_info:
            protospacer_name = knock_knock.pegRNAs.protospacer_name(ps_feature_name)
            ps_feature = self.editing_strategy.features[self.editing_strategy.target, protospacer_name]
            PAM_feature = self.editing_strategy.features[self.editing_strategy.target, f'{protospacer_name}_PAM']

            for feature, alpha in [(ps_feature, 0.3), (PAM_feature, 0.8)]:
                width = len(feature)
                rect_height = 0.045
            
                y = -0.5 * self.hspace
            
                if feature.strand == '-':
                    y -= rect_height

                x = feature.start - self.anchor - 0.5
                
                rect = matplotlib.patches.Rectangle((x, y), width, rect_height,
                                                    transform=transform,
                                                    clip_on=False,
                                                    alpha=alpha,
                                                    color=color,
                                                )
                ax.add_patch(rect)

            x = (min(ps_feature.start, PAM_feature.start) + max(ps_feature.end, PAM_feature.end)) * 0.5 - self.anchor

            y = -0.5 * self.hspace
            
            if feature.strand == '+':
                y += 1.4 * rect_height
                va = 'bottom'
            else:
                y -= 1.4 * rect_height
                va = 'top'
                
            ax.annotate(label,
                        xy=(x, y),
                        xycoords=('data', 'axes fraction'),
                        ha='center',
                        color=color,
                        size=6,
                        va=va,
                        weight='bold',
                       )

        for primer_feature, color in [
            (self.left_primer, top_color),
            (self.right_primer, bottom_color),
        ]:
            if primer_feature.strand == '+':
                xs = [
                    primer_feature.start - self.anchor,
                    primer_feature.end - self.anchor,
                    primer_feature.end - 3 - self.anchor,
                ]

                base_y = -0.5 * self.hspace + 1.5 * rect_height
                ys = [
                    base_y,
                    base_y,
                    base_y + 0.75 * rect_height,
                ]

            else:
                xs = [
                    primer_feature.end - self.anchor,
                    primer_feature.start - self.anchor,
                    primer_feature.start + 3 - self.anchor,
                ]

                base_y = -0.5 * self.hspace - 1.5 * rect_height
                ys = [
                    base_y,
                    base_y,
                    base_y - 0.75 * rect_height,
                ]

            if any(self.x_min <= x <= self.x_max for x in xs):
                ax.plot(xs, ys,
                        transform=transform,
                        clip_on=False,
                        linewidth=2,
                        color=color,
                       )

    def draw_pegRNA_sequence(self):
        ax = self.axs['pegRNA']
        transform = ax.get_xaxis_transform()

        if self.editing_strategy.pegRNA is not None:
            flap_template = ''.join(self.editing_strategy.pegRNA.components[name] for name in ['protospacer', 'scaffold', 'RTT'])
            flap = hits.utilities.reverse_complement(flap_template)
        else:
            flap = ' '*100

        for p, b in enumerate(flap):
            x = p + 1
            if self.x_min <= x <= self.x_max:
                ax.annotate(b,
                            xy=(x, -0.2 * self.hspace + 0.01),
                            xycoords=('data', 'axes fraction'),
                            family='monospace',
                            size=4,
                            ha='center',
                            va='bottom',
                           )

        ax.spines.left.set_position(('data', 0))
        ax.spines.right.set_position(('data', len(flap) + 1))
        ax.spines[['bottom', 'top']].set_visible(False)

        x_bounds = [0, len(flap) + 1]

        for y in [0, self.y_max]:
            ax.plot(x_bounds,
                    [y for x in x_bounds],
                    linewidth=1,
                    clip_on=False,
                    color='black',
                    alpha=1,
                   )

        if self.editing_strategy.pegRNA is not None:
            names = ['RTT', 'scaffold', 'protospacer']
            lengths = [len(self.editing_strategy.pegRNA.components[name]) for name in names]
            starts = [1] + list(np.cumsum(lengths) + 1)[:-1]

            ax.axvspan(0.5, len(self.editing_strategy.pegRNA.components['RTT']) + 0.5,
                                color=knock_knock.pegRNAs.default_feature_colors['RTT'],
                                alpha=0.25,
                                linewidth=0,
                              )
            
            for name, start, length in zip(names, starts, lengths):
                width = length
                rect_height = 0.045
            
                y = -0.2 * self.hspace
            
                x = start - 0.5

                color = knock_knock.pegRNAs.default_feature_colors[name]

                if name == 'protospacer':
                    color = 'tab:red'
                    alpha = 0.3
                else:
                    alpha = 0.8
                
                rect = matplotlib.patches.Rectangle((x, y), width, rect_height,
                                                    transform=transform,
                                                    clip_on=False,
                                                    alpha=alpha,
                                                    color=color,
                                                   )
                ax.add_patch(rect)

                ax.annotate(name,
                            xy=(x + 0.5 * width, y),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, -2),
                            textcoords='offset points',
                            ha='center',
                            color=color,
                            size=6,
                            va='top',
                            weight='bold',
                           )
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

    def format_axes(self):
        for ax in self.axs.values():
            ax.set_ylim(0, self.y_max)

        label_kwargs = dict(
            xy=(0, 0.5),
            xycoords='axes fraction',
            xytext=(-50, 0),
            textcoords='offset points',
            rotation=90,
            size=12,
            va='center',
            ha='center',
        )

        self.axs['top'].annotate('% of top strand reads,\ngenomic',
                                 **label_kwargs,
                                )

        self.axs['bottom'].invert_yaxis()
        self.axs['bottom'].annotate('% of bottom strand reads',
                                    **label_kwargs,
                                   )
        self.axs['bottom'].xaxis.tick_top()

        if self.pegRNA_name is not None:
            self.axs['pegRNA'].set_ylabel('% of top strand reads,\nRT\'ed flap', size=12)

    def add_legend(self, ax_name='top', **kwargs):
        self.axs[ax_name].legend(markerscale=2, **kwargs)

    def plot_fractions(self,
                       data,
                       condition_to_color=round_2_condition_to_color,
                       cumulative=False,
                       condition_keys_to_label=('cell_line', 'time_point', 'is_unedited'),
                       condition_filter=None,
                       condition_to_z_order=None,
                       clip_on=True,
                      ):

        if condition_to_z_order is None:
            condition_to_z_order = lambda c: 1

        for strand in ['top', 'bottom', self.pegRNA_name]:
            if strand not in data:
                continue

            if strand == self.pegRNA_name:
                ax = self.axs['pegRNA']
            else:
                ax = self.axs[strand]

            df = data[strand]

            def sort_key(column):
                condition = OrderedDict(zip(df.columns.names, column))
                return [condition[key] for key in condition_keys_to_label]
            
            for i, column in enumerate(sorted(df.columns, key=sort_key)):
                condition = OrderedDict(zip(df.columns.names, column))

                if condition_filter is not None and not condition_filter(condition):
                    continue

                label = condition_to_label(condition, condition_keys_to_label)

                color = condition_to_color(condition)
                z_order = condition_to_z_order(condition)

                nonzero_ps = df[column]
                nonzero_ps = nonzero_ps[nonzero_ps > 0]

                if strand == 'top' or strand == self.pegRNA_name:
                    cumulative_nonzero_ps = nonzero_ps.cumsum()
                else:
                    cumulative_nonzero_ps = nonzero_ps[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_nonzero_ps
                else:
                    to_plot = nonzero_ps

                ax.plot(to_plot,
                        'o',
                        color=color,
                        markersize=5,
                        clip_on=clip_on,
                        label=label,
                        zorder=z_order,
                       )

                if len(nonzero_ps) > 0:
                    all_xs = np.arange(min(nonzero_ps.index), max(nonzero_ps.index) + 1)
                    all_ys = nonzero_ps.reindex(all_xs, fill_value=0)

                    if strand == 'top' or strand == self.pegRNA_name:
                        cumulative_all_ys = all_ys.cumsum()
                    else:
                        cumulative_all_ys = all_ys[::-1].cumsum()[::-1]

                    if cumulative:
                        to_plot = cumulative_all_ys
                    else:
                        to_plot = all_ys

                    ax.plot(all_xs,
                            to_plot,
                            '-', 
                            color=color,
                            alpha=0.5,
                            linewidth=2.5,
                            clip_on=clip_on,
                            zorder=z_order,
                           )

def load_exp_counts(exp, key=('targeted genomic sequence', 'unedited'), collapse_UMIs=False):
    if collapse_UMIs:
        UMI_counts = Counter((outcome.category, outcome.subcategory, str(outcome.details), outcome.UMI_seq) for outcome in exp.outcome_iter())
        UMI_counts = pd.Series(UMI_counts).sort_values(ascending=False)
        UMI_counts.index.names = ['category', 'subcategory', 'details', 'UMI']
        counts = UMI_counts
    else:
        counts = exp.outcome_counts()

    counts = counts.sort_index()

    if key not in counts.index:
        return None
        
    counts = counts.loc[key]
    
    if counts.index.nlevels > 1:
        if collapse_UMIs:
            counts = counts.groupby(level='details').size()
        else:
            counts = counts.groupby(level='details').sum()

    if key[0] == 'targeted genomic sequence':
        details_key = 'target_edge'
    else:
        details_key = 'pegRNA_edge'
        
    just_edges = [knock_knock.outcome.Details.from_string(d)[details_key] for d in counts.index]

    counts.index = just_edges

    just_edges_counts = counts.groupby(level=0).sum()
    
    if collapse_UMIs:
        denominator = UMI_counts.groupby(level='category').size().reindex(['targeted genomic sequence', 'RTed sequence'], fill_value=0).sum()
    else:
        denominator = exp.outcome_counts(level='category').reindex(['targeted genomic sequence', 'RTed sequence'], fill_value=0).sum()

    just_edges_percentages = just_edges_counts / denominator * 100
    
    return just_edges_percentages
