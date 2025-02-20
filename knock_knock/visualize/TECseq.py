from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd

import hits.utilities
import knock_knock.pegRNAs

import bokeh.palettes

top_color = '#FF00FF'
bottom_color = '#00FF00'

def condition_to_color(condition):
    if condition['in_vitro_Cas9_sgRNA'] == 'none':
        palette = bokeh.palettes.Category20b_20

        if condition['time_point'] == '02':
            color = palette[0 + 1]
        elif condition['time_point'] == '06':
            color = palette[12 + 1]
        elif condition['time_point'] == '12':
            color = palette[4 + 1]
        else:
            color = 'black'

    else:
        palette = bokeh.palettes.Category20c_20

        if condition['time_point'] == '02':
            color = palette[0]
        elif condition['time_point'] == '06':
            color = palette[4]
        elif condition['time_point'] == '12':
            color = palette[8]
        else:
            color = 'black'

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

        target_name = 'HEK3_annotated'

        sgRNAs = [
            'PRS26322',
            'PRS26323',
            'PRS26583',
            'PRS26584',
            'sgRNA2483',
            'ngRNA2433',
        ]

        if pegRNA_name is not None:
            sgRNAs.append(pegRNA_name)

        self.pegRNA_name = pegRNA_name

        self.target_info = knock_knock.target_info.TargetInfo(base_dir, target_name, sgRNAs=sgRNAs)

        self.anchor = self.target_info.cut_afters['sgRNA2483_protospacer_+']

        left_primer_name = 'OLI19224'
        right_primer_name = 'OLI19254'

        self.left_primer = self.target_info.features[self.target_info.target, left_primer_name]
        self.right_primer = self.target_info.features[self.target_info.target, right_primer_name]

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

        ax_p = self.axs['top'].get_position()

        self.axs['pegRNA'] = self.fig.add_axes((ax_p.x0, ax_p.y1 + ax_p.height * 0.15, ax_p.width, ax_p.height),
                                               sharex=self.axs['top'],
                                              )


        self.axs['top'].set_xlim(self.x_min, self.x_max)

        for ax in [self.axs['top'], self.axs['bottom']]:
            ax.axvline(0.5, color='black', alpha=0.25)

        self.draw_genomic_sequence()
        self.annotate_genomic_features()

        self.draw_pegRNA_sequence()

        self.format_axes()

    def draw_genomic_sequence(self):
        sequence = self.target_info.reference_sequences[self.target_info.target]

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
            ('sgRNA2483', 'pegRNA', 'tab:red'),
            ('PRS26322', 'PRS26322', bottom_color),
            ('PRS26323', 'PRS26323', top_color),
            ('PRS26583', 'PRS26583', 'tab:grey'),
            ('PRS26584', 'PRS26584', 'tab:grey'),
            ('ngRNA2433', 'PE3 ngRNA', 'tab:grey'),
        ]

        transform = ax.get_xaxis_transform()

        for ps_feature_name, label, color in sgRNA_info:
            protospacer_name = knock_knock.pegRNAs.protospacer_name(ps_feature_name)
            ps_feature = self.target_info.features[self.target_info.target, protospacer_name]
            PAM_feature = self.target_info.features[self.target_info.target, f'{protospacer_name}_PAM']

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
                
            ax.plot(xs, ys,
                    transform=transform,
                    clip_on=False,
                    linewidth=2,
                    color=color,
                   )

    def draw_pegRNA_sequence(self):
        ax = self.axs['pegRNA']
        transform = ax.get_xaxis_transform()

        if self.target_info.pegRNA is not None:
            flap_template = ''.join(self.target_info.pegRNA.components[name] for name in ['protospacer', 'scaffold', 'RTT'])
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

        if self.target_info.pegRNA is not None:
            names = ['RTT', 'scaffold', 'protospacer']
            lengths = [len(self.target_info.pegRNA.components[name]) for name in names]
            starts = [1] + list(np.cumsum(lengths) + 1)[:-1]

            ax.axvspan(0.5, len(self.target_info.pegRNA.components['RTT']) + 0.5,
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

        self.axs['top'].set_ylabel('% of top strand reads,\ngenomic', size=12)

        self.axs['bottom'].invert_yaxis()
        self.axs['bottom'].set_ylabel('% of bottom strand reads', size=12)
        self.axs['bottom'].xaxis.tick_top()

        self.axs['pegRNA'].set_ylabel('% of top strand reads,\nRT\'ed flap', size=12)

    def add_legend(self):
        self.axs['top'].legend(markerscale=2)

    def plot_fractions(self,
                       data,
                       cumulative=False,
                       condition_keys_to_label=('group', 'time_point', 'in_vitro_Cas9_sgRNA'),
                       condition_filter=None,
                      ):

        for strand in ['top', 'bottom']:
            df = data[strand]
            
            for column in df:
                condition = OrderedDict(zip(df.columns.names, column))

                if condition_filter is not None and not condition_filter(condition):
                    continue

                label = condition_to_label(condition, condition_keys_to_label)

                color = condition_to_color(condition)

                nonzero_ps = df[column]

                if strand == 'top':
                    cumulative_nonzero_ps = nonzero_ps.cumsum()
                else:
                    cumulative_nonzero_ps = nonzero_ps[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_nonzero_ps
                else:
                    to_plot = nonzero_ps

                self.axs[strand].plot(to_plot, 'o', color=color, markersize=2, clip_on=True, label=label)

                all_xs = np.arange(min(nonzero_ps.index), max(nonzero_ps.index) + 1)
                all_ys = nonzero_ps.reindex(all_xs, fill_value=0)

                if strand == 'top':
                    cumulative_all_ys = all_ys.cumsum()
                else:
                    cumulative_all_ys = all_ys[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_all_ys
                else:
                    to_plot = all_ys

                self.axs[strand].plot(all_xs, to_plot, '-', color=color, markersize=2, clip_on=True)

        if self.pegRNA_name in data:
            df = data[self.pegRNA_name]
            
            for column in df:
                condition = OrderedDict(zip(df.columns.names, column))

                if condition_filter is not None and not condition_filter(condition):
                    continue

                label = condition_to_label(condition, condition_keys_to_label)

                color = condition_to_color(condition)

                nonzero_ps = df[column]

                if strand == 'top':
                    cumulative_nonzero_ps = nonzero_ps.cumsum()
                else:
                    cumulative_nonzero_ps = nonzero_ps[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_nonzero_ps
                else:
                    to_plot = nonzero_ps

                self.axs['pegRNA'].plot(to_plot, 'o', color=color, markersize=2, clip_on=True)

                all_xs = np.arange(min(nonzero_ps.index), max(nonzero_ps.index) + 1)
                all_ys = nonzero_ps.reindex(all_xs, fill_value=0)

                if strand == 'top':
                    cumulative_all_ys = all_ys.cumsum()
                else:
                    cumulative_all_ys = all_ys[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_all_ys
                else:
                    to_plot = all_ys

                self.axs['pegRNA'].plot(all_xs, to_plot, '-', color=color, markersize=2, clip_on=True, label=label)
