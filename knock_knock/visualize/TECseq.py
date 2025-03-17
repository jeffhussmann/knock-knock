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

def round_1_condition_to_color(condition):
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

def round_2_condition_to_color(condition):
    palette = bokeh.palettes.Dark2_8

    if condition['time_point'] == '06' and condition['cell_line'] == 'WT':
        color = palette[0]
    elif condition['time_point'] == '72' and condition['cell_line'] == 'WT':
        color = palette[2]
    elif condition['time_point'] == '06' and condition['cell_line'] == 'MLH1_KO':
        color = palette[1]
    elif condition['time_point'] == '72' and condition['cell_line'] == 'MLH1_KO':
        color = palette[3]
    else:
        raise ValueError(condition)

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

        #self.axs['top'].set_ylabel('% of top strand reads,\ngenomic', size=12)

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

    def add_legend(self):
        self.axs['top'].legend(markerscale=2)

    def plot_fractions(self,
                       data,
                       condition_to_color=round_1_condition_to_color,
                       cumulative=False,
                       condition_keys_to_label=('group', 'cell_line', 'time_point', 'in_vitro_Cas9_sgRNA'),
                       condition_filter=None,
                      ):

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
            
            for column in sorted(df, key=sort_key):
                condition = OrderedDict(zip(df.columns.names, column))

                if condition_filter is not None and not condition_filter(condition):
                    continue

                label = condition_to_label(condition, condition_keys_to_label)

                color = condition_to_color(condition)

                nonzero_ps = df[column]
                nonzero_ps = nonzero_ps[nonzero_ps > 0]

                if strand == 'top':
                    cumulative_nonzero_ps = nonzero_ps.cumsum()
                else:
                    cumulative_nonzero_ps = nonzero_ps[::-1].cumsum()[::-1]

                if cumulative:
                    to_plot = cumulative_nonzero_ps
                else:
                    to_plot = nonzero_ps

                ax.plot(to_plot, 'o', color=color, markersize=2, clip_on=True, label=label)

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

                ax.plot(all_xs, to_plot, '-', color=color, markersize=2, clip_on=True)

def load_group_denominator(group):
    denominator = group.outcome_counts(level='category').reindex(['RTed sequence', 'targeted genomic sequence'], fill_value=0).sum()
    return denominator

def load_group_counts(group, key=('targeted genomic sequence', 'unedited')):
    base_dir = '/home/jah/projects/knock-knock_prime/ashley'

    target_name = 'HEK3_annotated'

    sgRNAs = [
        'PRS26322',
        'PRS26323',
        'sgRNA2483',
        'ngRNA_+26',
    ]

    ti = knock_knock.target_info.TargetInfo(base_dir, target_name, sgRNAs=sgRNAs)

    if key not in group.outcome_counts().index:
        return None
        
    counts = group.outcome_counts().sort_index().loc[key]
    
    if counts.index.nlevels > 1:
        counts = counts.groupby(level='details').sum()
        
    just_edges = [knock_knock.TECseq_layout.EdgeMismatchOutcome.from_string(d).undo_anchor_shift(group.target_info.anchor).edge_outcome.edge for d in counts.index]

    counts.index = just_edges

    just_edges_counts = counts.groupby(level=0).sum()
    
    if key[0] == 'RTed sequence':
        offset = -len(group.target_info.pegRNA.components['PBS']) + 1
    else:
        offset = ti.features[('HEK3', group.target_info.sequencing_start_feature_name)].start - group.target_info.features[group.target_info.target, group.target_info.sequencing_start_feature_name].start - ti.cut_afters['sgRNA2483_protospacer_+']
    
    just_edges_counts.index = just_edges_counts.index + offset
    
    if isinstance(just_edges_counts, pd.Series):
        just_edges_counts = just_edges_counts.to_frame()
        just_edges_counts.columns.names = list(group.full_condition_keys)
    
    return just_edges_counts

def load_all_data(batch, genome='hg38', key=('targeted genomic sequence', 'unedited'), min_reads=1000):
    genome_and_sgRNAs = set()

    for _, row in batch.group_descriptions.iterrows():
        sgRNAs = row['sgRNAs']
        if sgRNAs is None:
            sgRNAs = ''
        sgRNAs = sgRNAs.replace(';', '+')
        
        genome = row['genome']
        
        genome_and_sgRNAs.add((genome, sgRNAs))
    
    top_primer = 'OLI19224'
    bottom_primer = 'OLI19254'
    
    data = {}

    pegRNA_names = set()
    
    for genome, sgRNAs in genome_and_sgRNAs:
        gn_suffix = sgRNAs
        group_data = {}
        
        top_group = batch.groups[f'{top_primer}_{genome}_{gn_suffix}_']
        top_counts = load_group_counts(top_group, key=key)
        top_denominator = load_group_denominator(top_group)
        
        if top_counts is not None:
            top_percentages = top_counts / top_denominator * 100
            group_data['top'] = top_percentages[top_denominator[top_denominator > min_reads].index]

        bottom_group = batch.groups[f'{bottom_primer}_{genome}_{gn_suffix}_']
        bottom_counts = load_group_counts(bottom_group, key=key)
        bottom_denominator = load_group_denominator(bottom_group)
        
        if bottom_counts is not None:
            bottom_percentages = bottom_counts / bottom_denominator * 100
            group_data['bottom'] = bottom_percentages[bottom_denominator[bottom_denominator > min_reads].index]

        if top_group.target_info.pegRNA is not None:
            pegRNA_counts = load_group_counts(top_group, key=('RTed sequence', 'n/a'))
            group_data[top_group.target_info.pegRNA.name] = pegRNA_counts / top_denominator * 100

            pegRNA_names.add(top_group.target_info.pegRNA.name)
            
        data[gn_suffix] = group_data

    all_data = {}

    for strand in ['top', 'bottom']:
        flipped = {gn_suffix: data[gn_suffix][strand] for gn_suffix in data if strand in data[gn_suffix]}
        if len(flipped) > 0:
            all_data[strand] = pd.concat({gn_suffix: data[gn_suffix][strand] for gn_suffix in data if strand in data[gn_suffix]}, axis=1, names=['group']).fillna(0).sort_index()

    for pegRNA_name in pegRNA_names:
        all_data[pegRNA_name] = pd.concat({gn_suffix: data[gn_suffix][pegRNA_name] for gn_suffix in data if pegRNA_name in data[gn_suffix]}, axis=1, names=['group']).fillna(0).sort_index()
    
    return all_data