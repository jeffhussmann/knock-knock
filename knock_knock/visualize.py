import sys
import traceback
import copy
import io
import itertools
from collections import defaultdict

import matplotlib
matplotlib.use('Agg', warn=False)

import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import ipywidgets
import pandas as pd

from hits import utilities, interval, sam, sw

from . import experiment as experiment_module
from . import target_info as target_info_module
from . import layout as layout_module

class ReadDiagram():
    def __init__(self, 
                 alignments,
                 target_info,
                 R2_alignments=None,
                 ref_centric=False,
                 parsimonious=False,
                 zoom_in=None,
                 size_multiple=1,
                 paired_end_read_length=None,
                 draw_qualities=False,
                 draw_mismatches=True,
                 draw_polyA=False,
                 draw_sequence=False,
                 max_qual=41,
                 process_mappings=None,
                 detect_orientation=False,
                 label_layout=False,
                 highlight_SNPs=False,
                 reverse_complement=None,
                 label_left=False,
                 flip_donor=False,
                 flip_target=False,
                 read_label='sequencing read',
                 target_on_top=False,
                 features_on_alignments=True,
                 ax=None,
                 features_to_hide=None,
                 features_to_show=None,
                 refs_to_hide=None,
                 draw_edge_numbers=True,
                 hide_non_target_alignments=False,
                 default_color='grey',
                 color_overrides=None,
                 title=None,
                 presentation_mode=False,
                 split_at_indels=False,
                 only_target_and_donor=False,
                 force_left_aligned=False,
                 gap_between_read_pair=5,
                 **kwargs):
        self.parsimonious = parsimonious

        def clean_up_alignments(als):
            if als is None:
                als = []
            als = copy.deepcopy(als)
            if refs_to_hide is not None:
                als = [al for al in als if al.reference_name not in refs_to_hide]

            if self.parsimonious:
                als = interval.make_parsimonious(als)

            if split_at_indels:
                split_als = []
                for al in als:
                    if al.reference_name in [self.target_info.target, self.target_info.donor]:
                        split_at_dels = sam.split_at_deletions(al, 2)
                        split_at_ins = []
                        for al in split_at_dels:
                            split_at_ins.extend(sam.split_at_large_insertions(al, 2))

                        target_seq_bytes = self.target_info.reference_sequences[al.reference_name].encode()
                        extended = [sw.extend_alignment(al, target_seq_bytes) for al in split_at_ins]
                        split_als.extend(extended)

                    else:
                        split_als.append(al)

                als = split_als

            return als

        self.target_info = target_info

        self.alignments = clean_up_alignments(alignments)
        self.R2_alignments = clean_up_alignments(R2_alignments)

        if only_target_and_donor:
            self.alignments = [al for al in self.alignments if al.reference_name in [self.target_info.target, self.target_info.donor]]

        self.ref_centric = ref_centric
        self.target_on_top = target_on_top

        self.zoom_in = zoom_in
        self.size_multiple = size_multiple
        self.paired_end_read_length = paired_end_read_length
        self.draw_qualities = draw_qualities
        self.draw_mismatches = draw_mismatches
        self.draw_polyA = draw_polyA
        self.draw_sequence = draw_sequence
        self.max_qual = max_qual
        self.process_mappings = process_mappings
        self.detect_orientation = detect_orientation
        self.label_layout = label_layout
        self.highlight_SNPs = highlight_SNPs
        self.reverse_complement = reverse_complement
        self.label_left = label_left
        self.flip_donor = flip_donor
        self.flip_target = flip_target
        self.read_label = read_label
        self.draw_edge_numbers = draw_edge_numbers
        self.features_on_alignments = features_on_alignments
        self.hide_non_target_alignments = hide_non_target_alignments
        self.default_color = default_color
        self.presentation_mode = presentation_mode
        self.force_left_aligned = force_left_aligned

        if color_overrides is None:
            color_overrides = {}
        self.color_overrides = color_overrides
        self.title = title
        self.ax = ax

        if features_to_hide is None:
            self.features_to_hide = set()
        else:
            self.features_to_hide = features_to_hide
        
        self.features_to_show = features_to_show
        
        if len(alignments) > 0:
            self.query_length = alignments[0].query_length
            self.query_name = alignments[0].query_name
        else:
            self.query_length = 50
            self.query_name = None
        
        if self.query_length < 750:
            self.width_per_unit = 0.04
        elif 750 <= self.query_length < 2000:
            self.width_per_unit = 0.01
        else:
            self.width_per_unit = 0.005

        self.height_per_unit = 40

        if self.ref_centric:
            self.gap_between_als = 0.003

            self.arrow_linewidth = 3
        else:
            self.gap_between_als = 0.005
            self.arrow_linewidth = 2

        self.arrow_width = self.query_length * 0.01
        self.arrow_height_over_width = self.width_per_unit / self.height_per_unit

        self.gap_between_read_pair = gap_between_read_pair

        self.text_y = -7

        self.cross_x = max(0.5, self.query_length * 0.002)
        self.cross_y = self.cross_x * self.width_per_unit / self.height_per_unit

        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = self.ax.figure
        
        if self.label_left:
            self.label_x = 0
            self.label_ha = 'right'
            self.label_x_offset = -30
        else:
            self.label_x = 1
            self.label_ha = 'left'
            self.label_x_offset = 20
        
        if self.reverse_complement is None:
            if self.paired_end_read_length is not None:
                self.reverse_complement = False

            elif self.detect_orientation and not all(al.is_unmapped for al in self.alignments):
                layout = layout_module.Layout(alignments, self.target_info)
                self.reverse_complement = (layout.strand == '-')

            else:
                self.reverse_complement = False

        self.feature_label_size = 10
        
        self.ref_name_to_color = defaultdict(lambda: default_color)
        self.ref_name_to_color[self.target_info.target] = 'C0'
        self.ref_name_to_color[self.target_info.donor] = 'C1'
        other_names = [n for n in self.target_info.reference_sequences if n not in [self.target_info.target, self.target_info.donor]]
        for i, name in enumerate(other_names):
            self.ref_name_to_color[name] = f'C{i % 8 + 2}'
        
        for name, color in self.color_overrides.items():
            self.ref_name_to_color[name] = color

        self.max_y = self.gap_between_als
        self.min_y = -self.gap_between_als 

        self.min_x = 0
        self.max_x = 1

        self.alignment_coordinates = defaultdict(list)

        self.plot_read()

        if self.ref_centric:
            self.draw_target_and_donor()

        self.update_size()

    def draw_read_arrows(self):
        ''' Draw black arrows that represent the sequencing read or read pair. '''
        arrow_kwargs = {
            'linewidth': self.arrow_linewidth,
            'color': 'black'
        }

        if self.paired_end_read_length is not None:
            # Overlapping arrows to show that the "read" has been stitched together.

            offsets = [0.0007, -0.0007]

            # Cap overhang at a fraction of the overlap length.
            capped_length = min(self.paired_end_read_length, self.query_length * 1.25)

            # If there is an overhang, shift the label down so it doesn't collide.
            if capped_length > self.query_length:
                label_y_offset = -10
            else:
                label_y_offset = 0

            endpoints = [
                [0, capped_length],
                [self.query_length - 1, self.query_length - 1 - capped_length],
            ]

            signs = [
                1,
                -1,
            ]

            for (start, end), sign, offset in zip(endpoints, signs, offsets):
                arrow_xs = [start, end, end - sign * self.arrow_width]
                arrow_ys = [offset, offset, offset + sign * self.arrow_width * self.arrow_height_over_width]
                self.ax.plot(arrow_xs, arrow_ys, clip_on=False, **arrow_kwargs)

        else:
            if not self.R2_alignments:
                arrow_infos = [
                    ([0, self.query_length - 1], self.reverse_complement),
                ]
            else:
                R2_start = self.query_length + self.gap_between_read_pair

                arrow_infos = [
                    ([0, self.query_length - 1], False),
                    ([R2_start, R2_start + self.query_length - 1], True),
                ]


            for (x_start, x_end), reverse_complement in arrow_infos:
                if not reverse_complement:
                    arrow_xs = [x_start, x_end, x_end - self.arrow_width]
                else:
                    arrow_xs = [x_end, x_start, x_start + self.arrow_width]

                arrow_y = self.arrow_width * self.arrow_height_over_width * (-1 if reverse_complement else 1)
                arrow_ys = [0, 0, arrow_y]
                self.ax.plot(arrow_xs, arrow_ys, **arrow_kwargs)

            label_y_offset = 0

        # Draw label on read.
        if self.presentation_mode:
            read_label_size = 18
        else:
            read_label_size = 10

        self.ax.annotate(self.read_label,
                         xy=(self.label_x, 0),
                         xycoords=('axes fraction', 'data'),
                         xytext=(self.label_x_offset, label_y_offset),
                         textcoords='offset points',
                         color='black',
                         ha=self.label_ha,
                         va='center',
                         size=read_label_size,
                        )

    def draw_alignments(self, alignments, is_R2=False):
        ax = self.ax
        alignments = [al for al in alignments if not al.is_unmapped]

        reverse_complement = self.reverse_complement or is_R2
        if is_R2:
            x_offset = self.query_length + self.gap_between_read_pair
        else:
            x_offset = 0

        # Ensure that target and donor are at the bottom followed by other references.
        reference_order = [self.target_info.target, self.target_info.donor]
        other_refs = sorted(set(al.reference_name for al in alignments if al.reference_name not in reference_order))
        reference_order += other_refs

        by_reference_name = defaultdict(list)
        for al in sorted(alignments, key=lambda al: (reference_order.index(al.reference_name), sam.query_interval(al))):
            by_reference_name[al.reference_name].append(al)
        
        if self.ref_centric:
            if self.target_on_top:
                rnames_below = [self.target_info.donor]
            else:
                rnames_below = [self.target_info.target]
            initial_offset = 5
        else:
            rnames_below = []
            initial_offset = 1

        rnames_above = [n for n in by_reference_name if n not in rnames_below]

        offsets = {}
        for names, sign in [(rnames_below, -1), (rnames_above, 1)]:
            block_sizes = [initial_offset] + [len(by_reference_name[n]) + 1 for n in names]
            cumulative_block_sizes = np.cumsum(block_sizes)
            starts = sign * cumulative_block_sizes
            for name, start in zip(names, starts):
                offsets[name] = start
                if is_R2:
                    offsets[name] += sign

        for ref_name, ref_alignments in by_reference_name.items():
            if self.hide_non_target_alignments and ref_name != self.target_info.target:
                alpha_multiplier = 0
            else:
                alpha_multiplier = 1

            if reverse_complement:
                for alignment in ref_alignments:
                    alignment.is_reverse = not alignment.is_reverse

            offset = offsets[ref_name]
            color = self.ref_name_to_color[ref_name]

            average_y = (offset  + 0.5 * (len(ref_alignments) - 1)) * self.gap_between_als
            if (not self.ref_centric) or ref_name not in (self.target_info.target, self.target_info.donor):
                if self.presentation_mode:
                    ref_label_size = 18

                    if ref_name.startswith('hg19'):
                        label = 'hg19 chr{}'.format(ref_name.split('_')[-1])
                    elif ref_name.startswith('bosTau7'):
                        label = 'bosTau7 {}'.format(ref_name.split('_')[-1])
                    else:
                        label = ref_name
                else:
                    label = ref_name
                    ref_label_size = 10

                ax.annotate(label,
                            xy=(self.label_x, average_y),
                            xycoords=('axes fraction', 'data'),
                            xytext=(self.label_x_offset, 0),
                            textcoords='offset points',
                            color=color,
                            ha=self.label_ha,
                            va='center',
                            alpha=1 * alpha_multiplier,
                            size=ref_label_size,
                        )
                        
            for i, alignment in enumerate(ref_alignments):
                start, end = sam.query_interval(alignment)

                def left_offset(x):
                    return x + x_offset - 0.5

                def right_offset(x):
                    return x + x_offset + 0.5

                def middle_offset(x):
                    return x + x_offset

                strand = sam.get_strand(alignment)
                y = (offset + i * np.sign(offset)) * self.gap_between_als
                
                # Annotate the ends of alignments with reference position numbers and vertical lines.
                for x, which, offset_function in ((start, 'start', left_offset), (end, 'end', right_offset)):
                    final_x = offset_function(x)

                    if (which == 'start' and strand == '+') or (which == 'end' and strand == '-'):
                        r = alignment.reference_start
                    else:
                        r = alignment.reference_end - 1

                    ax.plot([final_x, final_x], [0, y], color=color, alpha=0.3 * alpha_multiplier)

                    if which == 'start':
                        kwargs = {'ha': 'right', 'xytext': (-2, 0)}
                    else:
                        kwargs = {'ha': 'left', 'xytext': (2, 0)}

                    draw_numbers = self.draw_edge_numbers or (self.presentation_mode and ('hg19' in ref_name or 'bosTau7' in ref_name))
                    number_size = 8 if self.presentation_mode else 6

                    if draw_numbers:
                        ax.annotate('{0:,}'.format(r),
                                    xy=(final_x, y),
                                    xycoords='data',
                                    textcoords='offset points',
                                    color=color,
                                    va='center',
                                    size=number_size,
                                    alpha=1 * alpha_multiplier,
                                    **kwargs)

                if self.draw_mismatches:
                    mismatches = layout_module.get_mismatch_info(alignment, self.target_info)
                    for read_p, read_b, ref_p, ref_b, q in mismatches:
                        if q < self.max_qual * 0.75:
                            alpha = 0.25
                        else:
                            alpha = 0.85

                        cross_kwargs = dict(zorder=10, color='black', alpha=alpha * alpha_multiplier)
                        cross_ys = [y - self.cross_y, y + self.cross_y]

                        read_x = middle_offset(read_p)

                        ax.plot([read_x - self.cross_x, read_x + self.cross_x], cross_ys, **cross_kwargs)
                        ax.plot([read_x + self.cross_x, read_x - self.cross_x], cross_ys, **cross_kwargs)

                # Draw the alignment, with downward dimples at insertions and upward loops at deletions.
                xs = [left_offset(start)]
                ys = [y]
                indels = sorted(layout_module.get_indel_info(alignment), key=lambda t: t[1][0])
                for kind, info in indels:
                    if kind == 'deletion' or kind == 'splicing':
                        centered_at, length = info

                        if kind == 'deletion':
                            max_length = 100
                            label = str(length)
                        else:
                            max_length = 10
                            label = 'splicing'

                        # Cap how wide the loop can be.
                        capped_length = min(max_length, length)
                        
                        if length <= 1:
                            height = 0.0015
                            indel_xs = [centered_at, centered_at, centered_at]
                            indel_ys = [y, y + height, y]
                        else:
                            width = self.query_length * 0.001
                            height = 0.003

                            indel_xs = [
                                centered_at - width,
                                centered_at - 0.5 * capped_length,
                                centered_at + 0.5 * capped_length,
                                centered_at + width,
                            ]
                            indel_ys = [y, y + height, y + height, y]

                            ax.annotate(label,
                                        xy=(middle_offset(centered_at), y + height),
                                        xytext=(0, 1),
                                        textcoords='offset points',
                                        ha='center',
                                        va='bottom',
                                        size=6,
                                        alpha=1 * alpha_multiplier,
                                    )

                    elif kind == 'insertion':
                        starts_at, ends_at = info
                        centered_at = np.mean([starts_at, ends_at])
                        length = ends_at - starts_at

                        min_height = 0.0015
                        height = min_height * min(length**0.5, 3)
                        if length > 1:
                            ax.annotate(str(length),
                                        xy=(centered_at, y - height),
                                        xytext=(0, -1),
                                        textcoords='offset points',
                                        ha='center',
                                        va='top',
                                        size=6,
                                        alpha=1 * alpha_multiplier,
                                    )
                        indel_xs = [starts_at, centered_at, ends_at]
                        indel_ys = [y, y - height, y]
                        
                    xs.extend([middle_offset(x) for x in indel_xs])
                    ys.extend(indel_ys)
                    
                xs.append(right_offset(end))
                ys.append(y)

                ref_ps = (alignment.reference_start, alignment.reference_end - 1)
                if alignment.is_reverse:
                    ref_ps = ref_ps[::-1]

                coordinates = [
                    (middle_offset(start), middle_offset(end)),
                    ref_ps,
                    y,
                ]
                self.alignment_coordinates[ref_name].append(coordinates)
                
                self.max_y = max(self.max_y, max(ys))
                self.min_y = min(self.min_y, min(ys))
                
                kwargs = {'color': color, 'linewidth': 1.5, 'alpha': 1 * alpha_multiplier}
                ax.plot(xs, ys, **kwargs)

                length = end - start

                capped_arrow_width = min(self.arrow_width, length * 0.3)
                
                if strand == '+':
                    arrow_xs = [right_offset(end), right_offset(end - capped_arrow_width)]
                    arrow_ys = [y, y + capped_arrow_width * self.arrow_height_over_width]
                else:
                    arrow_xs = [left_offset(start), left_offset(start + capped_arrow_width)]
                    arrow_ys = [y, y - capped_arrow_width * self.arrow_height_over_width]
                    
                draw_arrow = True
                if self.zoom_in is not None:
                    if not all(self.min_x <= x <= self.max_x for x in arrow_xs):
                        draw_arrow = False

                if draw_arrow:
                    ax.plot(arrow_xs, arrow_ys, clip_on=False, **kwargs)

                features = copy.deepcopy(self.target_info.features)

                if self.features_to_show is not None:
                    features_to_show = self.features_to_show
                else:
                    features_to_show = [
                        (r_name, f_name) for r_name, f_name in features
                        if r_name == ref_name and
                        'edge' not in f_name and
                        'SNP' not in f_name and 
                        features[r_name, f_name].feature != 'sgRNA'
                    ]

                    features_to_show.extend([(ref_name, f_name) for f_name in self.target_info.sgRNA_features])

                q_to_r = {
                    sam.true_query_position(q, alignment): r
                    for q, r in alignment.aligned_pairs
                    if r is not None and q is not None
                }

                if self.highlight_SNPs:
                    if ref_name == self.target_info.donor:
                        SNVs = self.target_info.donor_SNVs['donor']
                    elif ref_name == self.target_info.target:
                        SNVs = self.target_info.donor_SNVs['target']
                    else:
                        SNVs = {}

                    for SNV_name, SNV_info in SNVs.items():
                        SNV_r = SNV_info['position']
                        qs = [q for q, r in q_to_r.items() if r == SNV_r]
                        if len(qs) != 1:
                            continue

                        q = qs[0]

                        left_x = q - 0.5
                        right_x = q + 0.5
                        bottom_y = y - (self.cross_y * 3)
                        top_y = y + (self.cross_y * 3)
                        path_xs = [left_x, right_x, right_x, left_x]
                        path_ys = [bottom_y, bottom_y, top_y, top_y]
                        path = np.array([path_xs, path_ys]).T
                        patch = plt.Polygon(path, color='black', alpha=0.2, linewidth=0)
                        ax.add_patch(patch)
                        
                for feature_reference, feature_name in features_to_show:
                    if ref_name != feature_reference:
                        continue

                    if (feature_reference, feature_name) not in features:
                        continue

                    if feature_name in self.features_to_hide:
                        continue

                    feature = features[feature_reference, feature_name]
                    feature_color = feature.attribute.get('color', 'grey')
                    
                    qs = [q for q, r in q_to_r.items() if feature.start <= r <= feature.end]
                    if not qs:
                        continue

                    query_extent = [min(qs), max(qs)]
                    
                    rs = [feature.start, feature.end]
                    if strand == '-':
                        rs = rs[::-1]

                    if np.sign(offset) == 1:
                        va = 'bottom'
                        text_y = 1
                    else:
                        va = 'top'
                        text_y = -1

                    if not self.ref_centric:
                        for ha, q, r in zip(['left', 'right'], query_extent, rs):
                            nts_missing = abs(q_to_r[q] - r)
                            if nts_missing != 0 and query_extent[1] - query_extent[0] > 20:
                                ax.annotate(str(nts_missing),
                                            xy=(q, 0),
                                            ha=ha,
                                            xytext=(3 if ha == 'left' else -3, text_y),
                                            textcoords='offset points',
                                            size=6,
                                            va=va,
                                        )
                        
                        if query_extent[1] - query_extent[0] > 18 or feature.attribute['ID'] == self.target_info.sgRNA:
                            label = feature.attribute['ID']

                            ax.annotate(label,
                                        xy=(np.mean(query_extent), 0),
                                        xycoords='data',
                                        xytext=(0, self.text_y),
                                        textcoords='offset points',
                                        va='top',
                                        ha='center',
                                        color=feature_color,
                                        size=10,
                                        weight='bold',
                                    )

                    if self.features_on_alignments:
                        xs = [min(qs) - 0.5 + x_offset, max(qs) + 0.5 + x_offset]
                        ax.fill_between(xs, [y] * 2, [0] * 2, color=feature_color, alpha=0.7)
                        

    def plot_read(self):
        ax = self.ax
        alignments = self.alignments

        if (not alignments) or (alignments[0].query_sequence is None):
            return self.fig

        if self.process_mappings is not None:
            layout_info = self.process_mappings(alignments, self.target_info)
            alignments = layout_info['to_plot']

        if self.zoom_in is not None:
            self.min_x = self.zoom_in[0]
            self.max_x = self.zoom_in[1]
        else:
            if not self.R2_alignments:
                total_query_length = self.query_length
            else:
                total_query_length = 2 * self.query_length + self.gap_between_read_pair

            self.min_x = -0.02 * total_query_length
            self.max_x = 1.02 * total_query_length
            
        self.draw_read_arrows()

        self.draw_alignments(self.alignments)
        if self.R2_alignments:
            self.draw_alignments(self.R2_alignments, True)

        if self.title is None:
            if self.label_layout:
                layout = layout_module.Layout(alignments, self.target_info)
                cat, subcat, details = layout.categorize()
                title = '{}\n{}, {}, {}'.format(self.query_name, cat, subcat, details)
            else:
                title = self.query_name
        else:
            title = self.title

        ax.set_title(title, y=1.02)
            
        ax.set_ylim(1.1 * self.min_y, 1.1 * self.max_y)
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_yticks([])
        
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_alpha(0.1)
        for edge in 'left', 'top', 'right':
            ax.spines[edge].set_color('none')
            
        if not self.ref_centric:
            ax.tick_params(pad=14)

        if self.draw_qualities:
            quals = alignments[0].query_qualities
            if alignments[0].is_reverse:
                quals = quals[::-1]

            qual_ys = np.array(quals) * self.max_y / self.max_qual
            ax.plot(qual_ys, color='black', alpha=0.5)

            if self.R2_alignments:
                quals = self.R2_alignments[0].query_qualities
                if not self.R2_alignments[0].is_reverse:
                    quals = quals[::-1]

                qual_ys = np.array(quals) * self.max_y / self.max_qual
                x_start = self.query_length + self.gap_between_read_pair
                xs = x_start + np.arange(self.query_length)[::-1]
                ax.plot(xs, qual_ys, color='black', alpha=0.5)

        if self.draw_polyA:
            seq = alignments[0].get_forward_sequence()
            for b, color in [('A', 'red'), ('G', 'brown')]:
                locations = utilities.homopolymer_lengths(seq, b)
                for start, length in locations:
                    if length > 10:
                        ax.fill_between([start, start + length - 1], [self.max_y + self.arrow_height] * 2, [0] * 2, color=color, alpha=0.2)
                        
                        ax.annotate('poly{}'.format(b),
                                    xy=(start + length / 2, 0),
                                    xycoords='data',
                                    xytext=(0, self.text_y),
                                    textcoords='offset points',
                                    va='top',
                                    ha='center',
                                    color=color,
                                    alpha=0.4,
                                    size=10,
                                    weight='bold',
                                )
                        
        if self.draw_sequence:
            seq = alignments[0].get_forward_sequence()

            for x, b in enumerate(seq):
                if self.min_x <= x <= self.max_x:
                    ax.annotate(b,
                                xy=(x, 0),
                                family='monospace',
                                size=3.5 * self.size_multiple,
                                xytext=(0, -2),
                                textcoords='offset points',
                                ha='center',
                                va='top',
                            )
            
            if self.R2_alignments:
                seq = self.R2_alignments[0].get_forward_sequence()

                for x, b in enumerate(seq):
                    x += self.query_length + self.gap_between_read_pair
                    if self.min_x <= x <= self.max_x:
                        ax.annotate(b,
                                    xy=(x, 0),
                                    family='monospace',
                                    size=4,
                                    xytext=(0, 2),
                                    textcoords='offset points',
                                    ha='center',
                                    va='bottom',
                                )
            
        return self.fig

    def draw_target_and_donor(self):
        if len(self.alignments) == 0:
            return

        def adjust_edges(xs):
            xs = list(xs)
            if xs[0] < xs[1]:
                xs[0] -= 0.5
                xs[1] += 0.5
            else:
                xs[0] += 0.5
                xs[1] -= 0.5
            return xs

        ti = self.target_info
        gap = 0.03
        
        if self.target_on_top:
            target_y = self.max_y + gap
            donor_y = self.min_y - gap
        else:
            target_y = self.min_y - gap
            donor_y = self.max_y + gap

        params = []

        if len(self.alignment_coordinates[ti.target]) > 0:
            params.append((ti.target, min(ti.cut_afters.values()), target_y, self.flip_target))

        if len(self.alignment_coordinates[ti.donor]) > 0:
            if (ti.donor, ti.donor_specific) in ti.features:
                donor_specific_feature = ti.features[ti.donor, ti.donor_specific]
                middle = np.mean([donor_specific_feature.start, donor_specific_feature.end])
            else:
                middle = len(ti.donor_sequence) / 2

            params.append((ti.donor, middle, donor_y, self.flip_donor))

        for ref_name, center_p, ref_y, flip in params:
            color = self.ref_name_to_color[ref_name]

            # To establish a mapping between reference position and x coordinate,
            # pick anchor points on the ref and read that will line up with each other. 
            if self.force_left_aligned or (len(self.alignment_coordinates[ref_name]) == 1 and ref_name == ti.target):
                xs, ps, y = self.alignment_coordinates[ref_name][0]
                anchor_ref = ps[0]
                anchor_read = xs[0]
            else:
                anchor_ref = center_p
                anchor_read = self.query_length // 2
            
            # With these anchors picked, define the mapping and its inverse.
            if flip:
                ref_p_to_x = lambda p: anchor_read - (p - anchor_ref)
                x_to_ref_p = lambda x: (anchor_read - x) + anchor_ref
            else:
                ref_p_to_x = lambda p: (p - anchor_ref) + anchor_read
                x_to_ref_p = lambda x: (x - anchor_read) + anchor_ref

            ref_edge = len(ti.reference_sequences[ref_name]) - 1

            # ref_start and ref_end are the smallest and largest ref positions
            # that get plottted. Initially set these to the inverse image of
            # the edges of the current x lims.
            if flip:
                left, right = self.max_x, self.min_x
            else:
                left, right = self.min_x, self.max_x

            ref_start = max(0, x_to_ref_p(left))
            ref_end = min(ref_edge, x_to_ref_p(right))

            ref_al_min = ref_start
            ref_al_max = ref_end

            for xs, ps, y in self.alignment_coordinates[ref_name]:
                ref_xs = [ref_p_to_x(p) for p in ps]
                ref_al_min = min(ref_al_min, min(ps))
                ref_al_max = max(ref_al_max, max(ps))

                xs = adjust_edges(xs)
                ref_xs = adjust_edges(ref_xs)

                # Shade parallelograms between alignments and reference.
                self.ax.fill_betweenx([y, ref_y], [xs[0], ref_xs[0]], [xs[1], ref_xs[1]], color=color, alpha=0.05)

                # Draw lines connecting alignment edges to reference.
                for x, ref_x in zip(xs, ref_xs):
                    self.ax.plot([x, ref_x], [y, ref_y], color=color, alpha=0.3)

            self.min_y = min(self.min_y, ref_y)
            self.max_y = max(self.max_y, ref_y)

            if ref_al_min <= ref_start:
                ref_start = max(0, ref_al_min - 10)

            if ref_al_max >= ref_end:
                ref_end = min(ref_edge, ref_al_max + 10)

            new_left = ref_p_to_x(ref_start)
            new_right = ref_p_to_x(ref_end)
            if flip:
                new_left, new_right = new_right, new_left

            self.min_x = min(self.min_x, new_left)
            self.max_x = max(self.max_x, new_right)

            self.ax.set_xlim(self.min_x, self.max_x)

            # Draw the actual reference.
            ref_xs = adjust_edges([ref_p_to_x(ref_start), ref_p_to_x(ref_end)])
            self.ax.plot(ref_xs, [ref_y, ref_y], color=color, linewidth=3, solid_capstyle='butt')

            if self.features_to_show is not None:
                features_to_show = [(r_name, f_name) for r_name, f_name in self.features_to_show
                                    if r_name == ref_name
                                   ]
            else:
                features_to_show = [(r_name, f_name) for r_name, f_name in ti.features
                                    if r_name == ref_name
                                    and 'edge' not in f_name
                                    and 'SNP' not in f_name
                                    and 'sgRNA' not in f_name
                                   ]

                for sgRNA in self.target_info.sgRNAs:
                    features_to_show.append((self.target_info.target, sgRNA))

            # Draw features.
            for feature_reference, feature_name in features_to_show:
                if feature_reference != ref_name:
                    continue

                if feature_name is None or feature_name in self.features_to_hide:
                    continue

                feature = ti.features.get((feature_reference, feature_name))
                if feature is None:
                    continue
                feature_color = feature.attribute.get('color', 'grey')
                
                xs = adjust_edges([ref_p_to_x(p) for p in [feature.start, feature.end]])
                    
                start = ref_y
                end = ref_y + np.sign(ref_y) * gap * 0.2

                bottom = min(start, end)
                top = max(start, end)
                self.min_y = min(bottom, self.min_y)
                self.max_y = max(top, self.max_y)

                if min(xs) >= self.min_x and max(xs) <= self.max_x:
                    self.ax.fill_between(xs, [start] * 2, [end] * 2, color=feature_color, alpha=0.7, linewidth=0)

                    name = feature.attribute['ID']

                    if self.presentation_mode:
                        size = 16
                        if 'sgRNA' in name:
                            label = 'sgRNA'
                        else:
                            label = ''
                    else:
                        size = 10
                        label = name

                    self.ax.annotate(label,
                                     xy=(np.mean(xs), end),
                                     xycoords='data',
                                     xytext=(0, 2 * np.sign(ref_y)),
                                     textcoords='offset points',
                                     va='top' if ref_y < 0 else 'bottom',
                                     ha='center',
                                     color=feature_color,
                                     size=size,
                                     weight='bold',
                                    )

            # Draw target and donor names next to diagrams.
            if self.presentation_mode:
                size = 18

                if ref_name == ti.target:
                    label = 'cut site sequence'
                elif ref_name == ti.donor:
                    label = 'single-strand\ndonor sequence'
                else:
                    label = ref_name

            else:
                size = 10
                label = ref_name

            self.ax.annotate(label,
                             xy=(self.label_x, ref_y),
                             xycoords=('axes fraction', 'data'),
                             xytext=(self.label_x_offset, 0),
                             textcoords='offset points',
                             color=color,
                             ha=self.label_ha,
                             va='center',
                             size=size,
                            )

            if ref_name == ti.target:
                for cut_after in ti.cut_afters.values():
                    cut_after_x = ref_p_to_x(cut_after)
                    self.ax.plot([cut_after_x, cut_after_x], [ref_y - gap * 0.18, ref_y + gap * 0.18], '--', color='black')
            
        self.ax.set_ylim(self.min_y - 0.1 * self.height, self.max_y + 0.1 * self.height)

    @property
    def height(self):
        return self.max_y - self.min_y
    
    @property
    def width(self):
        return self.max_x - self.min_x

    def update_size(self):
        #fig_width = 0.04 * (self.width + 50) * self.size_multiple
        #fig_height = 40 * self.height * self.size_multiple

        fig_width = self.width_per_unit * max(self.width, 50) * self.size_multiple
        fig_height = self.height_per_unit * 1.2 * self.height * self.size_multiple

        self.fig.set_size_inches((fig_width, fig_height))

def make_stacked_Image(diagrams, titles=None, **kwargs):
    if titles is None or titles == '':
        titles = itertools.repeat(titles)

    ims = []

    for diagram, title in zip(diagrams, titles):
        if title is not None:
            diagram.fig.axes[0].set_title(title)
        
        with io.BytesIO() as buffer:
            diagram.fig.savefig(buffer, format='png', bbox_inches='tight')
            im = PIL.Image.open(buffer)
            im.load()
            ims.append(im)

        plt.close(diagram.fig)
        
    if not ims:
        return None

    total_height = sum(im.height for im in ims)
    max_width = max(im.width for im in ims)

    stacked_im = PIL.Image.new('RGBA', size=(max_width, total_height), color='white')
    y_start = 0
    for im in ims:
        stacked_im.paste(im, (max_width - im.width, y_start))
        y_start += im.height

    return stacked_im
