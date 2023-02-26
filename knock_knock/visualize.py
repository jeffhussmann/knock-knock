import copy
from collections import defaultdict

import matplotlib
if 'inline' not in matplotlib.get_backend():
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from hits import utilities, interval, sam, sw
import hits.visualize

from . import layout as layout_module

memoized_property = utilities.memoized_property

def adjust_edges(xs):
    # Expand by 0.5 on both sides. Since whole numbers correspond to the center
    # of nts, this has the effect of including the entire nt for nts on the edge.
    xs = list(xs)

    if xs[0] < xs[1]:
        xs[0] -= 0.5
        xs[1] += 0.5
    else:
        xs[0] += 0.5
        xs[1] -= 0.5

    return xs

class ReadDiagram():
    def __init__(self, 
                 alignments,
                 target_info,
                 ref_centric=True,
                 parsimonious=False,
                 size_multiple=1,
                 draw_qualities=False,
                 draw_mismatches=True,
                 draw_sequence=False,
                 draw_ref_sequences=False,
                 max_qual=41,
                 process_mappings=None,
                 detect_orientation=False,
                 label_layout=False,
                 highlight_SNPs=False,
                 reverse_complement=False,
                 label_left=False,
                 flip_donor=False,
                 flip_target=False,
                 read_label='sequencing read',
                 donor_below=False,
                 manual_refs_below=None,
                 features_on_alignments=True,
                 ax=None,
                 features_to_hide=None,
                 features_to_show=None,
                 feature_heights=None,
                 refs_to_hide=None,
                 draw_edge_numbers=True,
                 hide_non_target_alignments=False,
                 hide_target_alignments=False,
                 hide_donor_alignments=False,
                 default_color='grey',
                 color_overrides=None,
                 title=None,
                 title_y=1.02,
                 mode='normal',
                 layout_mode='illumina',
                 label_differences=False,
                 label_overrides=None,
                 split_at_indels=False,
                 only_target_and_donor=False,
                 force_left_aligned=False,
                 force_right_aligned=False,
                 width_per_unit=None,
                 arrow_width=None,
                 emphasize_parsimonious=False,
                 manual_x_lims=None,
                 label_offsets=None,
                 center_on_primers=False,
                 invisible_alignments=None,
                 query_interval=None,
                 hide_xticks=False,
                 inferred_amplicon_length=None,
                 manual_anchors=None,
                 manual_fade=None,
                 refs_to_draw=None,
                 parallelogram_alpha=0.05,
                 supplementary_reference_sequences=None,
                 **kwargs,
                ):

        self.parsimonious = parsimonious
        self.emphasize_parismonious = emphasize_parsimonious
        self.ref_centric = ref_centric
        self.donor_below = donor_below
        self.manual_refs_below = manual_refs_below
        self.size_multiple = size_multiple
        self.draw_qualities = draw_qualities
        self.draw_mismatches = draw_mismatches
        self.draw_sequence = draw_sequence
        self.draw_ref_sequences = draw_ref_sequences
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
        self.hide_target_alignments = hide_target_alignments
        self.hide_donor_alignments = hide_donor_alignments
        self.default_color = default_color
        self.mode = mode
        self.layout_mode = layout_mode
        self.force_left_aligned = force_left_aligned
        self.force_right_aligned = force_right_aligned
        self.center_on_primers = center_on_primers
        self.manual_x_lims = manual_x_lims
        self.hide_xticks = hide_xticks
        self.inferred_amplicon_length = inferred_amplicon_length
        self.manual_fade = manual_fade
        self.label_differences = label_differences
        self.draw_arrowheads = kwargs.get('draw_arrowheads', True)
        self.refs_to_draw = refs_to_draw
        self.parallelogram_alpha = parallelogram_alpha

        self.target_info = target_info

        if supplementary_reference_sequences is None:
            supplementary_reference_sequences = {}

        self.all_reference_sequences = {**self.target_info.reference_sequences, **supplementary_reference_sequences}

        if self.refs_to_draw is None:
            self.refs_to_draw = set()

        if self.manual_refs_below is None:
            self.manual_refs_below = []
        
        if len(self.refs_to_draw) == 0 and self.ref_centric:
            self.refs_to_draw.add(self.target_info.target)
            if self.target_info.donor is not None:
                self.refs_to_draw.add(self.target_info.donor)

        if invisible_alignments is None:
            invisible_alignments = []
        self.invisible_alignments = invisible_alignments

        if manual_anchors is None:
            manual_anchors = {}
        self.manual_anchors = manual_anchors

        # If query_interval is initially None, will be set
        # to whole query after alignments are cleaned up.
        self.query_interval = query_interval

        if self.query_interval is not None:
            if isinstance(self.query_interval, int):
                self.query_interval = (0, self.query_interval)

        def clean_up_alignments(als):
            if als is None:
                als = []

            als = copy.deepcopy(als)
            als = [al for al in als if al is not None]

            if refs_to_hide is not None:
                als = [al for al in als if al.reference_name not in refs_to_hide]

            if split_at_indels:
                all_split_als = []

                refs_to_split = [self.target_info.target, self.target_info.donor]
                if self.target_info.pegRNA_names is not None:
                    refs_to_split.extend(self.target_info.pegRNA_names)

                for al in als:
                    if al.reference_name in refs_to_split:
                        split_als = layout_module.comprehensively_split_alignment(al, self.target_info, self.layout_mode)

                        seq_bytes = self.target_info.reference_sequence_bytes[al.reference_name]
                        extended = [sw.extend_alignment(al, seq_bytes) for al in split_als]
                        all_split_als.extend(extended)

                    else:
                        all_split_als.append(al)

                als = all_split_als

            # Truncation needs to happen after extension or extension will undo it.
            if self.query_interval is not None:
                # If reverse complement, query_interval is specific relative to flipped read and needs to be un-flipped
                # before applying
                if reverse_complement:
                    query_length = als[0].query_length
                    start, end = self.query_interval
                    query_interval_to_apply = (query_length - 1 - end, query_length - 1 - start)
                else:
                    query_interval_to_apply = self.query_interval

                als = [sam.crop_al_to_query_int(al, *query_interval_to_apply) for al in als]
                als = [al for al in als if al is not None]

            if only_target_and_donor:
                als = [al for al in als if al.reference_name in [self.target_info.target, self.target_info.donor]]

            all_als = als

            als = sam.make_nonredundant(als)

            if self.parsimonious:
                als = interval.make_parsimonious(als)

            return als, all_als

        if isinstance(alignments, dict):
            self.alignments, _ = clean_up_alignments(alignments['R1'])
            self.R2_alignments, _ = clean_up_alignments(alignments['R2'])
        else:
            self.alignments, self.all_alignments = clean_up_alignments(alignments)
            self.R2_alignments = None

        self.ref_line_width = kwargs.get('ref_line_width', 0.001) * self.size_multiple

        if label_offsets is None:
            label_offsets = {}
        self.label_offsets = label_offsets

        if self.mode == 'normal':
            self.font_sizes = {
                'read_label': 10,
                'ref_label': 10,
                'feature_label': 10,
                'number': 6,
                'sequence': 3.5 * self.size_multiple,
                'title': 16,
            }

            self.target_and_donor_y_gap = kwargs.get('target_and_donor_y_gap', 0.03)
            self.initial_alignment_y_offset = 5
            self.feature_line_width = 0.005
            self.gap_between_als = 0.003
            self.label_cut = False

            if self.label_left:
                self.label_x_offset = -30
            else:
                self.label_x_offset = 20

        elif self.mode == 'paper':
            self.font_sizes = {
                'read_label': kwargs.get('label_size', 12),
                'ref_label': kwargs.get('label_size', 12),
                'feature_label': kwargs.get('feature_label_size', 12),
                'number': kwargs.get('number_size', 8),
                'sequence': 3.5 * self.size_multiple,
                'title': 16,
            }

            self.target_and_donor_y_gap = kwargs.get('target_and_donor_y_gap', 0.015)
            self.initial_alignment_y_offset = kwargs.get('initial_alignment_y_offset', 5)
            self.feature_line_width = kwargs.get('feature_line_width', 0.005)
            self.gap_between_als = kwargs.get('gap_between_als', 0.003)
            self.label_cut = kwargs.get('label_cut', False)

            if self.label_left:
                self.label_x_offset = -10
            else:
                self.label_x_offset = 20

        elif self.mode == 'compact':
            self.font_sizes = {
                'read_label': 12,
                'ref_label': 12,
                'feature_label': 12,
                'number': 8,
                'sequence': 3.5 * self.size_multiple,
                'title': 22,
            }

            self.target_and_donor_y_gap = kwargs.get('target_and_donor_y_gap', 0.01)
            self.initial_alignment_y_offset = 3
            self.feature_line_width = 0.004
            self.gap_between_als = 0.003
            self.label_cut = kwargs.get('label_cut', False)

            if self.label_left:
                self.label_x_offset = -5
            else:
                self.label_x_offset = 5
        
        if color_overrides is None:
            color_overrides = {}

        self.color_overrides = color_overrides
        self.title = title
        self.title_y = title_y
        self.ax = ax

        self.reference_ys = {}

        if features_to_hide is None:
            features_to_hide = set()
        self.features_to_hide = features_to_hide
        
        self.features_to_show = features_to_show

        if feature_heights is None:
            feature_heights = {}

        self.feature_heights = feature_heights
        
        if len(self.alignments) > 0:
            self.query_length = self.alignments[0].query_length
            self.query_name = self.alignments[0].query_name
        else:
            self.query_length = 50
            self.query_name = None

        if self.R2_alignments is not None:
            self.R1_query_length = self.query_length
            self.R2_query_length = self.R2_alignments[0].query_length

            if self.inferred_amplicon_length is None or self.inferred_amplicon_length == -1:
                self.total_query_length = self.R1_query_length + self.R2_query_length + 20
            else:
                self.total_query_length = self.inferred_amplicon_length

            self.R2_query_start = self.total_query_length - self.R2_query_length
        else:
            self.total_query_length = self.query_length

        if self.query_interval is None:
            # TODO: -1 confuses me.
            self.query_interval = (0, self.total_query_length - 1)

        self.query_length = self.query_interval[1] - self.query_interval[0] + 1

        if width_per_unit is None:
            if self.query_length < 750:
                width_per_unit = 0.04
            elif 750 <= self.query_length < 2000:
                width_per_unit = 0.01
            elif 2000 <= self.query_length < 5000:
                width_per_unit = 0.005
            else:
                width_per_unit = 0.003

        self.width_per_unit = width_per_unit

        self.height_per_unit = 40

        if self.ref_centric:
            self.arrow_linewidth = 3
        else:
            self.arrow_linewidth = 2

        if arrow_width is None:
            arrow_width = self.query_length * 0.012
        self.arrow_width = arrow_width
        self.arrow_height_over_width = self.width_per_unit / self.height_per_unit

        self.text_y = -7

        # cross_x and cross_y are the width and height of each X arm
        # (i.e. half the width of the whole X)

        self.cross_x = kwargs.get('cross_x', 0.6)
        self.cross_y = self.cross_x * self.width_per_unit / self.height_per_unit

        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = self.ax.figure
        
        if self.label_left:
            self.label_x = 0
            self.label_ha = 'right'
        else:
            self.label_x = 1
            self.label_ha = 'left'
        
        if self.reverse_complement is None:
            if self.detect_orientation and not all(al.is_unmapped for al in self.alignments):
                layout = layout_module.Layout(alignments, self.target_info)
                self.reverse_complement = (layout.strand == '-')

            else:
                self.reverse_complement = False

        if label_overrides is None:
            label_overrides = {}

        self.label_overrides = label_overrides

        self.feature_label_size = 10
        
        self.ref_name_to_color = defaultdict(lambda: default_color)

        unused_colors = {f'C{i}' for i in range(10)} - set(self.color_overrides.values())

        def assign_ref_color(ref_name):
            if ref_name in self.color_overrides:
                color = self.color_overrides[ref_name]
            elif len(unused_colors) > 0:
                color = min(unused_colors)
                unused_colors.remove(color)
            else:
                color = 'grey'

            self.ref_name_to_color[ref_name] = color

        assign_ref_color(self.target_info.target)

        if self.target_info.donor is not None:
            assign_ref_color(self.target_info.donor)

        other_names = [n for n in self.target_info.reference_sequences if n not in [self.target_info.target, self.target_info.donor]]
        for name in other_names:
            assign_ref_color(name)
        
        if self.target_info.reference_name_in_genome_source is not None:
            self.ref_name_to_color[self.target_info.reference_name_in_genome_source] = self.ref_name_to_color[self.target_info.target]

        self.max_y = self.gap_between_als
        self.min_y = -self.gap_between_als 

        self.min_x = 0
        self.max_x = 1

        self.alignment_coordinates = defaultdict(list)

        self.plot_read()

        if self.ref_centric:
            self.draw_target_and_donor()

        self.update_size()

    @property
    def seq(self):
        seq = self.alignments[0].get_forward_sequence()
        if self.reverse_complement:
            seq = utilities.reverse_complement(seq)
        return seq

    @property
    def R2_seq(self):
        R2_seq = self.R2_alignments[0].get_forward_sequence()
        return R2_seq

    @memoized_property
    def features(self):
        all_features = self.target_info.features

        if self.features_to_show is not None:
            features_to_show = self.features_to_show
        else:
            features_to_show = [
                (r_name, f_name) for r_name, f_name in all_features
                if 'edge' not in f_name and 'SNP' not in f_name and all_features[r_name, f_name].feature != 'sgRNA'
            ]

            features_to_show.extend([(self.target_info.target, f_name) for f_name in self.target_info.protospacer_names])

        features = {k: v for k, v in all_features.items()
                    if k in features_to_show
                    and k not in self.features_to_hide
                   }

        return features

    def get_feature_color(self, feature_reference, feature_name):
        feature = self.features[feature_reference, feature_name]

        if (feature_reference, feature_name) in self.color_overrides:
            color = self.color_overrides[feature_reference, feature_name]
        elif feature_name in self.color_overrides:
            color = self.color_overrides[feature_name]
        else:
            color = feature.attribute.get('color')
            if color is None:
                color = self.default_color

        return color

    def get_feature_height(self, feature_reference, feature_name):
        if (feature_reference, feature_name) in self.feature_heights:
            height = self.feature_heights[feature_reference, feature_name]
        elif feature_name in self.feature_heights:
            height = self.feature_heights[feature_name]
        else:
            height = 1

        return height

    def get_feature_label(self, feature_reference, feature_name):
        if (feature_reference, feature_name) in self.label_overrides:
            label = self.label_overrides[feature_reference, feature_name]
        else:
            label = self.label_overrides.get(feature_name, feature_name)

        return label

    def draw_read_arrows(self):
        ''' Draw black arrows that represent the sequencing read or read pair. '''

        arrow_kwargs = {
            'linewidth': self.arrow_linewidth * self.size_multiple,
            'color': 'black',
            'solid_capstyle': 'butt',
        }

        if self.R2_alignments is None:
            arrow_infos = [
                (self.query_interval, self.reverse_complement),
            ]

        else:
            arrow_infos = [
                ([0, self.R1_query_length - 1], False),
                ([self.R2_query_start, self.total_query_length - 1], True),
            ]

        for (x_start, x_end), reverse_complement in arrow_infos:
            if not reverse_complement:
                arrow_xs = [x_start - 0.5, x_end + 0.5]
                if self.draw_arrowheads:
                    arrow_xs.append(x_end - self.arrow_width)
            else:
                arrow_xs = [x_end + 0.5, x_start - 0.5]
                if self.draw_arrowheads:
                    arrow_xs.append(x_start + self.arrow_width)

            arrow_y = self.arrow_width * self.arrow_height_over_width
            arrow_ys = [0, 0]
            if self.draw_arrowheads:
                arrow_ys.append(arrow_y)

            self.ax.plot(arrow_xs, arrow_ys, **arrow_kwargs)

        label_y_offset = 0

        self.ax.annotate(self.read_label,
                         xy=(self.label_x, 0),
                         xycoords=('axes fraction', 'data'),
                         xytext=(self.label_x_offset, label_y_offset),
                         textcoords='offset points',
                         color='black',
                         ha=self.label_ha,
                         va='center',
                         size=self.font_sizes['read_label'],
                        )

    def draw_alignments(self, alignments, is_R2=False):
        ax = self.ax

        # Copy before possibly fiddling with is_reverse below.
        alignments = copy.deepcopy([al for al in alignments if not al.is_unmapped])

        reverse_complement = self.reverse_complement or is_R2
        if is_R2:
            x_offset = self.R2_query_start
        else:
            x_offset = 0

        # Ensure that target and donor are closest to the read, followed by other references.
        reference_order = [self.target_info.target, self.target_info.donor]
        other_refs = sorted(set(al.reference_name for al in alignments if al.reference_name not in reference_order))
        reference_order += other_refs

        by_reference_name = defaultdict(list)
        for al in sorted(alignments, key=lambda al: (reference_order.index(al.reference_name), sam.query_interval(al))):
            by_reference_name[al.reference_name].append(al)

        # Prevent further population of defaultdict.
        by_reference_name = dict(by_reference_name)
        
        if self.ref_centric:
            rnames_below = [self.target_info.target]
            if self.donor_below:
                rnames_below.append(self.target_info.donor)

            rnames_below += self.manual_refs_below

            initial_offset = self.initial_alignment_y_offset

        else:
            rnames_below = []
            initial_offset = 1

        rnames_above = [n for n in by_reference_name if n not in rnames_below]

        offsets = {}
        for names, sign in [(rnames_below, -1), (rnames_above, 1)]:
            block_sizes = [initial_offset] + [len(by_reference_name.get(n, [])) + 1 for n in names]
            cumulative_block_sizes = np.cumsum(block_sizes)
            starts = sign * cumulative_block_sizes
            for name, start in zip(names, starts):
                offsets[name] = start
                if is_R2:
                    offsets[name] += sign

        for ref_name, ref_alignments in by_reference_name.items():

            hide_multiplier = 1
            if self.hide_non_target_alignments and ref_name != self.target_info.target:
                hide_multiplier = 0
            if self.hide_target_alignments and ref_name == self.target_info.target:
                hide_multiplier = 0
            if self.hide_donor_alignments and ref_name == self.target_info.donor:
                hide_multiplier = 0

            if reverse_complement:
                for alignment in ref_alignments:
                    alignment.is_reverse = not alignment.is_reverse

            offset = offsets[ref_name]
            color = self.ref_name_to_color[ref_name]

            average_y = (offset  + 0.5 * (len(ref_alignments) - 1)) * self.gap_between_als

            if (not self.ref_centric) or ref_name not in self.refs_to_draw:

                label = self.label_overrides.get(ref_name, ref_name)

                ax.annotate(label,
                            xy=(self.label_x, average_y),
                            xycoords=('axes fraction', 'data'),
                            xytext=(self.label_x_offset, 0),
                            textcoords='offset points',
                            color=color,
                            ha=self.label_ha,
                            va='center',
                            alpha=1 * hide_multiplier,
                            size=self.font_sizes['ref_label'],
                        )
                        
            for i, alignment in enumerate(ref_alignments):
                if self.emphasize_parismonious:
                    if alignment in self.alignments:
                        parsimony_multiplier = 1
                        parsimony_width_multiplier = 2
                    else:
                        parsimony_multiplier = 0.1
                        parsimony_width_multiplier = 1
                else:
                    parsimony_multiplier = 1
                    parsimony_width_multiplier = 1

                alpha_multiplier = parsimony_multiplier * hide_multiplier

                if alignment in self.invisible_alignments:
                    alpha_multiplier = 0

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
                r_start, r_end = alignment.reference_start, alignment.reference_end - 1
                if ref_name == self.target_info.reference_name_in_genome_source:
                    converted_coords = self.target_info.convert_genomic_alignment_to_target_coordinates(alignment)
                    if converted_coords:
                        r_start = converted_coords['start']
                        r_end = converted_coords['end'] - 1

                for x, which, offset_function in ((start, 'start', left_offset), (end, 'end', right_offset)):
                    final_x = offset_function(x)

                    if (which == 'start' and strand == '+') or (which == 'end' and strand == '-'):
                        r = r_start
                    else:
                        r = r_end

                    ax.plot([final_x, final_x], [0, y], color=color, alpha=0.3 * alpha_multiplier)

                    if which == 'start':
                        kwargs = {'ha': 'right', 'xytext': (-2, 0)}
                    else:
                        kwargs = {'ha': 'left', 'xytext': (2, 0)}

                    if self.draw_edge_numbers:

                        ax.annotate(f'{r:,}',
                                    xy=(final_x, y),
                                    xycoords='data',
                                    textcoords='offset points',
                                    color=color,
                                    va='center',
                                    size=self.font_sizes['number'],
                                    alpha=1 * alpha_multiplier,
                                    **kwargs,
                                   )

                if self.draw_mismatches:
                    mismatches = layout_module.get_mismatch_info(alignment, self.all_reference_sequences)
                    for mismatch_i, (read_p, read_b, ref_p, ref_b, q) in enumerate(mismatches):
                        if q < self.max_qual * 0.75:
                            alpha = 0.25
                        else:
                            alpha = 0.85

                        cross_kwargs = dict(zorder=10, linewidth=self.size_multiple, color='black', alpha=alpha * alpha_multiplier)
                        cross_ys = [y - self.cross_y, y + self.cross_y]

                        read_x = middle_offset(read_p)

                        ax.plot([read_x - self.cross_x, read_x + self.cross_x], cross_ys, **cross_kwargs)
                        ax.plot([read_x + self.cross_x, read_x - self.cross_x], cross_ys, **cross_kwargs)

                        if self.label_differences:
                            ax.annotate('mismatch',
                                        xy=(read_x, y + self.cross_y),
                                        xycoords='data',
                                        xytext=(0, 2 + 10 * mismatch_i),
                                        textcoords='offset points',
                                        ha='center',
                                        va='bottom',
                                        size=10,
                                       )
                                       
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
                        length = ends_at - starts_at + 1

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

                        indel_xs = [starts_at - 0.5, centered_at, ends_at + 0.5]
                        indel_ys = [y, y - height, y]

                        if self.label_differences:
                            ax.annotate(f'{length}-nt\ninsertion',
                                        xy=(centered_at, y - height),
                                        xycoords='data',
                                        xytext=(0, -2),
                                        textcoords='offset points',
                                        ha='center',
                                        va='top',
                                       )
                        
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
                    sam.get_strand(alignment),
                    alpha_multiplier,
                ]
                self.alignment_coordinates[ref_name].append(coordinates)
                
                self.max_y = max(self.max_y, max(ys))
                self.min_y = min(self.min_y, min(ys))
                
                kwargs = {
                    'color': color,
                    'linewidth': 1.5 * parsimony_width_multiplier * self.size_multiple,
                    'alpha': 1 * alpha_multiplier,
                    'solid_capstyle': 'butt',
                }

                ax.plot(xs, ys, **kwargs)

                length = end - start

                capped_arrow_width = min(self.arrow_width, length * 0.3)
                
                if strand == '+':
                    arrow_xs = [right_offset(end), right_offset(end - capped_arrow_width)]
                    arrow_ys = [y, y + capped_arrow_width * self.arrow_height_over_width]
                else:
                    arrow_xs = [left_offset(start), left_offset(start + capped_arrow_width)]
                    arrow_ys = [y, y - capped_arrow_width * self.arrow_height_over_width]
                    
                draw_arrow = self.draw_arrowheads
                if not all(self.min_x <= x <= self.max_x for x in arrow_xs):
                    draw_arrow = False

                if draw_arrow:
                    ax.plot(arrow_xs, arrow_ys, clip_on=False, **kwargs)

                q_to_r = {
                    sam.true_query_position(q, alignment): r
                    for q, r in alignment.aligned_pairs
                    if r is not None and q is not None
                }

                if self.highlight_SNPs:
                    SNVs = {}

                    if self.target_info.pegRNA_SNVs is not None:
                        if ref_name in self.target_info.pegRNA_SNVs:
                            SNVs = self.target_info.pegRNA_SNVs[ref_name]
                    
                    elif self.target_info.donor is not None:
                        donor_name = self.target_info.donor

                        if ref_name == donor_name:
                            SNVs = self.target_info.donor_SNVs['donor']
                        elif ref_name == self.target_info.target:
                            SNVs = self.target_info.donor_SNVs['target']

                    if len(SNVs) == 1:
                        box_half_width = self.cross_x * 1.5
                        box_half_height = self.cross_y * 2.5
                    else:
                        box_half_width = 0.5
                        box_half_height = self.cross_y * 2.5

                    for SNV_name, SNV_info in SNVs.items():
                        SNV_r = SNV_info['position']
                        qs = [q for q, r in q_to_r.items() if r == SNV_r]
                        if len(qs) != 1:
                            continue

                        q = qs[0]

                        left_x = q - box_half_width
                        right_x = q + box_half_width
                        bottom_y = y - box_half_height
                        top_y = y + box_half_height
                        path_xs = [left_x, right_x, right_x, left_x]
                        path_ys = [bottom_y, bottom_y, top_y, top_y]
                        path = np.array([path_xs, path_ys]).T
                        patch = plt.Polygon(path, color='black', alpha=0.25, linewidth=0)
                        ax.add_patch(patch)

                        if self.label_differences:
                            if ref_name == self.target_info.donor:
                                ax.annotate('edit\nposition',
                                            xy=(q, y + box_half_height),
                                            xycoords='data',
                                            xytext=(0, 2),
                                            textcoords='offset points',
                                            ha='center',
                                            va='bottom',
                                           )

                for feature_reference, feature_name in self.features:
                    if ref_name != feature_reference:
                        continue

                    if feature_name in self.features_to_hide:
                        continue

                    feature = self.features[feature_reference, feature_name]
                    feature_color = self.get_feature_color(feature_reference, feature_name)
                    
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

                            label_offset = self.label_offsets.get(label, 0)
                            y_points = -5 - label_offset * self.font_sizes['feature_label']

                            ax.annotate(label,
                                        xy=(np.mean(query_extent), 0),
                                        xycoords='data',
                                        xytext=(0, y_points),
                                        textcoords='offset points',
                                        va='top',
                                        ha='center',
                                        color=feature_color,
                                        size=10,
                                        weight='bold',
                                    )

                    if self.features_on_alignments:
                        xs = [min(qs) - 0.5 + x_offset, max(qs) + 0.5 + x_offset]
                        final_feature_color = hits.visualize.apply_alpha(feature_color, alpha=0.7 * alpha_multiplier, multiplicative=True)
                        ax.fill_between(xs, [y] * 2, [0] * 2, color=final_feature_color, edgecolor='none')
                        
    def plot_read(self):
        ax = self.ax

        if (not self.alignments) or (self.alignments[0].query_sequence is None):
            return self.fig

        self.min_x = self.query_interval[0] - 0.05 * self.total_query_length
        self.max_x = self.query_interval[1] + 0.05 * self.total_query_length
            
        self.draw_read_arrows()

        if self.emphasize_parismonious:
            self.draw_alignments(self.all_alignments)

            if self.R2_alignments is not None:
                raise NotImplementedError
        else:
            self.draw_alignments(self.alignments)

            if self.R2_alignments is not None:
                self.draw_alignments(self.R2_alignments, is_R2=True)

        if self.title is None:
            if self.label_layout:
                layout = layout_module.Layout(self.alignments, self.target_info)
                cat, subcat, details = layout.categorize()
                title = f'{self.query_name}\n{cat}, {subcat}, {details}'
            else:
                title = self.query_name
        else:
            title = self.title

        ax.set_title(title, y=self.title_y, size=self.font_sizes['title'])
            
        ax.set_ylim(1.1 * self.min_y, 1.1 * self.max_y)
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_yticks([])

        if self.hide_xticks:
            ax.set_xticks([])
        
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_alpha(0.1)
        for edge in 'left', 'top', 'right':
            ax.spines[edge].set_color('none')
            
        # Set how far tick labels should be from axes
        if self.draw_sequence:
            ax.tick_params(length=0, pad=self.font_sizes['sequence'] * 2)
        else:
            ax.tick_params(pad=2)

        ax.tick_params(labelsize=self.font_sizes['number'])

        if self.draw_qualities:
            def quals_to_ys(quals):
                return (np.array(quals) + 5) / self.max_qual * (self.initial_alignment_y_offset - 1) * self.gap_between_als

            qual_ys = quals_to_ys(self.alignments[0].get_forward_qualities())

            ax.plot(qual_ys, color='black', alpha=0.5)

            if self.R2_alignments is not None:
                qual_ys = quals_to_ys(self.R2_alignments[0].get_forward_qualities())

                x_start = self.R2_query_start
                xs = x_start + np.arange(self.R2_query_length)[::-1]

                ax.plot(xs, qual_ys, color='black', alpha=0.5)

            label_y = quals_to_ys([self.max_qual])[0]

            self.ax.annotate('quality scores',
                             xy=(self.label_x, label_y),
                             xycoords=('axes fraction', 'data'),
                             xytext=(self.label_x_offset, 0),
                             textcoords='offset points',
                             color='black',
                             ha=self.label_ha,
                             va='center',
                             size=int(np.floor(self.font_sizes['ref_label'] * 0.75)),
                            )

        if self.draw_sequence:
            # Note that query_interval is relative to reverse complement if relevant.
            start, end = self.query_interval
            seq_to_draw = self.seq[start:end + 1]

            # Drawing sequences for very long (i.e. Pabcio) reads is not
            # useful, and takes a long time.
            if len(seq_to_draw) <= 1000:

                seq_kwargs = dict(family='monospace',
                                size=self.font_sizes['sequence'],
                                ha='center',
                                textcoords='offset points',
                                va='top',
                                xytext=(0, -2 * self.size_multiple),
                                )

                for x, b in zip(range(start, end + 1), seq_to_draw):
                    if self.min_x <= x <= self.max_x:
                        ax.annotate(b, xy=(x, 0), **seq_kwargs)
                
                if self.R2_alignments is not None:
                    x_start = self.R2_query_start
                    for x_offset, b in enumerate(self.R2_seq):
                        x = x_start + x_offset
                        if self.min_x <= x <= self.max_x:
                            ax.annotate(b, xy=(x, 0), **seq_kwargs)
            
        return self.fig

    def draw_reference(self, ref_name, ref_y, flip,
                       label_features=True,
                       visible=True,
                      ):
        ti = self.target_info

        color = self.ref_name_to_color[ref_name]

        self.reference_ys[ref_name] = ref_y

        alignment_coordinates = self.alignment_coordinates[ref_name]

        # To establish a mapping between reference position and x coordinate,
        # pick anchor points on the ref and read that will line up with each other. 
        if ref_name in self.manual_anchors:
            anchor_read, anchor_ref = self.manual_anchors[ref_name]

        # Default to lining up the left edge vertically with the reference position
        # it is mapped to if there is only one alignment to this reference, or
        # if there are two but they might just be a single alignment split across R1
        # and R2.
        elif (self.force_left_aligned or 
              (len(alignment_coordinates) == 1) or 
              (len(alignment_coordinates) == 2 and ref_name == ti.donor and self.R2_alignments is not None) 
             ):
            xs, ps, y, strand, parsimony_multiplier = alignment_coordinates[0]

            anchor_ref = ps[0]

            if (strand == '+' and not flip) or (strand == '-' and flip):
                anchor_read = xs[0]
            else:
                anchor_read = xs[1]

        elif self.force_right_aligned:
            xs, ps, y, strand, parsimony_multiplier = alignment_coordinates[-1]

            anchor_ref = ps[-1]

            if (strand == '+' and not flip) or (strand == '-' and flip):
                anchor_read = xs[1]
            else:
                anchor_read = xs[0]

        else:
            if self.inferred_amplicon_length is not None and self.inferred_amplicon_length != -1:
                relevant_length = self.inferred_amplicon_length
            else:
                relevant_length = self.query_length

            if ref_name == ti.target and self.center_on_primers:
                if self.flip_target:
                    anchor_ref = ti.amplicon_interval.end - (len(ti.amplicon_interval) - relevant_length) / 2
                else:
                    anchor_ref = ti.amplicon_interval.start + (len(ti.amplicon_interval) - relevant_length) / 2

                leftmost_coordinates = alignment_coordinates[0]
                xs, ps, y, strand, parsimony_multiplier = leftmost_coordinates

                if (strand == '+' and not flip) or (strand == '-' and flip):
                    anchor_read = xs[0]
                else:
                    anchor_read = xs[1]

            elif len(alignment_coordinates) > 0:
                # Line up the longest alignment.
                longest_coordinates = max(alignment_coordinates, key=lambda t: abs(t[1][1] - t[1][0]))
                xs, ps, y, strand, parsimony_multiplier = longest_coordinates
                anchor_ref = ps[0]
                if (strand == '+' and not flip) or (strand == '-' and flip):
                    anchor_read = xs[0]
                else:
                    anchor_read = xs[1]

            else:
                anchor_read = relevant_length // 2
                anchor_ref = len(ti.reference_sequences[ref_name]) // 2

        # With these anchors picked, define the mapping and its inverse.
        if flip:
            ref_p_to_x = lambda p: anchor_read - (p - anchor_ref)
            x_to_ref_p = lambda x: (anchor_read - x) + anchor_ref
        else:
            ref_p_to_x = lambda p: (p - anchor_ref) + anchor_read
            x_to_ref_p = lambda x: (x - anchor_read) + anchor_ref

        ref_edge = len(ti.reference_sequences[ref_name]) - 1

        # ref_start and ref_end are the smallest and largest ref positions
        # that get plotted. Initially, set these to the inverse image of
        # the edges of the current x lims.
        if flip:
            left, right = self.max_x, self.min_x
        else:
            left, right = self.min_x, self.max_x

        ref_start = max(0, x_to_ref_p(left))
        ref_end = min(ref_edge, x_to_ref_p(right))

        ref_al_min = np.inf
        ref_al_max = -np.inf

        for xs, ps, y, strand, parsimony_multiplier in alignment_coordinates:
            ref_xs = [ref_p_to_x(p) for p in ps]
            ref_al_min = min(ref_al_min, min(ps))
            ref_al_max = max(ref_al_max, max(ps))

            xs = adjust_edges(xs)
            ref_xs = adjust_edges(ref_xs)

            # Shade parallelograms between alignments and reference.
            if ref_y < 0:
                ref_border_y = ref_y + self.ref_line_width
            else:
                ref_border_y = ref_y - self.ref_line_width

            self.ax.fill_betweenx([y, ref_border_y], [xs[0], ref_xs[0]], [xs[1], ref_xs[1]],
                                  color=color,
                                  alpha=self.parallelogram_alpha * parsimony_multiplier,
                                  visible=visible,
                                 )

            # Draw lines connecting alignment edges to reference.
            for x, ref_x in zip(xs, ref_xs):
                self.ax.plot([x, ref_x], [y, ref_border_y],
                             color=color,
                             alpha=0.3 * parsimony_multiplier,
                             clip_on=False,
                             visible=visible,
                            )

        if ref_name == ti.target and self.center_on_primers:
            ref_al_min = min(ref_al_min, ti.amplicon_interval.start)
            ref_al_max = max(ref_al_max, ti.amplicon_interval.end)

        self.min_y = min(self.min_y, ref_y)
        self.max_y = max(self.max_y, ref_y)

        if ref_al_min <= ref_start:
            ref_start = max(0, ref_al_min - 10)

        if ref_al_max >= ref_end:
            ref_end = min(ref_edge, ref_al_max + 10)

        if ref_name == ti.target:
            fade_left = True
            fade_right = True
        else:
            fade_left = (ref_start > 0)
            fade_right = (ref_end < ref_edge)

        new_left = ref_p_to_x(ref_start)
        new_right = ref_p_to_x(ref_end)

        leftmost_aligned_x = ref_p_to_x(ref_al_min)
        rightmost_aligned_x = ref_p_to_x(ref_al_max)

        if flip:
            new_left, new_right = new_right, new_left
            leftmost_aligned_x, rightmost_aligned_x = rightmost_aligned_x, leftmost_aligned_x

        if self.manual_x_lims is not None:
            self.min_x, self.max_x = self.manual_x_lims
        else:
            self.min_x = min(self.min_x, new_left - 0.5)
            self.max_x = max(self.max_x, new_right + 0.5)

        # If an alignment goes right up to the edge, don't fade.
        if leftmost_aligned_x <= self.min_x + 0.05 * (self.max_x - self.min_x):
            fade_left = False

        if rightmost_aligned_x >= self.min_x + 0.95 * (self.max_x - self.min_x):
            fade_right = False

        self.ax.set_xlim(self.min_x, self.max_x)

        # Draw the actual reference.
        ref_xs = adjust_edges([ref_p_to_x(ref_start), ref_p_to_x(ref_end)])

        if ref_name == ti.target:
            # Always extend the target all the way to the edges.
            ref_xs[0] = min(ref_xs[0], self.min_x)
            ref_xs[1] = max(ref_xs[1], self.max_x)

        rgba = matplotlib.colors.to_rgba(color)
        image = np.expand_dims(np.array([rgba]*1000), 0)

        if self.manual_fade is not None:
            fade_left = self.manual_fade[ref_name]
            fade_right = self.manual_fade[ref_name]

        if fade_left:
            left_alpha = 0
        else:
            left_alpha = 1

        if fade_right:
            right_alpha = 0
        else:
            right_alpha = 1

        image[:, :, 3] = np.concatenate([np.linspace(left_alpha, 1, 100),
                                         [1] * 800,
                                         np.linspace(1, right_alpha, 100),
                                        ],
                                       )

        self.ax.imshow(image,
                       extent=(ref_xs[0], ref_xs[1], ref_y - self.ref_line_width, ref_y + self.ref_line_width),
                       aspect='auto',
                       interpolation='none',
                       zorder=3,
                       visible=visible,
                      )

        # Draw features.

        for feature_reference, feature_name in self.features:
            if feature_reference != ref_name:
                continue

            feature = self.features[feature_reference, feature_name]
            feature_color = self.get_feature_color(feature_reference, feature_name)
            feature_height = self.get_feature_height(feature_reference, feature_name)

            xs = adjust_edges([ref_p_to_x(p) for p in [feature.start, feature.end]])
                
            start = ref_y + np.sign(ref_y) * np.sign(feature_height) * self.ref_line_width
            end = start + np.sign(ref_y) * self.feature_line_width * feature_height

            bottom = min(start, end)
            top = max(start, end)
            self.min_y = min(bottom, self.min_y)
            self.max_y = max(top, self.max_y)

            plot_interval = interval.Interval(self.min_x, self.max_x)

            left = max(min(xs), self.min_x)
            right = min(max(xs), self.max_x)
            feature_interval = interval.Interval(min(xs), max(xs))

            if interval.are_overlapping(plot_interval, feature_interval):
                left = max(min(xs), self.min_x)
                right = min(max(xs), self.max_x)

                final_feature_color = hits.visualize.apply_alpha(feature_color, alpha=0.7, multiplicative=True)
                self.ax.fill_between([left, right], [start] * 2, [end] * 2,
                                     color=final_feature_color,
                                     edgecolor='none',
                                     visible=visible,
                                    )

                if label_features:
                    name = feature.attribute['ID']

                    label = self.get_feature_label(feature.seqname, name)

                    if label is None:
                        continue

                    if 'PAM' in label:
                        label = 'PAM'

                    label_offset = self.label_offsets.get(name, 0)
                    y_points = 5 * self.feature_line_width / 0.005 + label_offset * (2 + self.font_sizes['feature_label'])
                    y_direction = np.sign(ref_y) * np.sign(feature_height)

                    self.ax.annotate(label,
                                     xy=(np.mean([left, right]), end),
                                     xycoords='data',
                                     xytext=(0, y_points * y_direction),
                                     textcoords='offset points',
                                     va='top' if y_direction < 0 else 'bottom',
                                     ha='center',
                                     color=feature_color,
                                     size=self.font_sizes['feature_label'],
                                     weight='bold',
                                     visible=visible,
                                     annotation_clip=False,
                                    )

        # Draw target and donor names next to diagrams.
        label = self.label_overrides.get(ref_name, ref_name)

        self.ax.annotate(label,
                         xy=(self.label_x, ref_y),
                         xycoords=('axes fraction', 'data'),
                         xytext=(self.label_x_offset, 0),
                         textcoords='offset points',
                         color=color,
                         ha=self.label_ha,
                         va='center',
                         size=self.font_sizes['ref_label'],
                         visible=visible,
                        )

        if ref_name == ti.target:
            # Draw the cut site(s).
            for cut_after_name, cut_after in ti.cut_afters.items():
                cut_after_x = ref_p_to_x(cut_after + 0.5)

                name, strand = cut_after_name.rsplit('_', 1)

                cut_y_bottom = ref_y - self.feature_line_width
                cut_y_middle = ref_y
                cut_y_top = ref_y + self.feature_line_width

                if strand == 'both':
                    ys = [cut_y_bottom, cut_y_top]
                elif (strand == '+' and not self.flip_target) or (strand == '-' and self.flip_target):
                    ys = [cut_y_middle, cut_y_top]
                elif (strand == '-' and not self.flip_target) or (strand == '+' and self.flip_target):
                    ys = [cut_y_bottom, cut_y_middle]
                else:
                    raise ValueError(strand)

                self.ax.plot([cut_after_x, cut_after_x],
                             ys,
                             '-',
                             linewidth=1,
                             color='black',
                             solid_capstyle='butt',
                             zorder=10,
                             visible=visible,
                            )

                color = ti.PAM_features[f'{name}_PAM'].attribute['color']

                if self.label_cut:
                    label = self.label_overrides.get(f'{name}_cut', f'{name}_cut')
                    self.ax.annotate(label,
                                     xy=(cut_after_x, cut_y_bottom),
                                     xycoords='data',
                                     xytext=(0, 10 * self.feature_line_width / 0.005 * np.sign(ref_y)),
                                     textcoords='offset points',
                                     color=color,
                                     ha='center',
                                     va='top' if ref_y < 0 else 'bottom',
                                     size=self.font_sizes['ref_label'],
                                    )

        if self.draw_ref_sequences:
            ref_sequence = ti.reference_sequences[ref_name]
            
            seq_kwargs = dict(family='monospace',
                              size=self.font_sizes['sequence'],
                              ha='center',
                              textcoords='offset points',
                              va='center',
                              xytext=(0, 0),
                              visible=visible,
                             )

            start = int(np.ceil(ref_start))
            end = int(np.floor(ref_end))
            for ref_p in range(start, end + 1):
                x = ref_p_to_x(ref_p)
                b = ref_sequence[ref_p]
                if flip:
                    b = utilities.complement(b)
                self.ax.annotate(b, xy=(x, ref_y), **seq_kwargs)
                    
        self.ax.set_ylim(self.min_y - 0.1 * self.height, self.max_y + 0.1 * self.height)

        return ref_p_to_x

    def draw_target_and_donor(self):
        if len(self.alignments) == 0:
            return

        ti = self.target_info
        
        target_y = self.min_y - self.target_and_donor_y_gap

        if self.donor_below:
            donor_y = target_y - self.target_and_donor_y_gap
        else:
            donor_y = self.max_y + self.target_and_donor_y_gap

        params = []

        if len(self.alignment_coordinates[ti.target]) > 0:
            params.append((ti.target, target_y, self.flip_target))

        if len(self.alignment_coordinates[ti.donor]) > 0:
            params.append((ti.donor, donor_y, self.flip_donor))

        for ref_name, ref_y, flip in params:
            self.draw_reference(ref_name, ref_y, flip)

    @property
    def height(self):
        return self.max_y - self.min_y
    
    @property
    def width(self):
        return self.max_x - self.min_x

    def update_size(self):
        fig_width = self.width_per_unit * max(self.width, 50) * self.size_multiple
        fig_height = self.height_per_unit * 1.2 * self.height * self.size_multiple

        self.fig.set_size_inches((fig_width, fig_height))