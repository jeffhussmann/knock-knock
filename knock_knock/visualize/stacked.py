import copy
import warnings

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hits.utilities
import hits.visualize

import knock_knock.pegRNAs
from knock_knock.target_info import degenerate_indel_from_string, SNV, SNVs
from knock_knock.outcome import *

@dataclass
class StackedDiagrams:
    outcome_order: list
    target_infos: Union[knock_knock.target_info.TargetInfo, dict]

    ax: Optional[plt.Axes] = None
    block_alpha: float = 0.1
    center_at_PAM: bool = False
    cut_color: Optional[Any] = hits.visualize.apply_alpha('black', 0.5)
    draw_all_sequence: Union[bool, float] = False
    draw_cuts: bool = True
    draw_donor_on_top: bool = False
    draw_imperfect_MH: bool = False
    draw_insertion_degeneracy: bool = True
    draw_perfect_MH: bool = True
    draw_wild_type_on_top: bool = False
    features_to_draw: Any = field(default_factory=list)
    flip_if_reverse: bool = True
    flip_MH_deletion_boundaries: bool = False
    force_flip: bool = False
    inches_per_nt: float = 0.12
    inches_per_outcome: float = 0.25
    line_widths: float = 1.5
    num_outcomes: Optional[int] = None
    color_overrides: dict = field(default_factory=dict)
    preserve_x_lims: bool = False
    replacement_text_for_complex: dict = field(default_factory=dict)
    shift_x: float = 0
    text_size: int = 8
    title: Optional[str] = None
    title_size: int = 14
    title_offset: int = 20
    title_color: Optional[Any] = 'black'
    window: int = 70

    del_multiple = 0.25
    wt_height = 0.6

    def __post_init__(self):
        # Can either supply outcomes as (c, s, d) tuples along with a single target_info,
        # or outcomes as (source_name, c, s, d) tuples along with a {source_name: target_info} dict.

        if isinstance(self.target_infos, knock_knock.target_info.TargetInfo):
            self.target_infos = {self.target_infos.name: self.target_infos}

        self.single_source_name = sorted(self.target_infos)[0]

        if all(len(outcome) == 3 for outcome in self.outcome_order):
            self.outcome_order = [(self.single_source_name, c, s, d) for c, s, d in self.outcome_order]

        if isinstance(self.window, int):
            self.window_left, self.window_right = -self.window, self.window
        else:
            self.window_left, self.window_right = self.window

        self.window_size = self.window_right - self.window_left + 1

        if self.num_outcomes is None:
            self.num_outcomes = len(self.outcome_order)

        self.outcome_order = self.outcome_order[:self.num_outcomes]

        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(self.inches_per_nt * self.window_size, self.inches_per_outcome * self.num_outcomes))
        else:
            self.fig = self.ax.figure

        if isinstance(self.draw_all_sequence, float):
            self.sequence_alpha = self.draw_all_sequence
        else:
            self.sequence_alpha = 0.1
            
        self.offsets = {}
        self.seqs = {}
        self.transform_seqs = {}
        self.windows = {}
        self.flip = {}
        self.guide = {}

        for source_name, ti in self.target_infos.items():
            self.guide[source_name] = ti.features[ti.target, ti.primary_protospacer]

            # TODO: flip behavior for multiple sources
            if self.force_flip or (self.flip_if_reverse and self.guide[source_name].strand == '-'):
                self.flip[source_name] = True
                transform_seq = hits.utilities.complement
            else:
                self.flip[source_name] = False
                transform_seq = hits.utilities.identity

            if self.center_at_PAM:
                if self.guide[source_name].strand == '+':
                    offset = ti.PAM_slice.start
                else:
                    offset = ti.PAM_slice.stop - 1
            else:
                offset = max(v for n, v in ti.cut_afters.items() if n.startswith(ti.primary_protospacer))

            self.offsets[source_name] = offset

            if self.flip[source_name]:
                this_window_left, this_window_right = -self.window_right, -self.window_left
            else:
                this_window_left, this_window_right = self.window_left, self.window_right

            self.windows[source_name] = (this_window_left, this_window_right)

            seq = ti.target_sequence[offset + this_window_left:offset + this_window_right + 1]
            self.seqs[source_name] = seq
            self.transform_seqs[source_name] = transform_seq

        cuts_drawn_at = set()
        if self.draw_cuts:
            for source_name, ti in self.target_infos.items():
                offset = self.offsets[source_name]
                window_left, window_right = self.windows[source_name]

                for cut_after in ti.cut_afters.values():
                    # temp fix for empirically-determined offset of Cpf1 cut.
                    x = (cut_after + 0.5) - offset + self.shift_x

                    if self.draw_wild_type_on_top and not self.draw_donor_on_top:
                        ys = [-0.5, self.num_outcomes + 0.5]
                    else:
                        ys = [-0.5, self.num_outcomes - 0.5]

                    if x not in cuts_drawn_at and window_left <= x <= window_right: 
                        self.ax.plot([x, x], ys,
                                     color=self.cut_color,
                                     linestyle='--',
                                     clip_on=False,
                                     linewidth=self.line_widths,
                                    )
                        cuts_drawn_at.add(x)

        self.draw_outcomes()

    def get_bottom_and_top(self, y, multiple=1):
        delta = StackedDiagrams.wt_height * 0.5 * multiple
        return y - delta, y + delta
    
    def draw_rect(self, source_name, x0, x1, y0, y1, alpha, color='black', fill=True, clip_to_window=True):
        window_left, window_right = self.windows[source_name]
        if clip_to_window:
            if x0 > window_right or x1 < window_left:
                return

            x0 = max(x0, window_left - 0.5)
            x1 = min(x1, window_right + 0.5)

        x0 += self.shift_x
        x1 += self.shift_x

        path = [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ]

        patch = plt.Polygon(path,
                            fill=fill,
                            closed=True,
                            alpha=alpha,
                            color=color,
                            linewidth=0 if fill else self.line_widths,
                            clip_on=False,
                           )
        self.ax.add_patch(patch)

    def draw_sequence(self, y, source_name, xs_to_skip=None, alpha=0.1):
        seq = self.seqs[source_name]
        transform_seq = self.transform_seqs[source_name]
        window_left, window_right = self.windows[source_name]

        if xs_to_skip is None:
            xs_to_skip = set()

        for x, b in zip(range(window_left, window_right + 1), seq):
            if x not in xs_to_skip:
                self.ax.annotate(transform_seq(b),
                            xy=(x, y),
                            xycoords='data', 
                            ha='center',
                            va='center',
                            size=self.text_size,
                            alpha=alpha,
                            annotation_clip=False,
                           )

    def draw_deletion(self, y, deletion, source_name, color='black', draw_MH=True, background_color='black'):
        xs_to_skip = set()

        seq = self.seqs[source_name]
        transform_seq = self.transform_seqs[source_name]
        window_left, window_right = self.windows[source_name]

        starts = np.array(deletion.starts_ats) - self.offsets[source_name]
        if draw_MH and self.draw_perfect_MH and len(starts) > 1:
            for x, b in zip(range(window_left, window_right + 1), self.seqs[source_name]):
                if (starts[0] <= x < starts[-1]) or (starts[0] + deletion.length <= x < starts[-1] + deletion.length):
                    self.ax.annotate(transform_seq(b),
                                     xy=(x, y),
                                     xycoords='data', 
                                     ha='center',
                                     va='center',
                                     size=self.text_size,
                                     color=hits.visualize.igv_colors[transform_seq(b)],
                                     weight='bold',
                                    )

                    xs_to_skip.add(x)

        if self.draw_imperfect_MH:
            before_MH = np.arange(starts.min() - 5, starts.min())
            after_MH = np.arange(starts.max(), starts.max() + 5)
            left_xs = np.concatenate((before_MH, after_MH))
            for left_x in left_xs:
                right_x = left_x + deletion.length

                # Ignore if overlaps perfect MH as a heuristic for whether interesting 
                if right_x < starts.max() or left_x >= starts.min() + deletion.length:
                    continue

                if all(0 <= x - window_left < len(seq) for x in [left_x, right_x]):
                    left_b = seq[left_x - window_left]
                    right_b = seq[right_x - window_left]
                    if left_b == right_b:
                        for x, b in ((left_x, left_b), (right_x, right_b)):
                            self.ax.annotate(transform_seq(b),
                                             xy=(x, y),
                                             xycoords='data', 
                                             ha='center',
                                             va='center',
                                             size=self.text_size,
                                             color=hits.visualize.igv_colors[b],
                                            )
                            xs_to_skip.add(x)

        if self.flip_MH_deletion_boundaries == False:
            del_start = starts[0] - 0.5
            del_end = starts[0] + deletion.length - 1 + 0.5

            for x in range(starts[0], starts[0] + deletion.length):
                xs_to_skip.add(x)
        else:
            del_start = starts[-1] - 0.5
            del_end = starts[-1] + deletion.length - 1 + 0.5

            for x in range(starts[-1], starts[-1] + deletion.length):
                xs_to_skip.add(x)
        
        bottom, top = self.get_bottom_and_top(y, multiple=StackedDiagrams.del_multiple)
        self.draw_rect(source_name, del_start, del_end, bottom, top, 0.4, color=color)

        bottom, top = self.get_bottom_and_top(y)
        self.draw_rect(source_name, window_left - 0.5, del_start, bottom, top, self.block_alpha, color=background_color)
        self.draw_rect(source_name, del_end, window_right + 0.5, bottom, top, self.block_alpha, color=background_color)

        return xs_to_skip

    def draw_insertion(self, y, insertion, source_name, draw_sequence=True):
        ti = self.target_infos[source_name]
        offset = self.offsets[source_name]
        transform_seq = self.transform_seqs[source_name]
        starts = np.array(insertion.starts_afters) - offset

        cut = ti.cut_after - offset
        if cut in starts:
            start_to_label = cut
        else:
            start_to_label = starts[0]

        purple_line_width = self.line_widths
        purple_line_alpha = 0.6
        if not self.draw_insertion_degeneracy:
            purple_line_width *= 1.5
            purple_line_alpha = 0.9

        for i, (start, bs) in enumerate(zip(starts, insertion.seqs)):
            ys = [y - 0.3, y + 0.3]
            xs = [start + 0.5, start + 0.5]

            if self.draw_insertion_degeneracy or (start == start_to_label):
                self.ax.plot(xs, ys, color='purple', linewidth=purple_line_width, alpha=purple_line_alpha, clip_on=False)

            if draw_sequence and start == start_to_label:
                width = 0.9
                center = start + 0.5
                left_edge = center - (len(bs) * 0.5 * width)
                for x_offset, b in enumerate(bs):
                    self.ax.annotate(transform_seq(b),
                                     xy=(left_edge + (x_offset * width) + width / 2, y + (StackedDiagrams.wt_height / 2)),
                                     xycoords='data',
                                     ha='center',
                                     va='center',
                                     size=self.text_size * 1,
                                     color=hits.visualize.igv_colors[transform_seq(b)],
                                     weight='bold',
                                     annotation_clip=False,
                                    )

    def draw_duplication(self, y, duplication, source_name):
        window_left, window_right = self.windows[source_name]
        offset = self.offsets[source_name]

        bottom, top = self.get_bottom_and_top(y)
        self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha)

        starts, ends = duplication.ref_junctions[0]
        starts = np.array(starts) - offset
        ends = np.array(ends) - offset
        bottom = y - 0.55 * StackedDiagrams.wt_height
        top = y + 0.55 * StackedDiagrams.wt_height

        for i, (start, end) in enumerate(zip(starts, ends)):
            if i == 0:
                alpha = 1
            else:
                alpha = 0.3

            y_offset = i * StackedDiagrams.wt_height * 0.1

            self.draw_rect(source_name, start - 0.5, end + 0.5, bottom + y_offset, top + y_offset, alpha, color='tab:purple', fill=False)

        if self.draw_all_sequence:
            self.draw_sequence(y, source_name, alpha=self.sequence_alpha)

    def draw_wild_type(self, y, source_name, on_top=False, guides_to_draw=None):
        ti = self.target_infos[source_name]
        offset = self.offsets[source_name]
        window_left, window_right = self.windows[source_name]

        bottom, top = self.get_bottom_and_top(y)

        if on_top or not self.draw_wild_type_on_top:
            self.draw_sequence(y, source_name, alpha=1)

            if guides_to_draw is None:
                guides_to_draw = [ti.primary_protospacer]

            for guide_name in guides_to_draw:
                if guide_name in ti.PAM_slices:
                    # PE3b protospacers might not exist
                    PAM_start = ti.PAM_slices[guide_name].start - 0.5 - offset
                    PAM_end = ti.PAM_slices[guide_name].stop + 0.5 - 1 - offset

                    guide = ti.features[ti.target, guide_name]
                    guide_start = guide.start - 0.5 - offset
                    guide_end = guide.end + 0.5 - offset

                    protospacer_color = self.color_overrides.get(guide_name, ti.protospacer_color)

                    PAM_name = f'{guide_name[:-len("_protospacer")]}_PAM'
                    PAM_color = self.color_overrides.get(PAM_name, ti.PAM_color)

                    self.draw_rect(source_name, guide_start, guide_end, bottom, top, None, color=protospacer_color)
                    self.draw_rect(source_name, PAM_start, PAM_end, bottom, top, None, color=PAM_color)

                    if not on_top:
                        # Draw PAMs.
                        self.draw_rect(source_name, window_left - 0.5, min(PAM_start, guide_start), bottom, top, self.block_alpha)
                        self.draw_rect(source_name, max(PAM_end, guide_end), window_right + 0.5, bottom, top, self.block_alpha)

        else:
            self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha)

        if on_top:
            for feature_name in self.features_to_draw:
                feature = ti.features[ti.target, feature_name]

                start = feature.start - 0.5 - offset
                end = feature.end + 0.5 - offset

                color = self.color_overrides.get(feature_name, feature.attribute.get('color', 'grey'))

                self.draw_rect(source_name, start, end, bottom, top, 0.8, color)
                self.ax.annotate(feature_name,
                                 xy=(np.mean([start, end]), top),
                                 xytext=(0, 5),
                                 textcoords='offset points',
                                 color='black',
                                 annotation_clip=False,
                                 ha='center',
                                 va='bottom',
                                )

    def draw_donor(self, y, HDR_outcome, deletion_outcome, insertion_outcome, source_name, on_top=False):
        ti = self.target_infos[source_name]
        transform_seq = self.transform_seqs[source_name]
        window_left, window_right = self.windows[source_name]
        SNP_ps = sorted(p for (s, p), b in ti.fingerprints[ti.target])

        bottom, top = self.get_bottom_and_top(y)

        p_to_i = SNP_ps.index
        i_to_p = dict(enumerate(SNP_ps))

        SNP_xs = set()
        observed_SNP_idxs = set()

        for ((strand, position), ref_base), read_base in zip(ti.fingerprints[ti.target], HDR_outcome.donor_SNV_read_bases):
            x = position - self.offset
            if window_left <= x <= window_right:
                # Note: read base of '-' means it was deleted
                if ref_base != read_base and read_base != '_' and read_base != '-':
                    SNP_xs.add(x)
                    observed_SNP_idxs.add(p_to_i(position))

                    self.ax.annotate(transform_seq(read_base),
                                     xy=(x + self.shift_x, y),
                                     xycoords='data', 
                                     ha='center',
                                     va='center',
                                     size=self.text_size,
                                     alpha=0.35,
                                     annotation_clip=False,
                                    )
            
                if read_base != '-':
                    if  read_base == '_':
                        color = 'grey'
                        alpha = 0.3
                    else:
                        color = hits.visualize.igv_colors[transform_seq(read_base)]
                        alpha = 0.7

                    self.draw_rect(source_name, x - 0.5, x + 0.5, bottom, top, alpha, color=color)

        if not on_top:
            # Draw rectangles around blocks of consecutive incorporated donor SNVs. 
            observed_SNP_idxs = sorted(observed_SNP_idxs)
            if observed_SNP_idxs:
                # no SNPs if just a donor deletion
                blocks = []
                block = [observed_SNP_idxs[0]]

                for i in observed_SNP_idxs[1:]:
                    if block == [] or i == block[-1] + 1:
                        block.append(i)
                    else:
                        blocks.append(block)
                        block = [i]

                blocks.append(block)

                x_buffer = 0.7
                y_multiple = 1.4

                bottom, top = self.get_bottom_and_top(y, y_multiple)

                for block in blocks:
                    start = i_to_p[block[0]] - self.offset
                    end = i_to_p[block[-1]] - self.offset
                    self.draw_rect(source_name, start - x_buffer, end + x_buffer, bottom, top, 0.5, fill=False)
        
        all_deletions = [(d, 'red', True) for d in HDR_outcome.donor_deletions]
        if deletion_outcome is not None:
            all_deletions.append((deletion_outcome.deletion, 'black', True))

        if len(all_deletions) == 0:
            self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha)
        elif len(all_deletions) == 1:
            deletion, color, draw_MH = all_deletions[0]

            if len(self.target_infos) > 1:
                background_color = ti.protospacer_color
            else:
                background_color = 'black'

            self.draw_deletion(y, deletion, source_name, color=color, draw_MH=draw_MH, background_color=background_color)

        elif len(all_deletions) > 1:
            raise NotImplementedError

        if insertion_outcome is not None:
            self.draw_insertion(y, insertion_outcome.insertion, source_name)

        if self.draw_all_sequence:
            self.draw_sequence(y, source_name, xs_to_skip=SNP_xs, alpha=self.sequence_alpha)

        if on_top:
            strands = set(SNV['strand'] for SNV in ti.donor_SNVs['donor'].values())
            if len(strands) > 1:
                raise ValueError('donor strand is weird')
            else:
                strand = strands.pop()

            arrow_ys = [y + StackedDiagrams.wt_height * 0.4, y, y - StackedDiagrams.wt_height * 0.4]

            for x in range(window_left, window_right + 1, 1):
                if x in SNP_xs:
                    continue

                if strand == '+':
                    arrow_xs = [x - 0.5, x + 0.5, x - 0.5]
                else:
                    arrow_xs = [x + 0.5, x - 0.5, x + 0.5]

                self.ax.plot(arrow_xs, arrow_ys,
                             color='black',
                             alpha=0.2,
                             clip_on=False,
                            )

    def draw_programmed_edit(self, y, programmed_edit_outcome, source_name):
        ti = self.target_infos[source_name]
        transform_seq = self.transform_seqs[source_name]
        window_left, window_right = self.windows[source_name]
        offset = self.offsets[source_name]

        bottom, top = self.get_bottom_and_top(y)

        # TODO: remove use of fingerprints here
        SNP_ps = sorted(p for (s, p), b in ti.fingerprints[ti.target])

        p_to_i = SNP_ps.index
        i_to_p = dict(enumerate(SNP_ps))

        SNP_xs = set()
        observed_SNP_idxs = set()

        if ti.pegRNA_SNVs is not None:
            SNV_names = sorted(ti.pegRNA_SNVs[ti.target])
            for SNV_name, read_base in zip(SNV_names, programmed_edit_outcome.SNV_read_bases):
                position = ti.pegRNA_SNVs[ti.target][SNV_name]['position']

                pegRNA_bases = set()
                for pegRNA_name in ti.pegRNA_names:
                    if SNV_name in ti.pegRNA_SNVs[pegRNA_name]:
                        pegRNA_base = ti.pegRNA_SNVs[pegRNA_name][SNV_name]['base']
                        if ti.pegRNA_SNVs[pegRNA_name][SNV_name]['strand'] == '-':
                            pegRNA_base = hits.utilities.reverse_complement(pegRNA_base)
                        pegRNA_bases.add(pegRNA_base)
                    
                if len(pegRNA_bases) != 1:
                    raise ValueError(SNV_name, pegRNA_bases)
                else:
                    pegRNA_base = list(pegRNA_bases)[0]

                x = position - offset
                if window_left <= x <= window_right:
                    # Note: read base of '-' means it was deleted
                    if read_base == pegRNA_base:
                        SNP_xs.add(x)
                        observed_SNP_idxs.add(p_to_i(position))

                        rect_color = hits.visualize.igv_colors[transform_seq(read_base)]
                        rect_alpha = 0.7

                        letter_color = 'black'
                        letter_alpha = 0.35
                        letter_weight = 'normal'
                    else:
                        rect_color = 'grey'
                        rect_alpha = 0.3

                        # 'get' since read_base might be '_'.
                        letter_color = hits.visualize.igv_colors.get(transform_seq(read_base))
                        letter_alpha = 1
                        letter_weight = 'bold'

                    if read_base != '-' and read_base != '_':
                        self.ax.annotate(transform_seq(read_base),
                                         xy=(x + self.shift_x, y),
                                         xycoords='data', 
                                         ha='center',
                                         va='center',
                                         size=self.text_size,
                                         alpha=letter_alpha,
                                         annotation_clip=False,
                                         color=letter_color,
                                         weight=letter_weight,
                                        )

                    self.draw_rect(source_name, x - 0.5, x + 0.5, bottom, top, rect_alpha, color=rect_color)

        # Draw rectangles around blocks of consecutive incorporated programmed SNVs. 
        observed_SNP_idxs = sorted(observed_SNP_idxs)
        if observed_SNP_idxs:
            # no SNPs if just a donor deletion
            blocks = []
            block = [observed_SNP_idxs[0]]

            for i in observed_SNP_idxs[1:]:
                if block == [] or i == block[-1] + 1:
                    block.append(i)
                else:
                    blocks.append(block)
                    block = [i]

            blocks.append(block)

            x_buffer = 0.7
            y_multiple = 1.4

            bottom, top = self.get_bottom_and_top(y, y_multiple)

            for block in blocks:
                start = i_to_p[block[0]] - offset
                end = i_to_p[block[-1]] - offset
                self.draw_rect(source_name, start - x_buffer, end + x_buffer, bottom, top, 0.5, fill=False)

        if len(programmed_edit_outcome.deletions) == 0:
            bottom, top = self.get_bottom_and_top(y)
            self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha)
        elif len(programmed_edit_outcome.deletions) == 1:
            deletion = DeletionOutcome(programmed_edit_outcome.deletions[0]).undo_anchor_shift(ti.anchor).deletion
            self.draw_deletion(y, deletion, source_name, color='black', draw_MH=False)
        else:
            raise NotImplementedError

        if len(programmed_edit_outcome.insertions) == 0:
            pass
        elif len(programmed_edit_outcome.insertions) == 1:
            insertion = InsertionOutcome(programmed_edit_outcome.insertions[0]).undo_anchor_shift(ti.anchor).insertion
            insertion = ti.expand_degenerate_indel(insertion)
            self.draw_insertion(y, insertion, source_name, draw_sequence=False)
        else:
            raise NotImplementedError
        
        if self.draw_all_sequence:
            self.draw_sequence(y, source_name, xs_to_skip=SNP_xs, alpha=self.sequence_alpha)

    def draw_pegRNA(self,
                    source_name,
                    pegRNA_name=None,
                    y_offset=1,
                    label_color=None,
                    label_features=True,
                   ):
        ti = self.target_infos[source_name]
        offset = self.offsets[source_name]

        if pegRNA_name is None:
            pegRNA_name = ti.pegRNA_names[0]
            _, _, _, _, (flap_subsequences, target_subsequences) = ti.pegRNA.extract_edits_from_alignment()
            components = ti.sgRNA_components[pegRNA_name]
        else:
            components = ti.sgRNA_components[pegRNA_name]
            flap_subsequences = [(0, len(components['RTT']))]
            target_subsequences = [(0, len(components['RTT']))]

        PBS = ti.features[ti.target, f'{pegRNA_name}_PBS']

        start = PBS.start - offset
        end = PBS.end - offset

        color = PBS.attribute['color']

        y = len(self.outcome_order) + y_offset

        bottom, top = self.get_bottom_and_top(y)
        self.draw_rect(source_name, start - 0.5, end + 0.5, bottom, top, 0.8, color)

        if label_features:
            self.ax.annotate('PBS',
                             xy=(np.mean([start, end]), top),
                             xytext=(0, 5),
                             textcoords='offset points',
                             color='black',
                             annotation_clip=False,
                             ha='center',
                             va='bottom',
                            )

        PBS_seq = hits.utilities.reverse_complement(components['PBS'])

        if self.guide[source_name].strand == '+':
            xs = end + np.arange(-len(PBS_seq) + 1, 1)
        else:
            xs = start + np.arange(len(PBS_seq) - 1, -1, -1)
            PBS_seq = hits.utilities.complement(PBS_seq)

        if self.flip[source_name]:
            if PBS.strand == '-':
                PBS_seq = hits.utilities.complement(PBS_seq)
            else:
                PBS_seq = PBS_seq[::-1]
        else:
            if PBS.strand == '+':
                PBS_seq = hits.utilities.complement(PBS_seq)
            else:
                PBS_seq = PBS_seq[::-1]

        for x, b in zip(xs, PBS_seq):
            self.ax.annotate(b,
                             xy=(x, y),
                             xycoords='data', 
                             ha='center',
                             va='center',
                             size=self.text_size,
                             annotation_clip=False,
                            )

        RTT_xs = []
        RTT_rc = hits.utilities.reverse_complement(components['RTT'])
        RTT_aligned_seq = ''

        RTT_offset = ti.cut_afters[f'{pegRNA_name}_protospacer_{PBS.strand}'] - offset

        for (target_start, target_end), (flap_start, flap_end) in zip(target_subsequences, flap_subsequences):
            # target_subsequences are in downstream_of_nick coords and end is exclusive.
            # 0 is cut_after

            RTT_subsequence = RTT_rc[flap_start:flap_end]
            if PBS.strand == '-':
                RTT_subsequence = hits.utilities.complement(RTT_subsequence)
            RTT_aligned_seq += RTT_subsequence

            if PBS.strand == '+':
                xs_start = target_start + 1
                xs_end = target_end + 1
                step = 1
            else:
                xs_start = -target_start
                xs_end = -target_end
                step = -1

            RTT_xs.extend(RTT_offset + np.arange(xs_start, xs_end, step))

            rect_start, rect_end = sorted([xs_start, xs_end])

            rect_start = RTT_offset + rect_start - (0.5 * step)
            rect_end = RTT_offset + rect_end - (0.5 * step)

            color = knock_knock.pegRNAs.default_feature_colors['RTT']

            self.draw_rect(source_name, rect_start, rect_end, bottom, top, 0.8, color)

        del_bottom, del_top = self.get_bottom_and_top(y, StackedDiagrams.del_multiple)
        for (_, previous_end), (next_start, _) in zip(target_subsequences, target_subsequences[1:]):
            self.draw_rect(source_name, previous_end + 1 - 0.5, next_start + 1 - 0.5, del_bottom, del_top, 0.4, color='black')

        seq = self.seqs[source_name]
        window_left, window_right = self.windows[source_name]
        for x, b in zip(RTT_xs, RTT_aligned_seq):
            if window_left <= x <= window_right:
                target_b = seq[-window_left + x]

                if self.flip[source_name]:
                    b_to_draw = hits.utilities.complement(b)
                else:
                    b_to_draw = b

                if b != target_b:
                    color = hits.visualize.igv_colors[b_to_draw]
                    weight = 'bold'
                else:
                    color = 'black'
                    weight = 'normal'

                self.ax.annotate(b_to_draw,
                                 xy=(x, y),
                                 xycoords='data', 
                                 ha='center',
                                 va='center',
                                 size=self.text_size,
                                 annotation_clip=False,
                                 color=color,
                                 weight=weight,
                                )

        if ti.pegRNA_programmed_insertion is not None:
            self.draw_insertion(y, ti.pegRNA_programmed_insertion, source_name)

        if label_features:
            self.ax.annotate('RTT',
                             xy=(np.mean(RTT_xs), top),
                             xytext=(0, 5),
                             textcoords='offset points',
                             color='black',
                             annotation_clip=False,
                             ha='center',
                             va='bottom',
                            )

        if label_color is not None:
            window_left, window_right = self.windows[source_name]
            self.draw_rect(source_name, window_left - 2, window_left - 1, y - 0.4, y + 0.4, 1, clip_to_window=False, color=label_color)

    def draw_outcomes(self):
        for i, (source_name, category, subcategory, details) in enumerate(self.outcome_order):
            ti = self.target_infos[source_name]
            transform_seq = self.transform_seqs[source_name]
            window_left, window_right = self.windows[source_name]
            offset = self.offsets[source_name]
            y = self.num_outcomes - i - 1

            bottom, top = self.get_bottom_and_top(y)

            if len(self.target_infos) > 1:
                background_color = ti.protospacer_color
            else:
                background_color = 'black'
                
            if (category == 'deletion') or \
               (category == 'simple indel' and subcategory.startswith('deletion')) or \
               (category == 'wild type' and subcategory == 'short indel far from cut' and degenerate_indel_from_string(details).kind == 'D'):

                deletion = DeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor).deletion
                deletion = ti.expand_degenerate_indel(deletion)

                xs_to_skip = self.draw_deletion(y, deletion, source_name, background_color=background_color)
                if self.draw_all_sequence:
                    self.draw_sequence(y, source_name, xs_to_skip, alpha=self.sequence_alpha)
            
            elif category == 'insertion' or (category == 'simple indel' and subcategory.startswith('insertion')):
                insertion = InsertionOutcome.from_string(details).undo_anchor_shift(ti.anchor).insertion
                insertion = ti.expand_degenerate_indel(insertion)

                self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha, color=background_color)
                self.draw_insertion(y, insertion, source_name)

                if self.draw_all_sequence:
                    self.draw_sequence(y, source_name, alpha=self.sequence_alpha)

            elif category == 'insertion with deletion':
                outcome = InsertionWithDeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                insertion = outcome.insertion_outcome.insertion
                deletion = outcome.deletion_outcome.deletion
                self.draw_insertion(y, insertion, source_name)
                self.draw_deletion(y, deletion, source_name, background_color=background_color)
                    
            elif category == 'mismatches' or (category == 'wild type' and subcategory == 'mismatches'):
                SNV_xs = set()
                self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha)

                if details == 'n/a':
                    snvs = []
                else:
                    snvs = SNVs.from_string(details) 

                # Undo anchor shift.
                snvs = SNVs([SNV(s.position + ti.anchor, s.basecall, s.quality) for s in snvs])

                for snv in snvs:
                    x = snv.position - offset
                    SNV_xs.add(x)
                    if window_left <= x <= window_right:
                        self.ax.annotate(transform_seq(snv.basecall),
                                         xy=(x, y),
                                         xycoords='data', 
                                         ha='center',
                                         va='center',
                                         size=self.text_size,
                                         color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                         weight='bold',
                                        )
                
                for (strand, position), ref_base in ti.fingerprints[ti.target]:
                    color = 'grey'
                    alpha = 0.3
                    left = position - offset - 0.5
                    right = position - offset + 0.5
                    self.draw_rect(source_name, left, right, bottom, top, alpha, color=color)

                if self.draw_all_sequence:
                    self.draw_sequence(y, source_name, xs_to_skip=SNV_xs, alpha=self.sequence_alpha)

            elif category == 'wild type' or category == 'WT':
                self.draw_wild_type(y, source_name)

            elif category == 'deletion + adjacent mismatch' or category == 'deletion + mismatches':
                outcome = DeletionPlusMismatchOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                xs_to_skip = self.draw_deletion(y, outcome.deletion_outcome.deletion, draw_MH=True)
                
                for snv in outcome.mismatch_outcome.snvs:
                    x = snv.position - self.offsets
                    xs_to_skip.add(x)

                    if window_left <= x <= window_right:
                        self.ax.annotate(transform_seq(snv.basecall),
                                         xy=(x, y),
                                         xycoords='data', 
                                         ha='center',
                                         va='center',
                                         size=self.text_size,
                                         color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                         weight='bold',
                                        )

                    if category == 'deletion + adjacent mismatch':
                        # Draw box around mismatch to distinguish from MH.
                        x_buffer = 0.7
                        y_multiple = 1.4
                        box_bottom, box_top = self.get_bottom_and_top(y, y_multiple)
                        self.draw_rect(source_name, x - x_buffer, x + x_buffer, box_bottom, box_top, 0.5, fill=False)

                if self.draw_all_sequence:
                    self.draw_sequence(y, source_name, xs_to_skip, alpha=self.sequence_alpha)

            elif category == 'donor' or \
                category == 'donor + deletion' or \
                category == 'donor + insertion' or \
                (category == 'intended edit' and subcategory == 'deletion') or \
                category == 'edit + deletion':

                if category == 'donor':
                    HDR_outcome = HDROutcome.from_string(details)
                    deletion_outcome = None
                    insertion_outcome = None

                elif category == 'donor + deletion':
                    HDR_plus_deletion_outcome = HDRPlusDeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                    HDR_outcome = HDR_plus_deletion_outcome.HDR_outcome
                    deletion_outcome = HDR_plus_deletion_outcome.deletion_outcome
                    insertion_outcome = None

                elif category == 'intended edit' and subcategory == 'deletion':
                    HDR_outcome = HDROutcome.from_string(details).undo_anchor_shift(ti.anchor)
                    deletion_outcome = None
                    insertion_outcome = None

                elif category == 'donor + insertion':
                    HDR_plus_insertion_outcome = HDRPlusInsertionOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                    HDR_outcome = HDR_plus_insertion_outcome.HDR_outcome
                    deletion_outcome = None
                    insertion_outcome = HDR_plus_insertion_outcome.insertion_outcome

                elif category == 'edit + deletion':
                    HDR_plus_deletion_outcome = HDRPlusDeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                    HDR_outcome = HDR_plus_deletion_outcome.HDR_outcome
                    deletion_outcome = HDR_plus_deletion_outcome.deletion_outcome
                    insertion_outcome = None

                else:
                    raise ValueError
        
                self.draw_donor(y, HDR_outcome, deletion_outcome, insertion_outcome, source_name, False)

            elif category in ['intended edit', 'partial replacement']:
                self.draw_programmed_edit(y, ProgrammedEditOutcome.from_string(details), source_name)

            elif category == 'duplication' and subcategory == 'simple':
                duplication_outcome = DuplicationOutcome.from_string(details).undo_anchor_shift(ti.anchor)
                self.draw_duplication(y, duplication_outcome, source_name)
                
            else:
                label = f'{category}, {subcategory}, {details}'

                label = self.replacement_text_for_complex.get(label, label)

                self.ax.annotate(label,
                                 xy=(0, y),
                                 xycoords=('axes fraction', 'data'), 
                                 xytext=(5, 0),
                                 textcoords='offset points',
                                 ha='left',
                                 va='center',
                                 size=self.text_size,
                                )
                if len(self.target_infos) > 1:
                    self.draw_rect(source_name, window_left - 0.5, window_right + 0.5, bottom, top, self.block_alpha, color=background_color)

        if self.draw_donor_on_top and len(ti.donor_SNVs['target']) > 0:
            donor_SNV_read_bases = ''.join(d['base'] for name, d in sorted(ti.donor_SNVs['donor'].items()))
            strands = set(SNV['strand'] for SNV in ti.donor_SNVs['donor'].values())
            if len(strands) > 1:
                raise ValueError('donor strand is weird')
            else:
                strand = strands.pop()

            HDR_outcome = HDROutcome(donor_SNV_read_bases, [])

            if strand == '-':
                y = self.num_outcomes + 0.5
            else:
                y = self.num_outcomes + 2.5

            self.draw_donor(y, HDR_outcome, None, None, self.single_source_name, on_top=True)

        if self.draw_wild_type_on_top:
            y = self.num_outcomes
            if self.draw_donor_on_top:
                y += 1.5

            
            self.draw_wild_type(y, self.single_source_name,
                                on_top=True,
                                guides_to_draw=self.target_infos[self.single_source_name].protospacer_names,
                               )
            self.ax.set_xticks([])
                    
        if not self.preserve_x_lims:
            # Some uses don't want x lims to be changed.
            x_lims = [window_left - 0.5, window_right + 0.5]
            if self.flip[self.single_source_name]:
                self.ax.set_xlim(*x_lims[::-1])
            else:
                self.ax.set_xlim(*x_lims)

            self.ax.xaxis.tick_top()
            self.ax.axhline(self.num_outcomes + 0.5 - 1, color='black', alpha=0.75, clip_on=False)

        self.ax.set_ylim(-0.5, self.num_outcomes - 0.5)
        self.ax.set_frame_on(False)

        if self.title is not None:
            self.ax.annotate(self.title,
                             xy=(0.5, 1),
                             xycoords=('axes fraction', 'axes fraction'),
                             xytext=(0, self.title_offset),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             size=self.title_size,
                             color=self.title_color,
                            )
            
        self.ax.set_yticks([])
        
class DiagramGrid:
    def __init__(self,
                 outcomes,
                 target_info,
                 inches_per_nt=0.12,
                 inches_per_outcome=0.25,
                 outcome_ax_width=3,
                 diagram_ax=None,
                 title=None,
                 label_aliases=None,
                 cut_color=None,
                 ax_on_bottom=False,
                 **diagram_kwargs,
                ):

        self.outcomes = outcomes
        self.target_info = target_info
        #PAM_feature = target_info.PAM_features[f'{target_info.primary_protospacer}_PAM']
        
        #self.PAM_color = PAM_feature.attribute['color']
        self.ax_on_bottom = ax_on_bottom

        #if cut_color == 'PAM':
        #    diagram_kwargs['cut_color'] = self.PAM_color

        self.inches_per_nt = inches_per_nt
        self.inches_per_outcome = inches_per_outcome
        self.outcome_ax_width = outcome_ax_width

        self.fig = None
        self.title = title

        self.ordered_axs = []
        self.ordered_axs_above = []

        self.axs_by_name = {
            'diagram': diagram_ax,
        }

        self.ims = []
        self.widths = {}

        self.diagram_kwargs = diagram_kwargs

        if label_aliases is None:
            label_aliases = {}
        self.label_aliases = label_aliases

        self.plot_diagrams()

    def plot_diagrams(self):
        self.diagrams = StackedDiagrams(self.outcomes,
                                        self.target_info,
                                        ax=self.axs_by_name['diagram'],
                                        title=self.title,
                                        inches_per_outcome=self.inches_per_outcome,
                                        inches_per_nt=self.inches_per_nt,
                                        **self.diagram_kwargs,
                                       )

        self.fig = self.diagrams.fig

        self.axs_by_name['diagram'] = self.diagrams.ax
        self.ordered_axs.append(self.diagrams.ax)
        self.ordered_axs_above.append(self.diagrams.ax)

        ax_p = self.diagrams.ax.get_position()
        self.widths['diagram'] = ax_p.width

        return self.fig

    def add_ax(self, name, width_multiple=10, gap_multiple=1, title='', side='right', title_size=12):
        width = self.width_per_heatmap_cell * width_multiple

        if side == 'right':
            ax_p = self.ordered_axs[-1].get_position()
            x0 = ax_p.x1 + self.width_per_heatmap_cell * gap_multiple
        else:
            ax_p = self.ordered_axs[0].get_position()
            x0 = ax_p.x0 - width - self.width_per_heatmap_cell * gap_multiple

        height = ax_p.height
        y0 = ax_p.y0

        ax = self.fig.add_axes((x0, y0, width, height), sharey=self.axs_by_name['diagram'])
        self.axs_by_name[name] = ax

        if side == 'right':
            self.ordered_axs.append(ax)
        else:
            self.ordered_axs.insert(0, ax)

        ax.set_yticks([])
        ax.xaxis.tick_top()
        ax.spines['left'].set_alpha(0.3)
        ax.spines['right'].set_alpha(0.3)
        ax.tick_params(labelsize=6)
        ax.grid(axis='x', alpha=0.3, clip_on=False)
        
        if self.ax_on_bottom:
            ax.spines['top'].set_visible(False)
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position('bottom')
            title_kwargs = dict(
                xy=(0.5, 0),
                xytext=(0, -20),
                va='top',
            )
        else:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_label_position('top')
            title_kwargs = dict(
                xy=(0.5, 1),
                xytext=(0, 20),
                va='bottom',
            )

        if title != '':
            ax.annotate(title,
                        xycoords='axes fraction',
                        textcoords='offset points',
                        ha='center',
                        size=title_size,
                        **title_kwargs,
                       )

        return ax

    def add_ax_above(self, name, height_multiple=10, gap=2):
        ax_p = self.ordered_axs_above[-1].get_position()
        x0 = ax_p.x0 

        width = ax_p.width
        height = self.height_per_heatmap_cell * height_multiple
        
        y0 = ax_p.y1 + self.height_per_heatmap_cell * gap

        ax = self.fig.add_axes((x0, y0, width, height), sharex=self.axs_by_name['diagram'])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        self.axs_by_name[name] = ax

        self.ordered_axs_above.append(ax)

    def plot_on_ax(self,
                   ax_name,
                   value_source,
                   interval_sources=None,
                   transform=None,
                   y_offset=0,
                   interval_alpha=1,
                   marker_alpha=1,
                   line_alpha=1,
                   label='',
                   marker='.',
                   fill=np.nan,
                   **plot_kwargs,
                  ):
        # To simplify logic of excluding panels, do nothing if ax_name is not an existing ax.
        ax = self.axs_by_name.get(ax_name)
        if ax is None:
            return

        if transform == 'percentage':
            transform = lambda f: 100 * f
        elif transform == 'log10':
            transform = np.log10

        ys = np.arange(len(self.outcomes) - 1, -1, -1) + y_offset

        if transform is not None:
            # suppress warnings from log of zeros
            # Using the warnings context manager doesn't work here, maybe because of pandas multithreading.
            warnings.filterwarnings('ignore')

            value_source = transform(value_source)

            if interval_sources is not None:
                interval_sources = {k: transform(v) for k, v in interval_sources.items()}

            warnings.resetwarnings()

        if len(self.outcomes[0]) == 3:
            xs = [value_source.get((c, s, d), fill) for c, s, d in self.outcomes]
        else:
            xs = [value_source.get((c, s, d), fill) for source_name, c, s, d in self.outcomes]

        ax.plot(xs, ys, marker=marker, linestyle='', alpha=marker_alpha, label=label, **plot_kwargs)
        ax.plot(xs, ys, marker=None, linestyle='-', alpha=line_alpha, **plot_kwargs)

        if interval_sources is not None:
            interval_xs = {
                side: [interval_sources[side].get(outcome, np.nan) for outcome in self.outcomes]
                for side in ['lower', 'upper']
            }

            for y, lower_x, upper_x in zip(ys, interval_xs['lower'], interval_xs['upper']):
                ax.plot([lower_x, upper_x], [y, y],
                        color=plot_kwargs.get('color'),
                        alpha=interval_alpha,
                        clip_on=plot_kwargs.get('clip_on', True),
                       )

    def plot_on_ax_above(self,
                         ax_name,
                         xs,
                         value_source,
                         marker_alpha=1,
                         line_alpha=0.75,
                         label='',
                         marker='.',
                         **plot_kwargs,
                        ):
        plot_kwargs = copy.copy(plot_kwargs)
        plot_kwargs.setdefault('linewidth', 1.5)
        plot_kwargs.setdefault('markersize', 7)
        plot_kwargs.setdefault('color', 'black')

        ax = self.axs_by_name[ax_name]

        ys = [value_source[x] for x in xs]

        ax.plot(xs, ys, marker=marker, linestyle='', alpha=marker_alpha, label=label, **plot_kwargs)
        ax.plot(xs, ys, marker=None, linestyle='-', alpha=line_alpha, **plot_kwargs)

    def style_frequency_ax(self, name, manual_ticks=None, label_size=8, include_percentage=False):
        ax = self.axs_by_name.get(name)
        if ax is None:
            return

        ax.tick_params(labelsize=label_size)

    def style_log10_frequency_ax(self, name, manual_ticks=None, label_size=8, include_percentage=False):
        # To simplify logic of excluding panels, do nothing if name is not an existing ax.
        ax = self.axs_by_name.get(name)
        if ax is None:
            return

        x_min, x_max = ax.get_xlim()

        x_ticks = []

        for exponent in [6, 5, 4, 3, 2, 1, 0]:
            xs = np.log10(np.arange(1, 10) * 10**-exponent)        
            for x in xs:
                if x_min < x < x_max:
                    ax.axvline(x, color='black', alpha=0.05, clip_on=False)

            if exponent <= 3:
                multiples = [1, 5]
            else:
                multiples = [1]

            for multiple in multiples:
                x = multiple * 10**-exponent
                if x_min <= np.log10(x) <= x_max:
                    x_ticks.append(x)

        if manual_ticks is not None:
            x_ticks = manual_ticks

        ax.set_xticks(np.log10(x_ticks))
        ax.set_xticklabels([f'{100 * x:g}' + ('%' if include_percentage else '') for x in x_ticks], size=label_size)

        for side in ['left', 'right']:
            ax.spines[side].set_visible(False)

    def style_fold_change_ax(self, name, label_size=8):
        # To simplify logic of excluding panels, do nothing if name is not an existing ax.
        ax = self.axs_by_name.get(name)
        if ax is None:
            return

        for side in ['left', 'right']:
            ax.spines[side].set_visible(False)

        ax.axvline(0, color='black', alpha=0.5, clip_on=False)

        plt.setp(ax.get_xticklabels(), size=label_size)

    @property
    def width_per_heatmap_cell(self):
        fig_width_inches, fig_height_inches = self.fig.get_size_inches()
        diagram_position = self.axs_by_name['diagram'].get_position()
        width_per_cell = diagram_position.height * 1 / len(self.outcomes) * fig_height_inches / fig_width_inches
        return width_per_cell

    @property
    def height_per_heatmap_cell(self):
        diagram_position = self.axs_by_name['diagram'].get_position()
        height_per_cell = diagram_position.height * 1 / len(self.outcomes)
        return height_per_cell

    def add_heatmap(self, vals, name,
                    gap_multiple=1,
                    color='black',
                    colors=None,
                    vmin=-2, vmax=2,
                    cmap=knock_knock.visualize.fold_changes_cmap,
                    draw_tick_labels=True,
                    text_size=10,
                    tick_label_rotation=90,
                    tick_label_pad=8,
                   ):
        ax_to_left = self.ordered_axs[-1]
        ax_to_left_p = ax_to_left.get_position()

        num_rows, num_cols = vals.shape

        heatmap_height = ax_to_left_p.height
        heatmap_width = self.width_per_heatmap_cell * num_cols

        gap = self.width_per_heatmap_cell * gap_multiple
        heatmap_left = ax_to_left_p.x1 + gap

        rect = [heatmap_left, ax_to_left_p.y0, heatmap_width, heatmap_height]
        ax = self.fig.add_axes(rect, sharey=ax_to_left)

        im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax)
        self.ims.append(im)
        plt.setp(ax.spines.values(), visible=False)

        if draw_tick_labels:
            if self.ax_on_bottom:
                ax.xaxis.tick_bottom()
            else:
                ax.xaxis.tick_top()

            ax.set_xticks(np.arange(num_cols))
            ax.set_xticklabels([])

            tick_labels = vals.columns.values
            if isinstance(tick_labels[0], tuple):
                tick_labels = [', '.join(map(str, l)) for l in tick_labels]

            if colors is None:
                colors = [color for _ in range(len(tick_labels))]

            for x, (label, color) in enumerate(zip(tick_labels, colors)):
                if self.ax_on_bottom:
                    label_kwargs = dict(
                        xy=(x, 0),
                        xytext=(0, -tick_label_pad),
                        va='top',
                    )
                else:
                    label_kwargs = dict(
                        xy=(x, 1),
                        xytext=(0, tick_label_pad),
                        va='bottom',
                    )
                ax.annotate(label,
                            xycoords=('data', 'axes fraction'),
                            textcoords='offset points',
                            rotation=tick_label_rotation,
                            ha='center',
                            color=color,
                            size=text_size,
                            **label_kwargs,
                           )

        else:
            ax.set_xticks([])

        self.axs_by_name[name] = ax
        self.ordered_axs.append(ax)

        return ax

    def add_colorbar(self,
                     width_multiple=5,
                     baseline_condition_name='non-targeting',
                     label_interpretation=True,
                     loc='right',
                     **kwargs,
                    ):
        if len(self.ims) == 0:
            return

        import repair_seq.visualize.heatmap

        ax_p = self.ordered_axs[-1].get_position()

        if loc == 'right':
            x0 = ax_p.x1 + 5 * self.width_per_heatmap_cell
            y0 = 0.5
        else:
            raise NotImplementedError

        width = width_multiple * self.width_per_heatmap_cell
        height = 1 * self.height_per_heatmap_cell
        repair_seq.visualize.heatmap.add_fold_change_colorbar(self.fig, self.ims[0], x0, y0, width, height,
                                                              baseline_condition_name=baseline_condition_name,
                                                              label_interpretation=label_interpretation,
                                                              **kwargs,
                                                             )

    def mark_subset(self, outcomes_to_mark, color,
                    title='',
                    width_multiple=1,
                    gap_multiple=0.5,
                    side='left',
                    size=12,
                    linewidth=5,
                   ):

        ax = self.add_ax(side=side, name='subset', width_multiple=width_multiple, gap_multiple=gap_multiple)
        outcomes = list(self.outcomes)
        outcomes_to_mark = [outcome for outcome in outcomes_to_mark if outcome in outcomes]
        # Note weird flipping
        indices = [len(outcomes) - 1 - outcomes.index(outcome) for outcome in outcomes_to_mark]
        for idx in indices:
            ax.plot([0.5, 0.5], [idx - 0.5, idx + 0.5],
                    linewidth=linewidth,
                    color=color,
                    solid_capstyle='butt',
                   )

        ax.axis('off')

        ax.annotate(title,
                    xy=(0.5, 1),
                    xycoords='axes fraction',
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color=color,
                    size=size,
                   )

    def draw_outcome_categories(self, effector='Cas9'):
        ax = self.add_ax(side='left', name='outcome_categories', width_multiple=1, gap_multiple=0.5)

        full_categories = defaultdict(list)

        for c, s, d in self.outcomes:
            if c == 'deletion':
                deletion = DeletionOutcome.from_string(d).undo_anchor_shift(self.target_info.anchor)
                directionality = deletion.classify_directionality(self.target_info)
                full_category = f'{c}, {directionality}'
            else:
                full_category = c
                
            full_categories[full_category].append((c, s, d))

        x = 0

        import repair_seq.visualize

        for cat in repair_seq.visualize.Cas9_category_display_order:
            cat_outcomes = full_categories[cat]
            if len(cat_outcomes) > 0:
                indices = sorted([len(self.outcomes) - 1 - self.outcomes.index(outcome) for outcome in cat_outcomes])
                connected_blocks = []
                current_block_start = indices[0]
                current_idx = indices[0]

                for next_idx in indices[1:]:
                    if next_idx - current_idx > 1:
                        block = (current_block_start, current_idx)
                        connected_blocks.append(block)
                        current_block_start = next_idx
                        
                    current_idx = next_idx

                # Close off last block
                block = (current_block_start, current_idx)
                connected_blocks.append(block) 

                for first, last in connected_blocks:
                    ax.plot([x, x], [first - 0.4, last + 0.4],
                            linewidth=2,
                            color=repair_seq.visualize.Cas9_category_colors[cat],
                            clip_on=False,
                            solid_capstyle='butt',
                           )

                x -= 1

        ax.set_xlim(x - 1, 1)
        #ax.set_ylim(-0.5, len(outcomes) - 0.5)
        ax.axis('off')

    def set_xlim(self, ax_name, lims):
        if lims is not None:
            ax = self.axs_by_name.get(ax_name)
            if ax is not None:
                ax.set_xlim(*lims)

    def plot_pegRNA_conversion_fractions_above(self, group, gap=4, height_multiple=10, conditions=None, **plot_kwargs):
        plot_kwargs = copy.copy(plot_kwargs)
        plot_kwargs.setdefault('line_alpha', 0.75)
        plot_kwargs.setdefault('linewidth', 1.5)
        plot_kwargs.setdefault('markersize', 7)
        plot_kwargs.setdefault('color', 'black')

        def SNV_name_to_x(SNV_name):
            SNVs = group.target_info.pegRNA_SNVs[group.target_info.target]
            p = SNVs[SNV_name]['position']
            x = p - group.target_info.cut_after
            return x
        
        pegRNA_conversion_fractions = group.pegRNA_conversion_fractions.copy()

        xs = pegRNA_conversion_fractions.index.map(SNV_name_to_x)
        pegRNA_conversion_fractions.index = xs
        pegRNA_conversion_fractions = pegRNA_conversion_fractions.sort_index()
        
        if 'pegRNA_conversion_fractions' not in self.axs_by_name:
            self.add_ax_above('pegRNA_conversion_fractions', gap=gap, height_multiple=height_multiple)
        
        for condition, fs in pegRNA_conversion_fractions.items():
            if conditions is not None and condition not in conditions:
                continue

            self.plot_on_ax_above('pegRNA_conversion_fractions',
                                  xs,
                                  fs * 100,
                                  label=condition,
                                  **plot_kwargs,
                                 )

    def style_pegRNA_conversion_plot(self, ax_name, y_max=None):
        if 'diagram' not in self.axs_by_name or ax_name not in self.axs_by_name:
            return

        diagram_x_lims = self.axs_by_name['diagram'].get_xlim()

        flipped = diagram_x_lims[0] > diagram_x_lims[1]

        ax = self.axs_by_name[ax_name]

        if y_max is None:
            ax.autoscale(axis='y')

        ax.set_ylim(0, y_max)

        xs = set()

        for line in ax.lines:
            xs.update(set(line.get_xdata()))
            
        x_bounds = [min(xs) - 1, max(xs) + 1]
        if flipped:
            x_bounds = x_bounds[::-1]

        for y in ax.get_yticks():
            if y == 0:
                alpha = 1
                clip_on = False
            else:
                alpha = 0.3
                clip_on = True

            ax.plot(x_bounds, [y for x in x_bounds], linewidth=0.5, clip_on=clip_on, color='black', alpha=alpha)

        ax.set_ylabel('Total %\nincorporation\nat position', size=12)
        ax.tick_params(labelsize=8)

        ax.spines.left.set_position(('data', x_bounds[0]))
        ax.spines.bottom.set_visible(False)