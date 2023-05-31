import warnings

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hits.utilities
import hits.visualize

import knock_knock.pegRNAs
from knock_knock.target_info import degenerate_indel_from_string, SNV, SNVs
from knock_knock.outcome import *

def plot(outcome_order,
         target_infos,
         num_outcomes=None,
         title=None,
         window=70,
         ax=None,
         flip_if_reverse=True,
         flip_MH_deletion_boundaries=False,
         force_flip=False,
         center_at_PAM=False,
         draw_cut_afters=True,
         inches_per_nt=0.12,
         inches_per_outcome=0.25,
         line_widths=1.5,
         draw_all_sequence=False,
         draw_perfect_MH=True,
         draw_imperfect_MH=False,
         draw_wild_type_on_top=False,
         draw_donor_on_top=False,
         draw_insertion_degeneracy=True,
         text_size=8,
         features_to_draw=None,
         replacement_text_for_complex=None,
         cut_color=hits.visualize.apply_alpha('black', 0.5),
         preserve_x_lims=False,
         shift_x=0,
         block_alpha=0.1,
         override_PAM_color=None,
         override_protospacer_color=None,
         **kwargs,
        ):

    # Can either supply outcomes as (c, s, d) tuples along with a single target_info,
    # or outcomes as (source_name, c, s, d) tuples along with a {source_name: target_info} dict.

    if all(len(outcome) == 3 for outcome in outcome_order):
        target_infos = {'single_source': target_infos}
        outcome_order = [('single_source', c, s, d) for c, s, d in outcome_order]

    single_source_name = sorted(target_infos)[0]

    if isinstance(window, int):
        window_left, window_right = -window, window
    else:
        window_left, window_right = window

    if features_to_draw is None:
        features_to_draw = []

    window_size = window_right - window_left + 1

    if num_outcomes is None:
        num_outcomes = len(outcome_order)

    outcome_order = outcome_order[:num_outcomes]

    if ax is None:
        fig, ax = plt.subplots(figsize=(inches_per_nt * window_size, inches_per_outcome * num_outcomes))
    else:
        fig = ax.figure

    if isinstance(draw_all_sequence, float):
        sequence_alpha = draw_all_sequence
    else:
        sequence_alpha = 0.1
        
    del_height = 0.15
        
    offsets = {}
    seqs = {}
    transform_seqs = {}
    windows = {}

    for source_name, ti in target_infos.items():
        guide = ti.features[ti.target, ti.primary_protospacer]

        # TODO: flip behavior for multiple sources
        if force_flip or (flip_if_reverse and guide.strand == '-'):
            flip = True
            transform_seq = hits.utilities.complement
        else:
            flip = False
            transform_seq = hits.utilities.identity

        if center_at_PAM:
            if guide.strand == '+':
                offset = ti.PAM_slice.start
            else:
                offset = ti.PAM_slice.stop - 1
        else:
            offset = max(v for n, v in ti.cut_afters.items() if n.startswith(ti.primary_protospacer))

        offsets[source_name] = offset

        if flip:
            this_window_left, this_window_right = -window_right, -window_left
        else:
            this_window_left, this_window_right = window_left, window_right

        windows[source_name] = (this_window_left, this_window_right)

        seq = ti.target_sequence[offset + this_window_left:offset + this_window_right + 1]
        seqs[source_name] = seq
        transform_seqs[source_name] = transform_seq

    cuts_drawn_at = set()
    if draw_cut_afters:
        for source_name, ti in target_infos.items():
            offset = offsets[source_name]
            window_left, window_right = windows[source_name]

            for cut_after in ti.cut_afters.values():
                # temp fix for empirically-determined offset of Cpf1 cut.
                x = (cut_after + 0.5) - offset + shift_x

                if draw_wild_type_on_top and not draw_donor_on_top:
                    ys = [-0.5, num_outcomes + 0.5]
                else:
                    ys = [-0.5, num_outcomes - 0.5]

                if x not in cuts_drawn_at and window_left <= x <= window_right: 
                    ax.plot([x, x], ys,
                            color=cut_color,
                            linestyle='--',
                            clip_on=False,
                            linewidth=line_widths,
                        )
                    cuts_drawn_at.add(x)
    
    def draw_rect(source_name, x0, x1, y0, y1, alpha, color='black', fill=True):
        window_left, window_right = windows[source_name]
        if x0 > window_right or x1 < window_left:
            return

        x0 = max(x0, window_left - 0.5) + shift_x
        x1 = min(x1, window_right + 0.5) + shift_x

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
                            linewidth=0 if fill else line_widths,
                            clip_on=False,
                           )
        ax.add_patch(patch)

    wt_height = 0.6

    def draw_sequence(y, source_name, xs_to_skip=None, alpha=0.1):
        seq = seqs[source_name]
        transform_seq = transform_seqs[source_name]
        window_left, window_right = windows[source_name]

        if xs_to_skip is None:
            xs_to_skip = set()

        for x, b in zip(range(window_left, window_right + 1), seq):
            if x not in xs_to_skip:
                ax.annotate(transform_seq(b),
                            xy=(x, y),
                            xycoords='data', 
                            ha='center',
                            va='center',
                            size=text_size,
                            alpha=alpha,
                            annotation_clip=False,
                           )

    def draw_deletion(y, deletion, source_name, color='black', draw_MH=True, background_color='black'):
        xs_to_skip = set()

        seq = seqs[source_name]
        transform_seq = transform_seqs[source_name]
        window_left, window_right = windows[source_name]

        starts = np.array(deletion.starts_ats) - offsets[source_name]
        if draw_MH and draw_perfect_MH and len(starts) > 1:
            for x, b in zip(range(window_left, window_right + 1), seqs[source_name]):
                if (starts[0] <= x < starts[-1]) or (starts[0] + deletion.length <= x < starts[-1] + deletion.length):
                    ax.annotate(transform_seq(b),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=hits.visualize.igv_colors[transform_seq(b)],
                                weight='bold',
                               )

                    xs_to_skip.add(x)

        if draw_imperfect_MH:
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
                            ax.annotate(transform_seq(b),
                                        xy=(x, y),
                                        xycoords='data', 
                                        ha='center',
                                        va='center',
                                        size=text_size,
                                        color=hits.visualize.igv_colors[b],
                                    )
                            xs_to_skip.add(x)

        if flip_MH_deletion_boundaries == False:
            del_start = starts[0] - 0.5
            del_end = starts[0] + deletion.length - 1 + 0.5

            for x in range(starts[0], starts[0] + deletion.length):
                xs_to_skip.add(x)
        else:
            del_start = starts[-1] - 0.5
            del_end = starts[-1] + deletion.length - 1 + 0.5

            for x in range(starts[-1], starts[-1] + deletion.length):
                xs_to_skip.add(x)
        
        draw_rect(source_name, del_start, del_end, y - del_height / 2, y + del_height / 2, 0.4, color=color)
        draw_rect(source_name, window_left - 0.5, del_start, y - wt_height / 2, y + wt_height / 2, block_alpha, color=background_color)
        draw_rect(source_name, del_end, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha, color=background_color)

        return xs_to_skip

    def draw_insertion(y, insertion, source_name, draw_sequence=True):
        ti = target_infos[source_name]
        offset = offsets[source_name]
        transform_seq = transform_seqs[source_name]
        starts = np.array(insertion.starts_afters) - offset

        cut = ti.cut_after - offset
        if cut in starts:
            start_to_label = cut
        else:
            start_to_label = starts[0]

        purple_line_width = line_widths
        purple_line_alpha = 0.6
        if not draw_insertion_degeneracy:
            purple_line_width *= 1.5
            purple_line_alpha = 0.9

        for i, (start, bs) in enumerate(zip(starts, insertion.seqs)):
            ys = [y - 0.3, y + 0.3]
            xs = [start + 0.5, start + 0.5]

            if draw_insertion_degeneracy or (start == start_to_label):
                ax.plot(xs, ys, color='purple', linewidth=purple_line_width, alpha=purple_line_alpha, clip_on=False)

            if draw_sequence and start == start_to_label:
                width = 0.9
                center = start + 0.5
                left_edge = center - (len(bs) * 0.5 * width)
                for x_offset, b in enumerate(bs):
                    ax.annotate(transform_seq(b),
                                xy=(left_edge + (x_offset * width) + width / 2, y + (wt_height / 2)),
                                xycoords='data',
                                ha='center',
                                va='center',
                                size=text_size * 1,
                                color=hits.visualize.igv_colors[transform_seq(b)],
                                weight='bold',
                                annotation_clip=False,
                               )

    def draw_duplication(y, duplication, source_name):
        window_left, window_right = windows[source_name]
        draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

        starts, ends = duplication.ref_junctions[0]
        starts = np.array(starts) - offset
        ends = np.array(ends) - offset
        bottom = y - 0.55 * wt_height
        top = y + 0.55 * wt_height

        for i, (start, end) in enumerate(zip(starts, ends)):
            if i == 0:
                alpha = 1
            else:
                alpha = 0.3

            y_offset = i * wt_height * 0.1

            draw_rect(source_name, start - 0.5, end + 0.5, bottom + y_offset, top + y_offset, alpha, color='tab:purple', fill=False)

        if draw_all_sequence:
            draw_sequence(y, source_name, alpha=sequence_alpha)

    def draw_wild_type(y, source_name, on_top=False, guides_to_draw=None):
        ti = target_infos[source_name]
        offset = offsets[source_name]
        window_left, window_right = windows[source_name]

        if on_top or not draw_wild_type_on_top:
            draw_sequence(y, source_name, alpha=1)

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

                    if override_protospacer_color is not None:
                        protospacer_color = override_protospacer_color
                    else:
                        protospacer_color = ti.protospacer_color

                    if override_PAM_color is not None:
                        PAM_color = override_PAM_color
                    else:
                        PAM_color = ti.PAM_color

                    draw_rect(source_name, guide_start, guide_end, y - wt_height / 2, y + wt_height / 2, None, color=protospacer_color)
                    draw_rect(source_name, PAM_start, PAM_end, y - wt_height / 2, y + wt_height / 2, None, color=PAM_color)


                    if not on_top:
                        # Draw PAMs.
                        draw_rect(source_name, window_left - 0.5, min(PAM_start, guide_start), y - wt_height / 2, y + wt_height / 2, block_alpha)
                        draw_rect(source_name, max(PAM_end, guide_end), window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

        else:
            draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)

        if on_top:
            for feature_name in features_to_draw:
                feature = ti.features[ti.target, feature_name]

                start = feature.start - 0.5 - offset
                end = feature.end + 0.5 - offset

                color = feature.attribute.get('color', 'grey')

                draw_rect(source_name, start, end, y - wt_height / 2, y + wt_height / 2, 0.8, color)
                ax.annotate(feature_name,
                            xy=(np.mean([start, end]), y + wt_height / 2),
                            xytext=(0, 5),
                            textcoords='offset points',
                            color='black',
                            annotation_clip=False,
                            ha='center',
                            va='bottom',
                           )

    def draw_donor(y, HDR_outcome, deletion_outcome, insertion_outcome, source_name, on_top=False):
        ti = target_infos[source_name]
        transform_seq = transform_seqs[source_name]
        window_left, window_right = windows[source_name]
        SNP_ps = sorted(p for (s, p), b in ti.fingerprints[ti.target])

        p_to_i = SNP_ps.index
        i_to_p = dict(enumerate(SNP_ps))

        SNP_xs = set()
        observed_SNP_idxs = set()

        for ((strand, position), ref_base), read_base in zip(ti.fingerprints[ti.target], HDR_outcome.donor_SNV_read_bases):
            x = position - offset
            if window_left <= x <= window_right:
                # Note: read base of '-' means it was deleted
                if ref_base != read_base and read_base != '_' and read_base != '-':
                    SNP_xs.add(x)
                    observed_SNP_idxs.add(p_to_i(position))

                    ax.annotate(transform_seq(read_base),
                                xy=(x + shift_x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                alpha=0.35,
                                #color=hits.visualize.igv_colors[transform_seq(read_base)],
                                #weight='bold',
                                annotation_clip=False,
                               )
            
                if read_base != '-':
                    if  read_base == '_':
                        color = 'grey'
                        alpha = 0.3
                    else:
                        color = hits.visualize.igv_colors[transform_seq(read_base)]
                        alpha = 0.7

                    draw_rect(source_name, x - 0.5, x + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

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
                for block in blocks:
                    start = i_to_p[block[0]] - offset
                    end = i_to_p[block[-1]] - offset
                    x_buffer = 0.7
                    y_buffer = 0.7
                    draw_rect(source_name, start - x_buffer, end + x_buffer, y - y_buffer * wt_height, y + y_buffer * wt_height, 0.5, fill=False)
        
        all_deletions = [(d, 'red', True) for d in HDR_outcome.donor_deletions]
        if deletion_outcome is not None:
            all_deletions.append((deletion_outcome.deletion, 'black', True))

        if len(all_deletions) == 0:
            draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
        elif len(all_deletions) == 1:
            deletion, color, draw_MH = all_deletions[0]

            if len(target_infos) > 1:
                background_color = ti.protospacer_color
            else:
                background_color = 'black'

            draw_deletion(y, deletion, source_name, color=color, draw_MH=draw_MH, background_color=background_color)

        elif len(all_deletions) > 1:
            raise NotImplementedError

        if insertion_outcome is not None:
            draw_insertion(y, insertion_outcome.insertion, source_name)

        if draw_all_sequence:
            draw_sequence(y, source_name, xs_to_skip=SNP_xs, alpha=sequence_alpha)

        if on_top:
            strands = set(SNV['strand'] for SNV in ti.donor_SNVs['donor'].values())
            if len(strands) > 1:
                raise ValueError('donor strand is weird')
            else:
                strand = strands.pop()

            arrow_ys = [y + wt_height * 0.4, y, y - wt_height * 0.4]

            for x in range(window_left, window_right + 1, 1):
                if x in SNP_xs:
                    continue

                if strand == '+':
                    arrow_xs = [x - 0.5, x + 0.5, x - 0.5]
                else:
                    arrow_xs = [x + 0.5, x - 0.5, x + 0.5]

                ax.plot(arrow_xs, arrow_ys,
                        color='black',
                        alpha=0.2,
                        clip_on=False,
                )

    def draw_programmed_edit(y, programmed_edit_outcome, source_name):
        ti = target_infos[source_name]
        transform_seq = transform_seqs[source_name]
        window_left, window_right = windows[source_name]
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

                    if read_base != '-' and read_base != '_':
                        ax.annotate(transform_seq(read_base),
                                    xy=(x + shift_x, y),
                                    xycoords='data', 
                                    ha='center',
                                    va='center',
                                    size=text_size,
                                    alpha=0.35,
                                    #color=hits.visualize.igv_colors[transform_seq(read_base)],
                                    #weight='bold',
                                    annotation_clip=False,
                                )
                
                    if read_base == '_':
                        color = 'grey'
                        alpha = 0.3
                    else:
                        color = hits.visualize.igv_colors[transform_seq(read_base)]
                        alpha = 0.7

                    draw_rect(source_name, x - 0.5, x + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

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
            for block in blocks:
                start = i_to_p[block[0]] - offset
                end = i_to_p[block[-1]] - offset
                x_buffer = 0.7
                y_buffer = 0.7
                draw_rect(source_name, start - x_buffer, end + x_buffer, y - y_buffer * wt_height, y + y_buffer * wt_height, 0.5, fill=False)

        if len(programmed_edit_outcome.deletions) == 0:
            draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
        elif len(programmed_edit_outcome.deletions) == 1:
            deletion = DeletionOutcome(programmed_edit_outcome.deletions[0]).undo_anchor_shift(ti.anchor).deletion
            draw_deletion(y, deletion, source_name, color='black', draw_MH=False)
        else:
            raise NotImplementedError

        if len(programmed_edit_outcome.insertions) == 0:
            pass
        elif len(programmed_edit_outcome.insertions) == 1:
            insertion = InsertionOutcome(programmed_edit_outcome.insertions[0]).undo_anchor_shift(ti.anchor).insertion
            insertion = ti.expand_degenerate_indel(insertion)
            draw_insertion(y, insertion, source_name, draw_sequence=False)
        else:
            raise NotImplementedError
        
        if draw_all_sequence:
            draw_sequence(y, source_name, xs_to_skip=SNP_xs, alpha=sequence_alpha)

    def draw_pegRNA(source_name):
        ti = target_infos[source_name]
        SNVs, _, _, _, (flap_subsequences, target_subsequences) = ti.pegRNA.extract_edits_from_alignment()

        feature = ti.features[ti.target, f'{ti.pegRNA.name}_PBS']

        start = feature.start - 0.5 - offset
        end = feature.end + 0.5 - offset

        color = feature.attribute['color']

        y = len(outcome_order) + 1

        draw_rect(source_name, start, end, y - wt_height / 2, y + wt_height / 2, 0.8, color)
        ax.annotate('PBS',
                    xy=(np.mean([start, end]), y + wt_height / 2),
                    xytext=(0, 5),
                    textcoords='offset points',
                    color='black',
                    annotation_clip=False,
                    ha='center',
                    va='bottom',
                   )

        PBS_seq = hits.utilities.reverse_complement(ti.pegRNA.components['PBS'])

        if guide.strand == '+':
            xs = range(-len(PBS_seq) + 1, 1)
        else:
            xs = range(len(PBS_seq), 0, -1)
            PBS_seq = hits.utilities.complement(PBS_seq)

        if flip:
            PBS_seq = hits.utilities.complement(PBS_seq)

        for x, b in zip(xs, PBS_seq):
            ax.annotate(b,
                        xy=(x, y),
                        xycoords='data', 
                        ha='center',
                        va='center',
                        size=text_size,
                        annotation_clip=False,
                       )

        RTT_xs = []
        RTT_rc = hits.utilities.reverse_complement(ti.pegRNA.components['RTT'])
        RTT_aligned_seq = ''
        for (target_start, target_end), (flap_start, flap_end) in zip(target_subsequences, flap_subsequences):
            # target_subsequences are in downstream_of_nick coords and end is exclusive.
            # 0 is cut_after

            RTT_subsequence = RTT_rc[flap_start:flap_end]
            if guide.strand == '-':
                RTT_subsequence = hits.utilities.complement(RTT_subsequence)
            RTT_aligned_seq += RTT_subsequence

            if guide.strand == '+':
                xs_start = target_start + 1
                xs_end = target_end + 1
                step = 1
            else:
                xs_start = -target_start
                xs_end = -target_end
                step = -1

            RTT_xs.extend(range(xs_start, xs_end, step))

            rect_start, rect_end = sorted([xs_start, xs_end])

            rect_start = rect_start - (0.5 * step)
            rect_end = rect_end - (0.5 * step)

            color = knock_knock.pegRNAs.default_feature_colors['RTT']

            draw_rect(source_name, rect_start, rect_end, y - wt_height / 2, y + wt_height / 2, 0.8, color)

        for (_, previous_end), (next_start, _) in zip(target_subsequences, target_subsequences[1:]):
            draw_rect(source_name, previous_end + 1 - 0.5, next_start + 1 - 0.5, y - del_height / 2, y + del_height / 2, 0.4, color='black')

        seq = seqs[source_name]
        window_left, window_right = windows[source_name]
        for x, b in zip(RTT_xs, RTT_aligned_seq):
            if window_left <= x <= window_right:
                target_b = seq[-window_left + x]

                if flip:
                    b_to_draw = hits.utilities.complement(b)
                else:
                    b_to_draw = b

                if b != target_b:
                    color = hits.visualize.igv_colors[b_to_draw]
                    weight = 'bold'
                else:
                    color = 'black'
                    weight = 'normal'

                ax.annotate(b_to_draw,
                            xy=(x, y),
                            xycoords='data', 
                            ha='center',
                            va='center',
                            size=text_size,
                            annotation_clip=False,
                            color=color,
                            weight=weight,
                        )

        if ti.pegRNA_programmed_insertion is not None:
            draw_insertion(y, ti.pegRNA_programmed_insertion, source_name)

        ax.annotate('RTT',
                    xy=(np.mean(RTT_xs), y + wt_height / 2),
                    xytext=(0, 5),
                    textcoords='offset points',
                    color='black',
                    annotation_clip=False,
                    ha='center',
                    va='bottom',
                   )

    draw_pegRNA('single_source')

    for i, (source_name, category, subcategory, details) in enumerate(outcome_order):
        ti = target_infos[source_name]
        transform_seq = transform_seqs[source_name]
        window_left, window_right = windows[source_name]
        y = num_outcomes - i - 1

        if len(target_infos) > 1:
            background_color = ti.protospacer_color
        else:
            background_color = 'black'
            
        if category == 'deletion' or \
           (category == 'simple indel' and subcategory.startswith('deletion')) or \
           (category == 'wild type' and subcategory == 'short indel far from cut' and degenerate_indel_from_string(details).kind == 'D'):

            deletion = DeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor).deletion
            deletion = ti.expand_degenerate_indel(deletion)

            xs_to_skip = draw_deletion(y, deletion, source_name, background_color=background_color)
            if draw_all_sequence:
                draw_sequence(y, source_name, xs_to_skip, alpha=sequence_alpha)
        
        elif category == 'insertion' or (category == 'simple indel' and subcategory.startswith('insertion')):
            insertion = InsertionOutcome.from_string(details).undo_anchor_shift(ti.anchor).insertion
            insertion = ti.expand_degenerate_indel(insertion)

            draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha, color=background_color)
            draw_insertion(y, insertion, source_name)

            if draw_all_sequence:
                draw_sequence(y, source_name, alpha=sequence_alpha)

        elif category == 'insertion with deletion':
            outcome = InsertionWithDeletionOutcome.from_string(details).undo_anchor_shift(ti.anchor)
            insertion = outcome.insertion_outcome.insertion
            deletion = outcome.deletion_outcome.deletion
            draw_insertion(y, insertion, source_name)
            draw_deletion(y, deletion, source_name, background_color=background_color)
                
        elif category == 'mismatches' or (category == 'wild type' and subcategory == 'mismatches'):
            SNV_xs = set()
            draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha)
            snvs = SNVs.from_string(details) 

            # Undo anchor shift.
            snvs = SNVs([SNV(s.position + ti.anchor, s.basecall, s.quality) for s in snvs])

            for snv in snvs:
                x = snv.position - offset
                SNV_xs.add(x)
                if window_left <= x <= window_right:
                    ax.annotate(transform_seq(snv.basecall),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                weight='bold',
                            )
            
            for (strand, position), ref_base in ti.fingerprints[ti.target]:
                color = 'grey'
                alpha = 0.3
                draw_rect(source_name, position - offset - 0.5, position - offset + 0.5, y - wt_height / 2, y + wt_height / 2, alpha, color=color)

            if draw_all_sequence:
                draw_sequence(y, source_name, xs_to_skip=SNV_xs, alpha=sequence_alpha)

        elif category == 'wild type' or category == 'WT':
            draw_wild_type(y, source_name)

        elif category == 'deletion + adjacent mismatch' or category == 'deletion + mismatches':
            outcome = DeletionPlusMismatchOutcome.from_string(details).undo_anchor_shift(ti.anchor)
            xs_to_skip = draw_deletion(y, outcome.deletion_outcome.deletion, draw_MH=True)
            
            for snv in outcome.mismatch_outcome.snvs:
                x = snv.position - offset
                xs_to_skip.add(x)

                if window_left <= x <= window_right:
                    ax.annotate(transform_seq(snv.basecall),
                                xy=(x, y),
                                xycoords='data', 
                                ha='center',
                                va='center',
                                size=text_size,
                                color=hits.visualize.igv_colors[transform_seq(snv.basecall.upper())],
                                weight='bold',
                            )

                if category == 'deletion + adjacent mismatch':
                    # Draw box around mismatch to distinguish from MH.
                    x_buffer = 0.7
                    y_buffer = 0.7
                    draw_rect(source_name, x - x_buffer, x + x_buffer, y - y_buffer * wt_height, y + y_buffer * wt_height, 0.5, fill=False)

            if draw_all_sequence:
                draw_sequence(y, source_name, xs_to_skip, alpha=sequence_alpha)

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
    
            draw_donor(y, HDR_outcome, deletion_outcome, insertion_outcome, source_name, False)

        elif category == 'intended edit':
            draw_programmed_edit(y, ProgrammedEditOutcome.from_string(details), source_name)

        elif category == 'duplication' and subcategory == 'simple':
            duplication_outcome = DuplicationOutcome.from_string(details).undo_anchor_shift(ti.anchor)
            draw_duplication(y, duplication_outcome, source_name)
            
        else:
            label = f'{category}, {subcategory}, {details}'

            if replacement_text_for_complex is not None:
                label = replacement_text_for_complex.get(label, label)

            ax.annotate(label,
                        xy=(0, y),
                        xycoords=('axes fraction', 'data'), 
                        xytext=(5, 0),
                        textcoords='offset points',
                        ha='left',
                        va='center',
                        size=text_size,
                        )
            if len(target_infos) > 1:
                draw_rect(source_name, window_left - 0.5, window_right + 0.5, y - wt_height / 2, y + wt_height / 2, block_alpha, color=background_color)

    if draw_donor_on_top and len(ti.donor_SNVs['target']) > 0:
        donor_SNV_read_bases = ''.join(d['base'] for name, d in sorted(ti.donor_SNVs['donor'].items()))
        strands = set(SNV['strand'] for SNV in ti.donor_SNVs['donor'].values())
        if len(strands) > 1:
            raise ValueError('donor strand is weird')
        else:
            strand = strands.pop()
        HDR_outcome = HDROutcome(donor_SNV_read_bases, [])
        if strand == '-':
            y = num_outcomes + 0.5
        else:
            y = num_outcomes + 2.5
        draw_donor(y, HDR_outcome, None, None, source_name, on_top=True)

    if draw_wild_type_on_top:
        y = num_outcomes
        if draw_donor_on_top:
            y += 1.5
        
        if len(target_infos) > 1:
            raise ValueError

        draw_wild_type(y, source_name, on_top=True, guides_to_draw=target_infos[single_source_name].protospacer_names)
        ax.set_xticks([])
                
    if not preserve_x_lims:
        # Some uses don't want x lims to be changed.
        x_lims = [window_left - 0.5, window_right + 0.5]
        if flip:
            ax.set_xlim(*x_lims[::-1])
        else:
            ax.set_xlim(*x_lims)

        ax.xaxis.tick_top()
        ax.axhline(num_outcomes + 0.5 - 1, color='black', alpha=0.75, clip_on=False)

    ax.set_ylim(-0.5, num_outcomes - 0.5)
    ax.set_frame_on(False)

    title_color = kwargs.get('title_color')
    if title_color is None:
        if len(target_infos) > 1:
            title_color = 'black'
        else:
            title_color = target_infos[single_source_name].PAM_color

    if title:
        ax.annotate(title,
                    xy=(0.5, 1),
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(0, kwargs.get('title_offset', 20)),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=kwargs.get('title_size', 14),
                    color=title_color,
                   )
        
    ax.set_yticks([])
    
    return fig, ax

def add_frequencies(fig, ax, count_source, outcome_order, fixed_guide='none', text_only=False):
    if isinstance(count_source, (dict, pd.Series)):
        counts = np.array([count_source[outcome] for outcome in outcome_order])
        if isinstance(count_source, dict):
            total = sum(count_source.values())
        else:
            total = sum(count_source.values)
        freqs = counts / total
    else:
        pool = count_source
        freqs = pool.non_targeting_fractions('perfect', fixed_guide).loc[outcome_order]
        counts = pool.non_targeting_counts('perfect', fixed_guide).loc[outcome_order]

    ys = np.arange(len(outcome_order) - 1, -1, -1)
    
    for y, freq, count in zip(ys, freqs, counts):
        ax.annotate('{:> 7.2%} {:>8s}'.format(freq, '({:,})'.format(count)),
                    xy=(1, y),
                    xycoords=('axes fraction', 'data'),
                    xytext=(6, 0),
                    textcoords=('offset points'),
                    family='monospace',
                    ha='left',
                    va='center',
                   )

    if not text_only:
        ax_p = ax.get_position()
        
        width_inches, height_inches = fig.get_size_inches()

        width = 2 / width_inches
        gap = 0.5 / width_inches

        freq_ax = fig.add_axes((ax_p.x1 + 4 * gap, ax_p.y0, width, ax_p.height), sharey=ax)
        freq_ax_p = freq_ax.get_position()
        log_ax = fig.add_axes((freq_ax_p.x1 + gap, ax_p.y0, width, ax_p.height), sharey=ax)
        log_ax_p = log_ax.get_position()
        cumulative_ax = fig.add_axes((log_ax_p.x1 + gap, ax_p.y0, width, ax_p.height), sharey=ax)
        
        freq_ax.plot(freqs, ys, 'o-', markersize=2, color='black')
        log_ax.plot(np.log10(freqs), ys, 'o-', markersize=2, color='black')

        cumulative = freqs.cumsum()
        cumulative_ax.plot(cumulative, ys, 'o-', markersize=2, color='black')
        
        freq_ax.set_xlim(0, max(freqs) * 1.05)
        cumulative_ax.set_xlim(0, cumulative[-1] * 1.05)
        
        for p_ax in [freq_ax, log_ax, cumulative_ax]:
            p_ax.set_yticks([])
            p_ax.xaxis.tick_top()
            p_ax.spines['left'].set_alpha(0.3)
            p_ax.spines['right'].set_alpha(0.3)
            p_ax.tick_params(labelsize=6)
            p_ax.grid(axis='x', alpha=0.3)
            
            p_ax.spines['bottom'].set_visible(False)
            
            p_ax.xaxis.set_label_position('top')
        
        freq_ax.set_xlabel('frequency', size=8)
        log_ax.set_xlabel('frequency (log10)', size=8)
        cumulative_ax.set_xlabel('cumulative frequency', size=8)
        
        ax.set_ylim(-0.5, len(outcome_order) - 0.5)

def add_values(fig, ax, vals, width=0.2):
    ax_p = ax.get_position()
    
    offset = 0.04

    ys = np.arange(len(vals) - 1, -1, -1)
    
    val_ax = fig.add_axes((ax_p.x1 + offset, ax_p.y0, width, ax_p.height), sharey=ax)
    
    val_ax.plot(vals, ys, 'o-', markersize=2, color='black')
    
    val_ax.set_yticks([])
    val_ax.xaxis.tick_top()
    val_ax.spines['left'].set_alpha(0.3)
    val_ax.spines['right'].set_alpha(0.3)
    val_ax.tick_params(labelsize=6)
    val_ax.grid(axis='x', alpha=0.3)
    
    val_ax.spines['bottom'].set_visible(False)
    
    val_ax.xaxis.set_label_position('top')
    
    ax.set_ylim(-0.5, len(vals) - 0.5)

def plot_with_frequencies(pool, outcomes, fixed_guide='none', text_only=False, count_source=None, **kwargs):
    if count_source is None:
        count_source = pool

    fig = plot(outcomes, pool.target_info, fixed_guide=fixed_guide, **kwargs)
    num_outcomes = kwargs.get('num_outcomes')
    add_frequencies(fig, fig.axes[0], count_source, outcomes[:num_outcomes], text_only=text_only)
    return fig

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
        PAM_feature = target_info.PAM_features[f'{target_info.primary_protospacer}_PAM']
        self.PAM_color = PAM_feature.attribute['color']
        self.ax_on_bottom = ax_on_bottom

        if cut_color == 'PAM':
            diagram_kwargs['cut_color'] = self.PAM_color

        self.inches_per_nt = inches_per_nt
        self.inches_per_outcome = inches_per_outcome
        self.outcome_ax_width = outcome_ax_width

        self.fig = None
        self.title = title

        self.ordered_axs = []

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
        if isinstance(self.outcomes[0], tuple) and len(self.outcomes[0]) == 3:
            self.fig, ax = plot(self.outcomes,
                                self.target_info,
                                ax=self.axs_by_name['diagram'],
                                title=self.title,
                                inches_per_outcome=self.inches_per_outcome,
                                inches_per_nt=self.inches_per_nt,
                                **self.diagram_kwargs,
                               )
        else:
            self.fig, ax = plt.subplots(figsize=(self.outcome_ax_width,
                                                 len(self.outcomes) * self.inches_per_outcome,
                                                ),
                                       )

            for i, outcome in enumerate(self.outcomes):
                if isinstance(outcome, tuple):
                    outcome = ', '.join(outcome)

                label = self.label_aliases.get(outcome, outcome)
                ax.annotate(label,
                            xy=(1, len(self.outcomes) - i - 1),
                            xycoords=('axes fraction', 'data'),
                            ha='right',
                            va='center',
                            size=self.diagram_kwargs.get('label_size', 8),
                )

            ax.set_ylim(-0.5, len(self.outcomes) - 0.5)

            ax.axis('off')
            ax.set_title(self.title)

        self.axs_by_name['diagram'] = ax
        self.ordered_axs.append(ax)

        ax_p = ax.get_position()
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

    def add_ax_above(self, height_multiple=10, gap=2):
        ax_p = self.axs_by_name['diagram'].get_position()
        x0 = ax_p.x0 

        width = ax_p.width
        height = self.height_per_heatmap_cell * height_multiple
        
        y0 = ax_p.y1 + self.height_per_heatmap_cell * gap

        ax = self.fig.add_axes((x0, y0, width, height), sharex=self.axs_by_name['diagram'])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        self.axs_by_name['above'] = ax

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

        xs = [value_source.get(outcome, np.nan) for outcome in self.outcomes]

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

    def plot_on_ax_above(self, xs, value_source,
                         marker_alpha=1,
                         line_alpha=1,
                         label='',
                         marker='.',
                         **plot_kwargs,
                        ):
        ax = self.axs_by_name['above']

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

    def plot_pegRNA_conversion_fractions_above(self, group, y_max=None):
        def SNV_name_to_x(SNV_name):
            SNVs = group.target_info.pegRNA_SNVs[group.target_info.target]
            p = SNVs[SNV_name]['position']
            x = p - group.target_info.cut_after - 0.5
            return x
        
        pegRNA_conversion_fractions = group.pegRNA_conversion_fractions.copy()

        x_lims = self.axs_by_name['diagram'].get_xlim()

        flipped = x_lims[0] > x_lims[1]

        xs = pegRNA_conversion_fractions.index.map(SNV_name_to_x)
        pegRNA_conversion_fractions.index = xs
        pegRNA_conversion_fractions = pegRNA_conversion_fractions.sort_index()
        
        if 'above' not in self.axs_by_name:
            self.add_ax_above(gap=4, height_multiple=7)
        
        for condition, fs in pegRNA_conversion_fractions.items():
            self.plot_on_ax_above(xs, fs * 100,
                                  line_alpha=0.5,
                                  linewidth=1,
                                  markersize=5,
                                  color='black',
                                  label=condition,
                                 )

        ax = self.axs_by_name['above']

        ax.set_ylim(0, y_max)

        ax.set_title(group.group)
        
        ax.set_ylabel('Total %\nincorporation\nat position', size=12)
        ax.tick_params(labelsize=8)

        x_bounds = [min(xs) - 1, max(xs) + 1]
        if flipped:
            x_bounds = x_bounds[::-1]

        ax.spines.left.set_position(('data', x_bounds[0]))
        ax.spines.bottom.set_visible(False)

        for y in ax.get_yticks():
            if y == 0:
                alpha = 1
                clip_on = False
            else:
                alpha = 0.25
                clip_on = True

            ax.plot(x_bounds, [y for x in x_bounds], clip_on=clip_on, color='black', alpha=alpha)

        return pegRNA_conversion_fractions