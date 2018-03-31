import matplotlib
matplotlib.use('Agg', warn=False)

import copy
import io
import PIL
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import ipywidgets

import sequencing.utilities as utilities
import sequencing.interval as interval
import sequencing.sam as sam

from . import experiment
from . import target_info as target_info_module
from . import layout as layout_module

def get_indel_info(alignment):
    indels = []
    for i, (kind, length) in enumerate(alignment.cigar):
        if kind == sam.BAM_CDEL:
            nucs_before = sam.total_read_nucs(alignment.cigar[:i])
            centered_at = np.mean([sam.true_query_position(p, alignment) for p in [nucs_before - 1, nucs_before]])
            indels.append(('deletion', (centered_at, length)))
        elif kind == sam.BAM_CINS:
            first_edge = sam.total_read_nucs(alignment.cigar[:i])
            second_edge = first_edge + length
            starts_at, ends_at = sorted(sam.true_query_position(p, alignment) for p in [first_edge, second_edge])
            indels.append(('insertion', (starts_at, ends_at)))
            
    return indels

def plot_read(alignments,
              target_info,
              parsimonious=False,
              show_qualities=False,
              zoom_in=None,
              size_multiple=1,
              paired=False,
              **kwargs,
             ):
    alignments = copy.deepcopy(alignments)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {name: 'C{0}'.format(i) for i, name in enumerate(target_info.reference_sequences)}

    if paired:
        reverse_complement = False
    elif not all(al.is_unmapped for al in alignments):
        layout_info = {'alignments': {'all': alignments}}
        layout_module.identify_flanking_target_alignments(layout_info, target_info)
        reverse_complement = (layout_info['strand'] == '-')
    else:
        reverse_complement = False

    per_rname = 0.06
    gap_between_als = 0.06 * 0.2
    arrow_height = 0.005
    arrow_width = 0.01
    
    max_y = gap_between_als
    
    if parsimonious:
        alignments = interval.make_parsimoninous(alignments)
        
    query_name = alignments[0].query_name
    query_length = alignments[0].query_length
    quals = alignments[0].query_qualities
    
    if zoom_in is not None:
        x_min = zoom_in[0] * query_length
        x_max = zoom_in[1] * query_length
    else:
        x_min = -0.02 * query_length
        x_max = 1.02 * query_length
    
    kwargs = {'linewidth': 2, 'color': 'black'}
    if paired:
        offsets = [0, -gap_between_als * 0.15]
        endpoints = [
            [0, 250],
            [query_length - 1, query_length - 1 - 250],
        ]
        signs = [1, -1]
        for (start, end), sign, offset in zip(endpoints, signs, offsets):
            ax.plot([start, end, end - sign * query_length * arrow_width],
                    [offset, offset, offset + sign * arrow_height],
                    clip_on=False,
                    **kwargs,
                   )
    else:
        ax.plot([0, query_length], [0, 0], **kwargs)

        arrow_ys = [0, arrow_height]

        if reverse_complement:
            arrow_xs = [0, query_length * arrow_width]
        else:
            arrow_xs = [query_length, query_length * (1 - arrow_width)]
        ax.plot(arrow_xs, arrow_ys, **kwargs)
    
    ax.annotate('sequencing read',
                xy=(1, 0),
                xycoords=('axes fraction', 'data'),
                xytext=(15, 0),
                textcoords='offset points',
                color='black',
                ha='left',
                va='center',
               )

    if all(al.is_unmapped for al in alignments):
        by_reference_name = []
    else:
        alignments = sorted(alignments, key=lambda al: (al.reference_name, sam.query_interval(al)))
        by_reference_name = list(utilities.group_by(alignments, lambda al: al.reference_name))
    
    rname_starts = np.cumsum([1] + [len(als) for n, als in by_reference_name])
    offsets = {name: start for (name, als), start in zip(by_reference_name, rname_starts)}
    
    for ref_name, ref_alignments in by_reference_name:
        if reverse_complement:
            for alignment in ref_alignments:
                alignment.is_reverse = not alignment.is_reverse

        ref_alignments = ref_alignments[:10]
        
        offset = offsets[ref_name]
        color = colors[ref_name]

        average_y = (offset  + 0.5 * (len(ref_alignments) - 1)) * gap_between_als
        ax.annotate(ref_name,
                    xy=(1, average_y),
                    xycoords=('axes fraction', 'data'),
                    xytext=(15, 0),
                    textcoords='offset points',
                    color=color,
                    ha='left',
                    va='center',
                   )

        for i, alignment in enumerate(ref_alignments):
            start, end = sam.query_interval(alignment)
            strand = sam.get_strand(alignment)
            y = (offset + i) * gap_between_als
            
            # Annotate the ends of alignments with reference position numbers and vertical lines.
            for x, which in ((start, 'start'), (end, 'end')):
                if (which == 'start' and strand == '+') or (which == 'end' and strand == '-'):
                    r = alignment.reference_start
                else:
                    r = alignment.reference_end - 1

                ax.plot([x, x], [0, y], color=color, alpha=0.3)
                if which == 'start':
                    kwargs = {'ha': 'right', 'xytext': (-2, 0)}
                else:
                    kwargs = {'ha': 'left', 'xytext': (2, 0)}

                ax.annotate('{0:,}'.format(r),
                            xy=(x, y),
                            xycoords='data',
                            textcoords='offset points',
                            color=color,
                            va='center',
                            size=6,
                            **kwargs,
                           )
                
            # Draw the alignment, with downward dimples at insertions and upward loops at deletions.
            xs = [start]
            ys = [y]
            indels = sorted(get_indel_info(alignment), key=lambda t: t[1][0])
            for kind, info in indels:
                if kind == 'deletion':
                    centered_at, length = info

                    # Cap how wide the loop can be.
                    capped_length = min(100, length)
                    
                    if length <= 2:
                        height = 0.0015
                        indel_xs = [centered_at, centered_at, centered_at]
                        indel_ys = [y, y + height, y]
                    else:
                        width = query_length * 0.001
                        height = 0.006

                        indel_xs = [
                            centered_at - width,
                            centered_at - 0.5 * capped_length,
                            centered_at + 0.5 * capped_length,
                            centered_at + width,
                        ]
                        indel_ys = [y, y + height, y + height, y]

                        ax.annotate(str(length),
                                    xy=(centered_at, y + height),
                                    xytext=(0, 1),
                                    textcoords='offset points',
                                    ha='center',
                                    va='bottom',
                                    size=6,
                                   )

                elif kind == 'insertion':
                    starts_at, ends_at = info
                    centered_at = np.mean([starts_at, ends_at])
                    length = ends_at - starts_at
                    if length <= 2:
                        height = 0.0015
                    else:
                        height = 0.004
                        ax.annotate(str(length),
                                    xy=(centered_at, y - height),
                                    xytext=(0, -1),
                                    textcoords='offset points',
                                    ha='center',
                                    va='top',
                                    size=6,
                                   )
                    indel_xs = [starts_at, centered_at, ends_at]
                    indel_ys = [y, y - height, y]
                    
                xs.extend(indel_xs)
                ys.extend(indel_ys)
                
            xs.append(end)
            ys.append(y)
            
            max_y = max(max_y, max(ys))
            
            kwargs = {'color': color, 'linewidth': 1.5}
            ax.plot(xs, ys, **kwargs)
            
            if strand == '+':
                arrow_xs = [end, end - query_length * arrow_width]
                arrow_ys = [y, y + arrow_height]
            else:
                arrow_xs = [start, start + query_length * arrow_width]
                arrow_ys = [y, y - arrow_height]
                
            draw_arrow = True
            if zoom_in is not None:
                if not all(x_min <= x <= x_max for x in arrow_xs):
                    draw_arrow = False

            if draw_arrow:
                ax.plot(arrow_xs, arrow_ys, clip_on=False, **kwargs)

            features = target_info.features
            target = target_info.target
            donor = target_info.donor
            features_to_show = [
                (target, 'forward primer'),
                (target, 'reverse primer'),
                (target, "3' HA"),
                (target, "5' HA"),
                (target, 'sgRNA'),
                (donor, "3' HA"),
                (donor, "5' HA"),
                (donor, 'GFP'),
            ]
            
            q_to_r = {sam.true_query_position(q, alignment): r
                      for q, r in alignment.aligned_pairs
                      if r is not None and q is not None
                     }
            for feature_reference, feature_name in features_to_show:
                if ref_name != feature_reference:
                    continue

                feature = features[feature_reference, feature_name]
                feature_color = feature.attribute['color']
                
                qs = [q for q, r in q_to_r.items() if feature.start <= r <= feature.end]
                if not qs:
                    continue

                xs = [min(qs), max(qs)]
                if xs[1] - xs[0] < 5:
                    continue
                
                rs = [feature.start, feature.end]
                if strand == '-':
                    rs = rs[::-1]
                    
                for ha, q, r in zip(['left', 'right'], xs, rs):
                    nts_missing = abs(q_to_r[q] - r)
                    if nts_missing != 0 and xs[1] - xs[0] > 20:
                        ax.annotate(str(nts_missing),
                                    xy=(q, 0),
                                    ha=ha,
                                    va='bottom',
                                    xytext=(3 if ha == 'left' else -3, 1),
                                    textcoords='offset points',
                                    size=6,
                                   )
                        
                bottom_y = -5
                    
                ax.fill_between(xs, [y] * 2, [0] * 2, color=feature_color, alpha=0.7)
                
                if xs[1] - xs[0] < 20:
                    continue
                ax.annotate(feature.attribute['ID'],
                            xy=(np.mean(xs), 0),
                            xycoords='data',
                            xytext=(0, bottom_y),
                            textcoords='offset points',
                            va='top',
                            ha='center',
                            color=feature_color,
                            size=10,
                            weight='bold',
                           )

    ax.set_title(query_name, y=1.2)
        
    ax.set_ylim(-0.2 * max_y, 1.1 * max_y)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_alpha(0.1)
    for edge in 'left', 'top', 'right':
        ax.spines[edge].set_color('none')
        
    ax.tick_params(pad=14)
    fig.set_size_inches((12 * size_multiple, 4 * max_y / 0.15 * size_multiple))
    
    if show_qualities:
        ax.plot(np.array(quals) * max_y / 93, color='black', alpha=0.5)
        
    return fig

def make_stacked_Image(als_iter, target_info, **kwargs):
    ims = []
    for als in als_iter:
        if als is None:
            continue
            
        fig = plot_read(als, target_info, **kwargs)

        #fig.axes[0].set_title('_', y=1.2, color='white')
        
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format='png', bbox_inches='tight')
            im = PIL.Image.open(buffer)
            im.load()
            ims.append(im)
        plt.close(fig)
        
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
    
def explore(by_outcome=False):
    target_names = [t.name for t in target_info_module.get_all_targets()]

    widgets = {
        'target': ipywidgets.Select(options=target_names, value=target_names[0]),
        'dataset': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='400px')),
        'parsimonious': ipywidgets.ToggleButton(value=True),
        'show_qualities': ipywidgets.ToggleButton(value=False),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
        'zoom_in': ipywidgets.FloatRangeSlider(value=[-0.02, 1.02], min=-0.02, max=1.02, step=0.001, continuous_update=False, layout=ipywidgets.Layout(width='1200px')),
    }

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    exps = experiment.get_all_experiments()

    def populate_datasets(change):
        target = widgets['target'].value
        previous_value = widgets['dataset'].value
        datasets = sorted([exp.name for exp in exps if exp.target_info.name == target])
        widgets['dataset'].options = datasets

        if datasets:
            if previous_value in datasets:
                widgets['dataset'].value = previous_value
                populate_outcomes(None)
            else:
                widgets['dataset'].value = datasets[0]
        else:
            widgets['dataset'].value = None

    def populate_outcomes(change):
        previous_value = widgets['outcome'].value
        exp = experiment.Experiment(widgets['dataset'].value)
        outcomes = exp.outcomes
        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in outcomes]
        if outcomes:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    def populate_read_ids(change):
        exp = experiment.Experiment(widgets['dataset'].value)

        if by_outcome:
            qnames = exp.outcome_query_names(widgets['outcome'].value)
        else:
            qnames = list(itertools.islice(exp.query_names(), 200))

        widgets['read_id'].options = qnames

        if qnames:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    populate_datasets({'name': 'initial'})
    if by_outcome:
        populate_outcomes({'name': 'initial'})
    populate_read_ids({'name': 'initial'})

    widgets['target'].observe(populate_datasets, names='value')
    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
    widgets['dataset'].observe(populate_outcomes, names='value')

    def plot(dataset, read_id, **kwargs):
        exp = experiment.Experiment(dataset)

        if by_outcome:
            als = exp.get_read_alignments(read_id, kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        fig = plot_read(als, exp.target_info, size_multiple=1.75, **kwargs)

        return fig

    interactive = ipywidgets.interactive(plot, **widgets)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['target', 'dataset', 'outcome', 'read_id']
    else:
        top_row_keys = ['target', 'dataset', 'read_id']

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row(['parsimonious', 'show_qualities']),
         widgets['zoom_in'],
         interactive.children[-1],
        ],
    )

    return layout

def make_length_plot(read_lengths, color, outcome_lengths=None):
    def plot_nonzero(ax, xs, ys, color, highlight):
        nonzero = ys.nonzero()
        if highlight:
            alpha = 0.95
            markersize = 2
        else:
            alpha = 0.7
            markersize = 0

        ax.plot(xs[nonzero], ys[nonzero], 'o', color=color, markersize=markersize, alpha=alpha)
        ax.plot(xs, ys, '-', color=color, alpha=0.3 * alpha)

    fig, ax = plt.subplots(figsize=(14, 5))

    ys = read_lengths
    xs = np.arange(len(ys))

    if outcome_lengths is None:
        all_color = color
        highlight = True
    else:
        all_color = 'black'
        highlight = False

    plot_nonzero(ax, xs, ys, all_color, highlight=highlight)
    ax.set_ylim(0, max(ys) * 1.05)

    if outcome_lengths is not None:
        ys = outcome_lengths
        xs = np.arange(len(ys))
        outcome_color = color
        plot_nonzero(ax, xs, ys, outcome_color, highlight=True)

    ax.set_xlabel('Length of read')
    ax.set_ylabel('Number of reads')
    ax.set_xlim(0, 8000)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    return fig
