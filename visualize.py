import matplotlib
matplotlib.use('Agg', warn=False)

import itertools
import subprocess
import pysam
import numpy as np
import yaml
import matplotlib.pyplot as plt
import shutil
import matplotlib.ticker
import bokeh.palettes
import functools
import tempfile
import ipywidgets

from pathlib import Path
from collections import Counter

import Sequencing.fastq as fastq
import Sequencing.utilities as utilities
import Sequencing.interval as interval
import Sequencing.sam as sam
import Sequencing.visualize_structure as visualize_structure

import pacbio_experiment
import target_info
import layout

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

def plot_read(dataset,
              read_id,
              outcome=None,
              parsimonious=False,
              show_qualities=False,
              x_lims=None,
              size_multiple=1,
             ):

    exp = pacbio_experiment.PacbioExperiment(dataset)

    if outcome is not None:
        bam_fn = exp.outcome_fns(outcome)['bam_by_name']
    else:
        bam_fn = exp.fns['bam_by_name']
    
    features = exp.target_info.features

    bam_fh = pysam.AlignmentFile(bam_fn)
    colors = {name: 'C{0}'.format(i) for i, name in enumerate(bam_fh.references)}

    fig, ax = plt.subplots(figsize=(12, 4))

    read_groups = utilities.group_by(bam_fh, lambda r: r.query_name)
    try:
        if isinstance(read_id, int):
            name, group = next(itertools.islice(read_groups, read_id, read_id + 1))
        else:
            name, group = next(itertools.dropwhile(lambda t: t[0] != read_id, read_groups))
    except StopIteration:
        plt.close(fig)
        return None
    
    if not all(al.is_unmapped for al in group):
        layout_info = {'alignments': {'all': group}}
        layout.identify_flanking_target_alignments(layout_info, exp.target_info)
        if layout_info['strand'] == '-':
            reverse_complement = True
        else:
            reverse_complement = False
    else:
        reverse_complement = False
    
    per_rname = 0.06
    gap_between_als = 0.06 * 0.2
    arrow_height = 0.005
    arrow_width = 0.01
    
    max_y = gap_between_als
    
    if parsimonious:
        group = interval.make_parsimoninous(group)
        
    query_name = group[0].query_name
    query_length = group[0].query_length
    quals = group[0].query_qualities
    
    kwargs = {'linewidth': 2, 'color': 'black'}
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

    if all(al.is_unmapped for al in group):
        by_reference_name = []
    else:
        group = sorted(group, key=lambda al: (al.reference_name, sam.query_interval(al)))
        by_reference_name = list(utilities.group_by(group, lambda al: al.reference_name))
    
    rname_starts = np.cumsum([1] + [len(als) for n, als in by_reference_name])
    offsets = {name: start for (name, als), start in zip(by_reference_name, rname_starts)}
    
    for reference_name, alignments in by_reference_name:
        if reverse_complement:
            for alignment in alignments:
                alignment.is_reverse = not alignment.is_reverse

        alignments = alignments[:10]
        
        offset = offsets[reference_name]
        color = colors[reference_name]

        average_y = (offset  + 0.5 * (len(alignments) - 1)) * gap_between_als
        ax.annotate(reference_name,
                    xy=(1, average_y),
                    xycoords=('axes fraction', 'data'),
                    xytext=(15, 0),
                    textcoords='offset points',
                    color=color,
                    ha='left',
                    va='center',
                   )

        for i, alignment in enumerate(alignments):
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
                    
                    if length <= 2:
                        height = 0.0015
                        indel_xs = [centered_at, centered_at, centered_at]
                        indel_ys = [y, y + height, y]
                    else:
                        width = query_length * 0.001
                        height = 0.006
                        indel_xs = [centered_at - width, centered_at - 0.5 * length, centered_at + 0.5 * length, centered_at + width]
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
                
            ax.plot(arrow_xs, arrow_ys, clip_on=False, **kwargs)

            target = exp.target_info.target
            donor = exp.target_info.donor
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
                if reference_name != feature_reference:
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

    ax.set_title('{0}: {1}'.format(dataset, query_name), y=1.2)
        
    ax.set_ylim(-0.2 * max_y, 1.1 * max_y)
    ax.set_xlim(-0.02 * query_length, 1.02 * query_length)
    ax.set_yticks([])
    
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_alpha(0.1)
    for edge in 'left', 'top', 'right':
        ax.spines[edge].set_color('none')
        
    ax.tick_params(pad=14)
    fig.set_size_inches((12 * size_multiple, 4 * max_y / 0.15 * size_multiple))
    
    bam_fh.close()

    if show_qualities:
        ax.plot(np.array(quals) * max_y / 93, color='black', alpha=0.5)
        
    if x_lims is not None:
        ax.set_xlim(*x_lims)
        
    return fig

def interactive():
    target_names = [t.name for t in target_info.get_all_targets()]

    widgets = dict(
        target = ipywidgets.Select(options=target_names),
        dataset = ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        read_id = ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='400px')),
        parsimonious = ipywidgets.ToggleButton(value=True),
        show_qualities = ipywidgets.ToggleButton(value=False),
    )

    for k, v in widgets.items():
        v.description = k

    exps = pacbio_experiment.get_all_experiments()

    def populate_datasets(change):
        target = widgets['target'].value
        previous_value = widgets['dataset'].value
        datasets = sorted([exp.name for exp in exps if exp.target_info.name == target])
        widgets['dataset'].options = datasets
        
        if len(datasets) > 0:
            if previous_value in datasets:
                widgets['dataset'].value = previous_value
                populate_outcomes(None)
            else:
                widgets['dataset'].value = datasets[0]
        else:
            widgets['dataset'].value = None

    def populate_read_ids(change):
        dataset = widgets['dataset'].value
        exp = pacbio_experiment.PacbioExperiment(dataset)
        
        qnames = list(itertools.islice(exp.query_names(), 200))
        
        widgets['read_id'].options = qnames
        
        if len(qnames) > 0:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    populate_datasets({'name': 'initial'})
    populate_read_ids({'name': 'initial'})

    widgets['target'].observe(populate_datasets, names='value')
    widgets['dataset'].observe(populate_read_ids, names='value')

    figure = ipywidgets.interactive(plot_read,
                                    size_multiple=ipywidgets.fixed(1.75),
                                    outcome=ipywidgets.fixed(None),
                                    x_lims=ipywidgets.fixed(None),
                                    **widgets,
                                   )
    figure.update()

    layout = ipywidgets.VBox(
        [ipywidgets.HBox([widgets['target'], widgets['dataset'], widgets['read_id']]),
         ipywidgets.HBox([widgets['parsimonious'], widgets['show_qualities']]),
         figure.children[-1],
        ],
    )

    return layout

def interactive_by_outcome():
    target_names = [t.name for t in target_info.get_all_targets()]

    widgets = dict(
        target = ipywidgets.Select(options=target_names, value=target_names[0]),
        dataset = ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        read_id = ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='400px')),
        parsimonious = ipywidgets.ToggleButton(value=True),
        show_qualities = ipywidgets.ToggleButton(value=False),
        outcome = ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
    )

    exps = pacbio_experiment.get_all_experiments()

    def populate_datasets(change):
        target = widgets['target'].value
        previous_value = widgets['dataset'].value
        datasets = sorted([exp.name for exp in exps if exp.target_info.name == target])
        widgets['dataset'].options = datasets

        if len(datasets) > 0:
            if previous_value in datasets:
                widgets['dataset'].value = previous_value
                populate_outcomes(None)
            else:
                widgets['dataset'].value = datasets[0]
        else:
            widgets['dataset'].value = None

    def populate_outcomes(change):
        previous_value = widgets['outcome'].value
        exp = pacbio_experiment.PacbioExperiment(widgets['dataset'].value)
        outcomes = exp.outcomes
        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in outcomes]
        if len(outcomes) > 0:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    def populate_read_ids(change):
        exp = pacbio_experiment.PacbioExperiment(widgets['dataset'].value)
        qnames = exp.outcome_query_names(widgets['outcome'].value)
        widgets['read_id'].options = qnames
        if len(qnames) > 0:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    populate_datasets({'name': 'initial'})
    populate_outcomes({'name': 'initial'})
    populate_read_ids({'name': 'initial'})

    widgets['target'].observe(populate_datasets, names='value')
    widgets['dataset'].observe(populate_outcomes, names='value')
    widgets['outcome'].observe(populate_read_ids, names='value')

    figure = ipywidgets.interactive(plot_read,
                                    size_multiple=ipywidgets.fixed(1.75),
                                    x_lims=ipywidgets.fixed(None),
                                    **widgets,
                                   )
    figure.update()

    layout = ipywidgets.VBox(
        [ipywidgets.HBox([widgets['target'], widgets['dataset'], widgets['outcome'], widgets['read_id']]),
         ipywidgets.HBox([widgets['parsimonious'], widgets['show_qualities']]),
         figure.children[-1],
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

    plot_nonzero(ax, xs, ys, color, highlight=highlight)
    ax.set_ylim(0, max(ys) * 1.05)

    if outcome_lengths is not None:
        ys = outcome_lengths
        xs = np.arange(len(ys))
        outcome_color = color
        plot_nonzero(ax, xs, ys, color=color, highlight=True)

    ax.set_xlabel('Length of read')
    ax.set_ylabel('Numbr of reads')
    ax.set_xlim(0, 8000)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    return fig
            
def make_outcome_text_alignments(target, dataset, num_examples=10):
    outcomes = pacbio.get_outcomes(target, dataset)
    for outcome in outcomes:
        fns = pacbio.make_fns(target, dataset, outcome)
        visualize_structure.visualize_bam_alignments(fns['bam_by_name'], fns['ref_fasta'], fns['text'], num_examples)
