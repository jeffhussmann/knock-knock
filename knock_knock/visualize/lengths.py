from textwrap import dedent

import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import pandas as pd

def plot_outcome_stratified_lengths(outcome_stratified_lengths,
                                    categorizer,
                                    editing_strategy,
                                    level='subcategory',
                                    length_ranges=None,
                                    x_lims=None,
                                    min_total_to_label=0.1,
                                    zoom_factor=0.1,
                                    smooth_window=0,
                                    title=None,
                                    x_tick_multiple=100,
                                    draw_instructions=False,
                                    only_relevant=False,
                                    truncate_to_max_observed_length=False,
                                   ):
    '''
    outcome_stratified_lengths: knock_knock.lengths.OutcomeStratifiedLengths
    categorizer: knock_knock.architecture.Categorizer
    '''

    if truncate_to_max_observed_length:
        outcome_stratified_lengths = outcome_stratified_lengths.truncate_to_max_observed_length(only_relevant=only_relevant)

    lengths_df = outcome_stratified_lengths.lengths_df(level=level, only_relevant=only_relevant)

    total_reads = outcome_stratified_lengths.total_reads(only_relevant=only_relevant)
    highest_points = outcome_stratified_lengths.highest_points(level=level, smooth_window=smooth_window, only_relevant=only_relevant)
    outcome_to_color = outcome_stratified_lengths.outcome_to_color(smooth_window=smooth_window, only_relevant=only_relevant)

    if total_reads == 0:
        return

    if x_lims is None:
        x_lims = (min(lengths_df.columns), max(lengths_df.columns))

    panel_groups = []

    remaining_outcomes = sorted(highest_points)

    while len(remaining_outcomes) > 0:
        panel_max = max(highest_points[outcome] for outcome in remaining_outcomes)

        outcomes_in_panel = []

        for outcome in remaining_outcomes:
            highest_point = highest_points[outcome]
            if panel_max * zoom_factor < highest_point <= panel_max:
                outcomes_in_panel.append(outcome)
                
        panel_groups.append((panel_max, outcomes_in_panel))

        remaining_outcomes = [outcome for outcome in remaining_outcomes if outcome not in outcomes_in_panel]

    num_panels = len(panel_groups)

    fig, axs = plt.subplots(num_panels, 1, figsize=(14, 6 * num_panels), gridspec_kw=dict(hspace=0.12))

    if num_panels == 1:
        # Want to be able to treat axs as a 1D array.
        axs = [axs]

    ax = axs[0]

    if title is not None:
        ax.annotate(title,
                    xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 40), textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=14,
                   )

    y_maxes = []

    listed_order = sorted(lengths_df.index, key=categorizer.order)

    non_highlight_color = 'grey'

    for panel_i, (ax, (y_max, group)) in enumerate(zip(axs, panel_groups)):
        high_enough_to_show = []

        for outcome in listed_order:
            lengths = lengths_df.loc[outcome]
            total_fraction = lengths.sum() / total_reads
            window = smooth_window * 2 + 1
            smoothed_lengths = pd.Series(lengths).rolling(window=window, center=True, min_periods=1).sum()
            ys = smoothed_lengths / total_reads

            sanitized_string = categorizer.outcome_to_sanitized_string(outcome)

            if outcome in group:
                gid = f'line_highlighted_{sanitized_string}_{panel_i}'
                color = outcome_to_color[outcome]
                alpha = 1
            else:
                color = non_highlight_color
                # At higher zoom levels, fade the grey lines more to avoid clutter.
                if panel_i == 0:
                    alpha = 0.6
                    gid = f'line_nonhighlighted_6_{sanitized_string}_{panel_i}'
                elif panel_i == 1:
                    alpha = 0.3
                    gid = f'line_nonhighlighted_3_{sanitized_string}_{panel_i}'
                else:
                    alpha = 0.05
                    gid = f'line_nonhighlighted_05_{sanitized_string}_{panel_i}'

            if isinstance(outcome, str):
                outcome_for_label = outcome
            else:
                category, subcategory = outcome
                outcome_for_label = f'{category}: {subcategory}'

            if total_fraction * 100 > min_total_to_label:
                high_enough_to_show.append(outcome)
                label = f'{total_fraction:6.2%} {outcome_for_label}'
            else:
                label = None

            ax.plot(ys * 100, label=label, color=color, alpha=alpha, gid=gid)

            if length_ranges is not None:
                if outcome in group:
                    for start, end in length_ranges(outcome=outcome):
                        ax.axvspan(start - 0.5, end + 0.5,
                                   gid=f'length_range_{sanitized_string}_{start}_{end}',
                                   alpha=0.0,
                                   facecolor='white',
                                   edgecolor='black',
                                   zorder=100,
                                  )

        legend_cols = int(np.ceil(len(high_enough_to_show) / 18))

        legend = ax.legend(bbox_to_anchor=(1.05, 1),
                            loc='upper left',
                            prop=dict(family='monospace', size=9),
                            framealpha=0.3,
                            ncol=legend_cols,
                            )

        for outcome, line in zip(high_enough_to_show, legend.get_lines()):
            if line.get_color() != non_highlight_color:
                line.set_linewidth(5)
                sanitized_string = categorizer.outcome_to_sanitized_string(outcome)
                line.set_gid(f'outcome_{sanitized_string}')

        ax.set_ylim(0, y_max * 1.05)
        y_maxes.append(y_max)

    for panel_i, ax in enumerate(axs):
        main_ticks = list(range(outcome_stratified_lengths.min_relevant_length, outcome_stratified_lengths.max_relevant_length, x_tick_multiple))
        main_tick_labels = [str(x) for x in main_ticks]

        extra_ticks = [
            outcome_stratified_lengths.max_relevant_length,
            outcome_stratified_lengths.length_to_store_unknown,
        ]
        extra_tick_labels = [
            r'$\geq$' + f'{outcome_stratified_lengths.max_relevant_length}',
            '?',
        ]

        ax.set_xticks(main_ticks + extra_ticks, labels=main_tick_labels + extra_tick_labels)

        ax.set_xlim(*x_lims)
        ax.set_ylabel('Percentage of reads', size=12)

        if len(editing_strategy.primers) == 0:
            x_label = 'Read length change from unedited'
        else:
            x_label = 'Amplicon length'

        ax.set_xlabel(x_label, size=12)

        if panel_i == 0:
            for i, (name, length) in enumerate(editing_strategy.expected_lengths.items()):
                ax.axvline(length, color='black', alpha=0.2)

                y = 1 + 0.05 * i
                ax.annotate(name,
                            xy=(length, y), xycoords=('data', 'axes fraction'),
                            xytext=(0, 1), textcoords='offset points',
                            ha='center', va='bottom',
                            size=10,
                            )

    def draw_inset_guide(fig, top_ax, bottom_ax, bottom_y_max, panel_i):
        params_dict = {
            'top': {
                'offset': 0.04,
                'width': 0.007,
                'transform': top_ax.get_yaxis_transform(),
                'ax': top_ax,
                'y': bottom_y_max,
            },
            'bottom': {
                'offset': 0.01,
                'width': 0.01,
                'transform': bottom_ax.transAxes,
                'ax': bottom_ax,
                'y': 1,
            },
        }

        for which, params in params_dict.items():
            start = 1 + params['offset']
            end = start + params['width']
            y = params['y']
            transform = params['transform']
            ax = params['ax']

            params['start'] = start
            params['end'] = end

            params['top_corner'] = [end, y]
            params['bottom_corner'] = [end, 0]

            ax.plot([start, end, end, start],
                    [y, y, 0, 0],
                    transform=transform,
                    clip_on=False,
                    color='black',
                    linewidth=3,
                    gid=f'zoom_toggle_{which}_{panel_i}',
                    )

            ax.fill([start, end, end, start],
                    [y, y, 0, 0],
                    transform=transform,
                    clip_on=False,
                    color='white',
                    gid=f'zoom_toggle_{which}_{panel_i}',
                    )

            if which == 'top' and panel_i in [0, 1]:
                if panel_i == 0:
                    bracket_message = 'Click this bracket to explore lower-frequency outcomes\nby zooming in on the bracketed range. Click again to close.'
                else:
                    bracket_message = 'Click this bracket to zoom in further.'

                if draw_instructions:
                    ax.annotate(bracket_message,
                                xy=(end, y / 2),
                                xycoords=transform,
                                xytext=(10, 0),
                                textcoords='offset points',
                                ha='left',
                                va='center',
                                size=12,
                                gid=f'help_message_bracket_{panel_i + 1}',
                               )

        inverted_fig_tranform = fig.transFigure.inverted().transform    

        for which, top_coords, bottom_coords in (('top', params_dict['top']['top_corner'], params_dict['bottom']['top_corner']),
                                                 ('bottom', params_dict['top']['bottom_corner'], params_dict['bottom']['bottom_corner']),
                                                ):
            top_in_fig = inverted_fig_tranform(params_dict['top']['transform'].transform((top_coords)))
            bottom_in_fig = inverted_fig_tranform(params_dict['bottom']['transform'].transform((bottom_coords)))

            xs = [top_in_fig[0], bottom_in_fig[0]]
            ys = [top_in_fig[1], bottom_in_fig[1]]
            line = matplotlib.lines.Line2D(xs, ys,
                                           transform=fig.transFigure,
                                           clip_on=False,
                                           linestyle='--',
                                           color='black',
                                           alpha=0.5,
                                           gid=f'zoom_dotted_line_{panel_i}_{which}',
                                          )
            fig.lines.append(line)

    for panel_i, (y_max, top_ax, bottom_ax) in enumerate(zip(y_maxes[1:], axs, axs[1:])):
        draw_inset_guide(fig, top_ax, bottom_ax, y_max, panel_i)

    if draw_instructions:
        top_ax = axs[0]

        help_box_width = 0.04
        help_box_height = 0.1
        help_box_y = 1.05

        top_ax.annotate('?',
                        xy=(1 - 0.5 * help_box_width, help_box_y + 0.5 * help_box_height),
                        xycoords='axes fraction',
                        ha='center',
                        va='center',
                        size=22,
                        weight='bold',
                        gid='help_toggle_question_mark',
                       )

        help_box = matplotlib.patches.Rectangle((1 - help_box_width, help_box_y),
                                                help_box_width,
                                                help_box_height,
                                                transform=top_ax.transAxes,
                                                clip_on=False,
                                                color='black',
                                                alpha=0.2,
                                                gid='help_toggle',
                                               )
        top_ax.add_patch(help_box)

        legend_message = dedent('''\
            Click the colored line next to an outcome category
            in the legend to activate that category. Once
            activated, hovering the cursor over the plot will
            show an example diagram of the activated category
            of the length that the cursor is over. Press
            Esc when done to deactivate the category.'''
        )

        top_ax.annotate(legend_message,
                        xy=(1, 0.95),
                        xycoords='axes fraction',
                        xytext=(-10, 0),
                        textcoords='offset points',
                        ha='right',
                        va='top',
                        size=12,
                        gid='help_message_legend',
                       )

    for ax in axs:
        ax.tick_params(axis='y', which='both', left=True, right=True)

    return fig