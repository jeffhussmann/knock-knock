from pathlib import Path

import knock_knock.visualize

import ipywidgets

def explore(base_dir, by_outcome=False, target=None, experiment=None, clear_output=True, **kwargs):
    if target is None:
        target_names = sorted([t.name for t in target_info.get_all_targets(base_dir)])
    else:
        target_names = [target]

    default_filename = Path.cwd() / 'figure.png'

    widgets = {
        'target': ipywidgets.Select(options=target_names, value=target_names[0], layout=ipywidgets.Layout(height='200px')),
        'experiment': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='600px')),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
    }

    non_widgets = {
        'file_name': ipywidgets.Text(value=str(default_filename)),
        'save': ipywidgets.Button(description='Save snapshot'),
    }

    toggles = [
        ('parsimonious', False),
        ('relevant', True),
        ('ref_centric', True),
        ('draw_sequence', False),
        ('draw_qualities', False),
        ('draw_mismatches', True),
        ('draw_read_pair', False),
        ('force_left_aligned', False),
        ('center_on_primers', False),
        ('split_at_indels', False),
    ]
    for key, default_value in toggles:
        widgets[key] = ipywidgets.ToggleButton(value=kwargs.pop(key, default_value))

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    if experiment is None:
        conditions = {}
        exps = get_all_experiments(base_dir, as_dictionary=False)
    else:
        exps = [experiment]

    output = ipywidgets.Output()

    @output.capture()
    def populate_experiments(change):
        target = widgets['target'].value
        previous_value = widgets['experiment'].value
        datasets = sorted([(f'{exp.batch}: {exp.sample_name}', exp)
                           for exp in exps
                           if exp.target_info.name == target
                          ])
        widgets['experiment'].options = datasets

        if datasets:
            if previous_value in datasets:
                widgets['experiment'].value = previous_value
                populate_outcomes(None)
            else:
                widgets['experiment'].index = 0
        else:
            widgets['experiment'].value = None

    @output.capture()
    def populate_outcomes(change):
        previous_value = widgets['outcome'].value
        exp = widgets['experiment'].value
        if exp is None:
            return

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

    @output.capture()
    def populate_read_ids(change):
        exp = widgets['experiment'].value

        if exp is None:
            return

        if by_outcome:
            outcome = widgets['outcome'].value
            if outcome is None:
                qnames = []
            else:
                qnames = exp.outcome_query_names(outcome)[:200]
        else:
            qnames = list(islice(exp.query_names(), 200))

        widgets['read_id'].options = qnames

        if qnames:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    populate_experiments({'name': 'initial'})
    if by_outcome:
        populate_outcomes({'name': 'initial'})
    populate_read_ids({'name': 'initial'})

    widgets['target'].observe(populate_experiments, names='value')

    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
        widgets['experiment'].observe(populate_outcomes, names='value')
    else:
        widgets['experiment'].observe(populate_read_ids, names='value')

    @output.capture(clear_output=clear_output)
    def plot(experiment, read_id, **plot_kwargs):
        exp = experiment

        if exp is None:
            return

        if by_outcome:
            als = exp.get_read_alignments(read_id, outcome=plot_kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        print(als[0].query_name)
        print(als[0].get_forward_sequence())

        l = exp.categorizer(als, exp.target_info, mode=exp.layout_mode)
        info = l.categorize()
        
        if widgets['relevant'].value:
            als = l.relevant_alignments

        inferred_amplicon_length = l.inferred_amplicon_length

        plot_kwargs.setdefault('features_to_show', exp.target_info.features_to_show)

        diagram = knock_knock.visualize.ReadDiagram(als, exp.target_info,
                                        inferred_amplicon_length=inferred_amplicon_length,
                                        **plot_kwargs)
        fig = diagram.fig

        fig.axes[0].set_title(' '.join((l.query_name,) + info[:3]))

        return diagram.fig

    all_kwargs = {**{k: ipywidgets.fixed(v) for k, v in kwargs.items()}, **widgets}

    interactive = ipywidgets.interactive(plot, **all_kwargs)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] if k in widgets else non_widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['target', 'experiment', 'outcome', 'read_id']
    else:
        top_row_keys = ['target', 'experiment', 'read_id']

    @output.capture(clear_output=False)
    def save(_):
        fig = interactive.result
        fn = non_widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

    non_widgets['save'].on_click(save)

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row([k for k, d in toggles]),
         make_row(['file_name', 'save']),
         interactive.children[-1],
         output,
        ],
    )

    return layout
