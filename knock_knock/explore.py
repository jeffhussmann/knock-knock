from itertools import islice
from pathlib import Path

import knock_knock.visualize.architecture
import knock_knock.experiment

import ipywidgets
from ipywidgets import Select, Layout, Text, Textarea, Button, ToggleButton, ToggleButtons, Label, HBox, VBox

class Explorer:
    def __init__(self,
                 by_outcome=True,
                 read_type=None,
                 **plot_kwargs,
                ):
        
        self.by_outcome = by_outcome
        self.plot_kwargs = plot_kwargs
        self.read_type = read_type

        self.output = ipywidgets.Output()

        default_filename = Path.cwd() / 'figure.png'

        self.widgets = {
            'category': Select(
                description='Category:',
                options=[],
                continuous_update=False,
                layout=Layout(height='200px', width='300px'),
            ),
            'subcategory': Select(
                description='Subcategory:',
                options=[],
                continuous_update=False,
                layout=Layout(height='200px', width='300px'),
            ),
            'read_id': Select(
                description='Read:',
                options=[],
                layout=Layout(height='200px', width='500px'),
            ),
        }

        selection_widget_keys = self.set_up_read_selection_widgets()

        self.non_widgets = {
            'file_name': Text(value=str(default_filename)),
            'save': Button(description='Save snapshot'),
            'read_details': Textarea(description='Read details:', layout=Layout(height='200px', width='1000px')),
            'alignments_to_show': ToggleButtons(
                options=['all', 'parsimonious', 'relevant'],
                value='relevant',
                style=dict(
                    button_width = '120px',
                ),
                layout={'width': '400px'},
            ),
        }

        draw_button_info = [
            ('draw_qualities', False, 'qualities'),
            ('draw_mismatches', True, 'mismatches'),
        ]

        self.draw_button_names = {name for name, *_ in draw_button_info}

        for key, default_value, label in draw_button_info:
            value = self.plot_kwargs.pop(key, default_value)
            self.non_widgets[key] = ToggleButton(value=value, description=label)

        all_kwargs = {
            **{k: ipywidgets.fixed(v) for k, v in self.plot_kwargs.items()},
            **{k: v for k, v in self.widgets.items() if k not in {'category', 'subcategory', 'condition', 'replicate'}},
        }

        self.interactive = ipywidgets.interactive(self.plot, **all_kwargs)
        self.interactive.update()

        self.non_widgets['alignments_to_show'].observe(self.interactive.update, names='value')

        for key in self.draw_button_names:
            self.non_widgets[key].observe(self.interactive.update, names='value')

        def resolve_key(key):
            return self.widgets[key] if key in self.widgets else self.non_widgets[key]

        def make_collection(keys, orientation, description=None):
            Box = {'row': HBox, 'col': VBox}[orientation]

            collection = Box([resolve_key(key) for key in keys],
                             layout={'padding': '1px 1px 5px 1px'},
                            )

            if description is not None:
                collection = HBox([Label(description), collection])

            return collection

        def make_row(keys, description=None):
            return make_collection(keys, 'row', description=description)

        def make_col(keys, description=None):
            return make_collection(keys, 'col', description=description)

        self.non_widgets['save'].on_click(self.save)

        self.layout = VBox([
            make_row(selection_widget_keys),
            HBox([
                make_row([
                    'alignments_to_show',
                ],
                description='Alignments to show:',
                ),
                make_row([
                    'draw_qualities',
                    'draw_mismatches',
                ],
                description='Draw:',
                ),
            ]),
            make_row(['file_name', 'save']),
            self.interactive.children[-1],
            make_row(['read_details']),
            self.output,
            ],
        )

    def get_current_outcome(self):
        return (self.widgets['category'].value, self.widgets['subcategory'].value)

    def populate_categories(self, change):
        with self.output:
            exp = self.get_current_experiment()
            if exp is None:
                return

            previous_value = self.widgets['category'].value

            categories = exp.outcome_counts(level='category').sort_values(ascending=False).index.values
            self.widgets['category'].options = categories

            if len(categories) > 0:
                if previous_value in categories:
                    self.widgets['category'].value = previous_value
                    self.populate_read_ids(None)
                else:
                    self.widgets['category'].value = self.widgets['category'].options[0]
            else:
                self.widgets['category'].value = None

    def populate_subcategories(self, change):
        with self.output:
            exp = self.get_current_experiment()
            if exp is None:
                return

            previous_value = self.widgets['subcategory'].value

            category = self.widgets['category'].value

            subcategories = [subcat for cat, subcat in exp.subcategories_by_frequency if cat == category]
            self.widgets['subcategory'].options = subcategories

            if len(subcategories) > 0:
                if previous_value in subcategories:
                    self.widgets['subcategory'].value = previous_value
                    self.populate_read_ids(None)
                else:
                    self.widgets['subcategory'].value = self.widgets['subcategory'].options[0]
            else:
                self.widgets['subcategory'].value = None

    def populate_read_ids(self, change):
        with self.output:
            exp = self.get_current_experiment()

            if exp is None:
                return

            if self.by_outcome:
                outcome = self.get_current_outcome()
                if outcome is None:
                    qnames = []
                else:
                    qnames = exp.outcome_query_names(outcome)[:200]
            else:
                qnames = list(islice(exp.query_names(read_type=self.read_type), 200))

            self.widgets['read_id'].options = qnames

            if qnames:
                self.widgets['read_id'].value = qnames[0]
                self.widgets['read_id'].index = 0
            else:
                self.widgets['read_id'].value = None

    def get_alignments(self, exp, read_id):
        if self.by_outcome:
            als = exp.get_read_alignments(read_id, outcome=self.get_current_outcome())
        else:
            als = exp.get_read_alignments(read_id, read_type=self.read_type)

        if als is None or len(als) == 0:
            raise ValueError

        return als

    def plot(self, read_id, **plot_kwargs):
        ''' Note: executing %matplotlib inline breaks this by setting
        the value returned by matplotlib.get_backend() to 'inline', prevening
        ipywidgets' show_inline_matplotlib_plots from working.
        '''

        with self.output:
            exp = self.get_current_experiment()

            if exp is None:
                return None

            als = self.get_alignments(exp, read_id)

            if als is None:
                return None

            if isinstance(als, dict):
                read_details = [
                    f'exp: {exp.identifier}',
                    f'{als["R1"][0].query_name}:',
                    f'R1 sequence: {als["R1"][0].get_forward_sequence()}',
                    f'R2 sequence: {als["R2"][0].get_forward_sequence()}',
                ]

                if self.by_outcome:
                    architecture = exp.no_overlap_pair_categorizer(als, exp.editing_strategy)
            else:
                read_details = [
                    f'exp: {exp.identifier}',
                    f'{als[0].query_name}{f" ({read_id})" if read_id != als[0].query_name else ""}:',
                    f'sequence: {als[0].get_forward_sequence()}',
                ]

                if self.by_outcome:
                    architecture = exp.categorizer(als,
                                             exp.editing_strategy,
                                             platform=exp.platform,
                                             error_corrected=exp.has_UMIs,
                                            )

            if self.by_outcome:
                architecture.categorize()

                inferred_amplicon_length = architecture.inferred_amplicon_length

                read_details = read_details[:2] + [
                    f'  category: {architecture.category}',
                    f'  subcategory: {architecture.subcategory}',
                    f'  details: {architecture.details}',
                    f'  inferred amplicon length: {inferred_amplicon_length}', 
                ] + read_details[2:]

                if self.non_widgets['alignments_to_show'].value == 'relevant':
                    plot_kwargs['relevant'] = True
                else:
                    plot_kwargs['relevant'] = False

            else:
                inferred_amplicon_length = None

            if self.non_widgets['alignments_to_show'].value == 'parsimonious':
                plot_kwargs['parsimonious'] = True

            for k in self.draw_button_names:
                plot_kwargs[k] = self.non_widgets[k].value

            if self.by_outcome:
                diagram = architecture.plot(**plot_kwargs)
            else:
                diagram = knock_knock.visualize.architecture.ReadDiagram(als,
                                                                         exp.editing_strategy,
                                                                         inferred_amplicon_length=inferred_amplicon_length,
                                                                         title='',
                                                                         architecture=architecture,
                                                                         **plot_kwargs,
                                                                        )

            read_details = '\n'.join(read_details)
            self.non_widgets['read_details'].value = read_details

            return diagram.fig

    def save(self, _):
        fig = self.interactive.result
        fn = self.non_widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

class BaseDirExplorer(Explorer):
    def __init__(self, base_dir, by_outcome=True, **plot_kwargs):
        self.experiments = knock_knock.experiment.get_all_experiments(base_dir)
        self.batch_names = sorted({batch_name for batch_name, _ in self.experiments})
        super().__init__(by_outcome, **plot_kwargs)

    def get_current_experiment(self):
        return self.experiments[self.widgets['batch'].value, self.widgets['sample'].value]

    def populate_samples(self, change):
        current_batch = self.widgets['batch'].value
        samples = sorted({sample for batch_name, sample in self.experiments if batch_name == current_batch})
        self.widgets['sample'].options = samples
        self.widgets['sample'].value = samples[0]

    def set_up_read_selection_widgets(self):
        self.widgets.update({
            'batch': Select(options=self.batch_names, continuous_update=False, layout=Layout(height='200px', width='300px')),
            'sample': Select(options=[], continuous_update=False, layout=Layout(height='200px', width='300px')),
        })

        self.populate_samples({'name': 'initial'})
        self.widgets['batch'].observe(self.populate_samples, names='value')

        if self.by_outcome:
            self.populate_categories({'name': 'initial'})
            self.populate_subcategories({'name': 'initial'})
            self.widgets['sample'].observe(self.populate_categories, names='value')
            self.widgets['category'].observe(self.populate_subcategories, names='value')
            self.widgets['subcategory'].observe(self.populate_read_ids, names='value')
            selection_widget_keys = ['batch', 'sample', 'category', 'subcategory', 'read_id']
        else:
            self.widgets['sample'].observe(self.populate_read_ids, names='value')
            selection_widget_keys = ['batch', 'sample', 'read_id']

        self.populate_read_ids({'name': 'initial'})

        return selection_widget_keys

class SingleExperimentExplorer(Explorer):
    def __init__(self, experiment, by_outcome=True, **plot_kwargs):
        self.experiment = experiment
        super().__init__(by_outcome, **plot_kwargs)

    def get_current_experiment(self):
        return self.experiment

    def set_up_read_selection_widgets(self):
        if self.by_outcome:
            self.populate_categories({'name': 'initial'})
            self.populate_subcategories({'name': 'initial'})
            self.widgets['category'].observe(self.populate_subcategories, names='value')
            self.widgets['subcategory'].observe(self.populate_read_ids, names='value')
            selection_widget_keys = ['category', 'subcategory', 'read_id']
        else:
            selection_widget_keys = ['read_id']

        self.populate_read_ids({'name': 'initial'})

        return selection_widget_keys

class ArrayedGroupExplorer(Explorer):
    def __init__(self,
                 group,
                 initial_condition=None,
                 by_outcome=True,
                 **plot_kwargs,
                ):
        self.group = group

        if initial_condition is None and len(self.group.conditions) > 0:
            initial_condition = self.group.conditions[0]

        self.initial_condition = initial_condition

        self.experiments = {}

        super().__init__(by_outcome, **plot_kwargs)

    def populate_replicates(self, change):
        with self.output:
            if len(self.group.conditions) > 0:
                condition = self.widgets['condition'].value
            else:
                condition = tuple()

            exps = self.group.condition_replicates(condition)

            self.widgets['replicate'].options = [(exp.description['replicate'], exp) for exp in exps]
            self.widgets['replicate'].index = 0

    def get_current_experiment(self):
        experiment = self.widgets['replicate'].value
        return experiment

    def set_up_read_selection_widgets(self):

        selection_widget_keys = []

        if len(self.group.conditions) > 0:
            condition_options = [(', '.join(c) if isinstance(c, tuple) else c, c) for c in self.group.conditions] 
            self.widgets.update({
                'condition': Select(
                    description='Condition:',
                    options=condition_options,
                    value=self.initial_condition,
                    layout=Layout(height='200px', width='300px'),
                ),
            })
            self.widgets['condition'].observe(self.populate_replicates, names='value')
            selection_widget_keys.append('condition')

        self.widgets.update({
            'replicate': Select(
                description='Replicate:',
                options=[],
                layout=Layout(height='200px', width='150px'),
            ),
        })

        self.populate_replicates({'name': 'initial'})

        selection_widget_keys.append('replicate')
        
        if self.by_outcome:
            self.populate_categories({'name': 'initial'})
            self.populate_subcategories({'name': 'initial'})

            self.widgets['replicate'].observe(self.populate_categories, names='value')
            self.widgets['category'].observe(self.populate_subcategories, names='value')
            self.widgets['subcategory'].observe(self.populate_read_ids, names='value')
            selection_widget_keys.extend(['category', 'subcategory'])
        else:
            self.widgets['replicate'].observe(self.populate_read_ids, names='value')

        selection_widget_keys.append('read_id')

        self.populate_read_ids({'name': 'initial'})

        return selection_widget_keys
