from itertools import islice
from pathlib import Path

import knock_knock.visualize
import knock_knock.target_info
import knock_knock.experiment
import knock_knock.layout

import ipywidgets
from ipywidgets import Select, Layout, Text, Button, ToggleButton, Label, HBox, VBox

class Explorer:
    def __init__(self,
                 by_outcome=True,
                 **plot_kwargs,
                ):
        
        self.by_outcome = by_outcome
        self.plot_kwargs = plot_kwargs

        self.output = ipywidgets.Output()

        default_filename = Path.cwd() / 'figure.png'

        self.widgets = {
            'category': Select(options=[], continuous_update=False, layout=Layout(height='200px', width='300px')),
            'subcategory': Select(options=[], continuous_update=False, layout=Layout(height='200px', width='300px')),
            'read_id': Select(options=[], layout=Layout(height='200px', width='500px')),
        }

        selection_widget_keys = self.set_up_read_selection_widgets()

        self.non_widgets = {
            'file_name': Text(value=str(default_filename)),
            'save': Button(description='Save snapshot'),
            'read_details': ipywidgets.Textarea(description='Read details:', layout=Layout(height='200px', width='1000px')),
            'alignments_to_show': ipywidgets.ToggleButtons(
                options=['all', 'parsimonious', 'relevant'],
                value='relevant',
                style={'button_width': '80px'},
                layout={'width': '150px'},
            ),
            'alignment_registration': ipywidgets.ToggleButtons(
                options=['centered on primers', 'left', 'right'],
                value='centered on primers',
                style={'button_width': '120px'},
                layout={'width': '190px'},
            ),
        }

        draw_button_info = [
            ('ref_centric', True, 'target and donor'),
            ('draw_sequence', False, 'sequence'),
            ('draw_qualities', False, 'qualities'),
            ('draw_mismatches', True, 'mismatches'),
        ]

        self.draw_buttons = {}
        for key, default_value, label in draw_button_info:
            value = self.plot_kwargs.pop(key, default_value)
            self.draw_buttons[key] = ToggleButton(value=value, description=label)

        toggles = [
            ('split_at_indels', False),
        ]
        for key, default_value in toggles:
            value = self.plot_kwargs.pop(key, default_value)
            self.widgets[key] = ToggleButton(value=value)

        # For some reason, the target widget doesn't get a label without this.
        for k, v in self.widgets.items():
            v.description = k

        all_kwargs = {
            **{k: ipywidgets.fixed(v) for k, v in self.plot_kwargs.items()},
            **self.widgets,
        }

        self.interactive = ipywidgets.interactive(self.plot, **all_kwargs)
        self.interactive.update()

        self.non_widgets['alignments_to_show'].observe(self.interactive.update, names='value')
        self.non_widgets['alignment_registration'].observe(self.interactive.update, names='value')

        for key in self.draw_buttons:
            self.draw_buttons[key].observe(self.interactive.update, names='value')

        def make_row(keys):
            return HBox([self.widgets[k] if k in self.widgets else self.non_widgets[k] for k in keys])

        def make_col(keys):
            return VBox([self.widgets[k] if k in self.widgets else self.non_widgets[k] for k in keys])

        self.non_widgets['save'].on_click(self.save)

        self.layout = ipywidgets.VBox(
            [make_row(selection_widget_keys),
             HBox([HBox([Label('Alignments to show:'), self.non_widgets['alignments_to_show']]),
                   HBox([Label('Alignment registration:'), self.non_widgets['alignment_registration']]),
                   HBox([Label('Draw:'), VBox([self.draw_buttons[k] for k in ['draw_sequence', 'ref_centric', 'draw_qualities', 'draw_mismatches']])]),
                  ]),
             make_row([k for k, d in toggles]),
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

            categories = exp.category_counts.groupby('category').sum().sort_values(ascending=False).index.values
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

            subcategories = exp.category_counts.loc[category].sort_values(ascending=False).index.values
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
                qnames = list(islice(exp.query_names(), 200))

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
            als = exp.get_read_alignments(read_id)

        return als

    def plot(self, read_id, **plot_kwargs):
        with self.output:
            exp = self.get_current_experiment()

            if exp is None:
                return None

            als = self.get_alignments(exp, read_id)

            if als is None:
                return None

            if isinstance(als, dict):
                read_details = [
                    'query name: ' + als['R1'][0].query_name,
                    'R1 sequence: ' + als['R1'][0].get_forward_sequence(),
                    'R2 sequence: ' + als['R2'][0].get_forward_sequence(),
                ]

                if self.by_outcome:
                    layout = knock_knock.layout.NonoverlappingPairLayout(als['R1'], als['R2'], exp.target_info)
            else:
                read_details = [
                    'query name: ' + als[0].query_name,
                    'sequence: ' + als[0].get_forward_sequence(),
                ]

                if self.by_outcome:
                    layout = exp.categorizer(als, exp.target_info, mode=exp.layout_mode, error_corrected=exp.has_UMIs)

            if self.by_outcome:
                layout.categorize()

                inferred_amplicon_length = layout.inferred_amplicon_length

                read_details = read_details[:1] + [
                    'category: ' + layout.category,
                    'subcategory: ' + layout.subcategory,
                    'details: ' + layout.details,
                ] + read_details[1:]

                if self.non_widgets['alignments_to_show'].value == 'relevant':
                    als = layout.relevant_alignments

            else:
                inferred_amplicon_length = None


            if self.non_widgets['alignments_to_show'].value == 'parsimonious':
                plot_kwargs['parsimonious'] = True

            plot_kwargs['centered_on_primers'] = False
            plot_kwargs['force_left_aligned'] = False
            plot_kwargs['force_right_aligned'] = False

            if self.non_widgets['alignment_registration'].value == 'centered_on_primers':
                plot_kwargs['centered_on_primers'] = True
            elif self.non_widgets['alignment_registration'].value == 'left':
                plot_kwargs['force_left_aligned'] = True
            elif self.non_widgets['alignment_registration'].value == 'right':
                plot_kwargs['force_right_aligned'] = True

            for k in self.draw_buttons:
                plot_kwargs[k] = self.draw_buttons[k].value

            for k, v in exp.diagram_kwargs.items():
                plot_kwargs.setdefault(k, v)

            diagram = knock_knock.visualize.ReadDiagram(als,
                                                        exp.target_info,
                                                        inferred_amplicon_length=inferred_amplicon_length,
                                                        title='',
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
        self.widgets.update({
        })

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