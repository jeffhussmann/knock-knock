import shutil

import pandas as pd
import yaml

import hits.utilities

import knock_knock.illumina_experiment
import knock_knock.pacbio_experiment

memoized_property = hits.utilities.memoized_property

class Experiment:
    def __init__(self, identifier, results_prefix):
        self.results_prefix = results_prefix
        super().__init__(identifier)

    @memoized_property
    def results_dir(self):
        d = super().results_dir

        return d.parent / f'{self.results_prefix}_{d.name}'

class IlluminaExperiment(Experiment, knock_knock.illumina_experiment.IlluminaExperiment):
    pass

class PacbioExperiment(Experiment, knock_knock.pacbio_experiment.PacbioExperiment):
    pass

class Extractor:

    @property
    def data_dir(self):
        return type(self).base_dir / 'data' / self.name

    @property
    def sample_sheet_fn(self):
        return self.data_dir / 'sample_sheet.csv'

    @memoized_property
    def sample_sheet(self):
        return pd.read_csv(self.sample_sheet_fn, index_col='sample_name').loc[type(self).sample_name]

    @property
    def strategy_dir(self):
        return type(self).base_dir / 'strategies' / self.name

    @memoized_property
    def platform(self):
        return self.sample_sheet['platform']

    def experiment(self, results_prefix):
        identifier = knock_knock.experiment.ExperimentIdentifier(type(self).base_dir,
                                                                 self.name,
                                                                 type(self).sample_name,
                                                                )

        if self.platform == 'illumina':
            exp_class = knock_knock.test.IlluminaExperiment
        elif self.platform == 'pacbio':
            exp_class = knock_knock.test.PacbioExperiment
        else:
            raise ValueError

        return exp_class(identifier, results_prefix)

    def copy_editing_strategy(self, editing_strategy):
        strat = editing_strategy

        self.strategy_dir.mkdir(exist_ok=True, parents=True)

        manifest_fn = self.strategy_dir / 'manifest.yaml'

        manifest_fn.write_text(yaml.safe_dump(strat.manifest))

        for source in strat.sources:
            shutil.copy(strat.dir / (source + '.gb'), self.strategy_dir / (source + '.gb'))

        sgRNAs = knock_knock.pegRNAs.read_csv(strat.fns['sgRNAs'], process=False).loc[strat.sgRNAs]

        sgRNAs.to_csv(self.strategy_dir / 'sgRNAs.csv')