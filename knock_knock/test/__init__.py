import shutil

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
    def copy_editing_strategy(self, editing_strategy):
        strat = editing_strategy

        self.strategy_dir.mkdir(exist_ok=True, parents=True)

        manifest_fn = self.strategy_dir / 'manifest.yaml'

        manifest_fn.write_text(yaml.safe_dump(strat.manifest))

        for source in strat.sources:
            shutil.copy(strat.dir / (source + '.gb'), self.strategy_dir / (source + '.gb'))

        sgRNAs = knock_knock.pegRNAs.read_csv(strat.fns['sgRNAs'], process=False).loc[strat.sgRNAs]

        sgRNAs.to_csv(self.strategy_dir / 'sgRNAs.csv')