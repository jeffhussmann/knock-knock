import shutil
from pathlib import Path

import pytest

import hits.utilities

import knock_knock.illumina_experiment

memoized_property = hits.utilities.memoized_property

parent = Path(__file__).parent

class Experiment(knock_knock.illumina_experiment.IlluminaExperiment):
    @memoized_property
    def results_dir(self):
        d = self.base_dir / 'test_results' 

        d = self.add_batch_name_directories(d)

        d /= self.sample_name

        return d 

@pytest.fixture(scope='module', params=knock_knock.experiment.get_all_batch_names(parent))
def experiment(request):
    prefixed_name = request.param

    experiment = {
        'expected': knock_knock.illumina_experiment.IlluminaExperiment(parent, prefixed_name, 'simulated'),
        'actual': Experiment(parent, prefixed_name, 'simulated'),
    }

    if experiment['actual'].results_dir.is_dir():
        shutil.rmtree(experiment['actual'].results_dir)

    for stage in ['preprocess', 'align', 'categorize']:
        experiment['actual'].process(stage=stage)

    yield experiment

    shutil.rmtree(experiment['actual'].results_dir)

def test_outcome_counts(experiment):
    expected = experiment['expected'].outcome_counts
    actual = experiment['actual'].outcome_counts
    assert (expected == actual).all()