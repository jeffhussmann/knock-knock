import knock_knock.illumina_experiment
import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout
import knock_knock.Bxb1_layout

from hits.utilities import memoized_property

class PrimeEditingExperiment(knock_knock.illumina_experiment.IlluminaExperiment):
    @memoized_property
    def categorizer(self):
        return knock_knock.prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        longest_seen = max((outcome.inferred_amplicon_length for outcome in outcomes), default=0)
        return max(min(longest_seen, 600), 100)
    
class TwinPrimeExperiment(PrimeEditingExperiment):
    @memoized_property
    def categorizer(self):
        return knock_knock.twin_prime_layout.Layout

class Bxb1TwinPrimeExperiment(TwinPrimeExperiment):
    @memoized_property
    def categorizer(self):
        return knock_knock.Bxb1_layout.Layout