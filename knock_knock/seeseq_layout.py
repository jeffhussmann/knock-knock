import hits.utilities
import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout

memoized_property = hits.utilities.memoized_property

class Layout(knock_knock.prime_editing_layout.Layout):
    @property
    def inferred_amplicon_length(self):
        return self.read_length

class DualFlapLayout(knock_knock.twin_prime_layout.Layout):
    @property
    def inferred_amplicon_length(self):
        return self.read_length