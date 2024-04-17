import hits.utilities
import knock_knock.prime_editing_layout

memoized_property = hits.utilities.memoized_property

class Layout(knock_knock.prime_editing_layout.Layout):
    category_order = [
        ('RTed sequence',
            ('includes scaffold',
             'no scaffold',
            ),
        ),
        ('targeted genomic sequence',
            ('edited',
             'unedited',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
            ),
        ),
    ]

    @memoized_property
    def minimal_cover(self):
        covered = self.extension_chain['query_covered_incremental']

        minimal_cover = None

        for key in ['first target', 'pegRNA', 'second target']:
            if (key in covered) and (self.whole_read - covered[key]).is_empty:
                minimal_cover = key
                break

        return minimal_cover

    def register_unintended_rejoining(self):
        chain = self.extension_chain
        chain_edges = self.extension_chain_edges

        pegRNA_al = chain['alignments']['pegRNA']

        self.category = 'RTed sequence'

        if self.alignment_scaffold_overlap(pegRNA_al) >= 1:
            self.subcategory = 'includes scaffold'
        else:
            self.subcategory = 'no scaffold'

        self.outcome = knock_knock.prime_editing_layout.UnintendedRejoiningOutcome(chain_edges['left'], -1, -1, [])

        self.relevant_alignments = []

        for side, key in [
            (self.target_info.pegRNA_side, 'first target'),
            (self.target_info.pegRNA_side, 'pegRNA'),
            (self.target_info.non_pegRNA_side, 'first target'),
        ]:
            if key in chain['alignments']:
                self.relevant_alignments.append(chain['alignments'][key])

    def categorize(self):
        if self.minimal_cover == 'first target':
            self.category = 'targeted genomic sequence'
            self.subcategory = 'unedited'
            self.details = str(self.extension_chain_edges['left'])
            self.outcome = None

        elif self.minimal_cover == 'pegRNA':
            self.register_unintended_rejoining()

        elif self.minimal_cover == 'second target':
            self.category = 'targeted genomic sequence'
            self.subcategory = 'edited'
            self.details = str(self.extension_chain_edges['left'])
            self.outcome = None

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'
            self.outcome = None

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome