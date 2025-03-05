import hits.utilities
import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout

memoized_property = hits.utilities.memoized_property
memoized_with_args = hits.utilities.memoized_with_args

class Layout(knock_knock.prime_editing_layout.Layout):
    category_order = [
        ('unknown editing status',
            ('unknown editing status',
            ),
        ),
    ] + knock_knock.prime_editing_layout.Layout.category_order

    @property
    def inferred_amplicon_length(self):
        return self.read_length

    @memoized_with_args
    def minimal_cover_by_side(self, side):
        covered = self.extension_chains_by_side[side]['query_covered_incremental']

        minimal_cover = None

        for key in ['first target',
                    'pegRNA',
                    'first pegRNA',
                    'second target',
                   ]:
            
            if (key in covered) and (self.not_covered_by_primers - covered[key]).is_empty:
                minimal_cover = key
                break

        return minimal_cover

    @memoized_property
    def minimal_cover(self):
        return self.minimal_cover_by_side('left')

    def categorize(self):
        super().categorize()

        if self.category == 'wild type' and not self.extension_chain_covers_both_HAs:
            self.category = 'unknown editing status'
            self.subcategory = 'unknown editing status'

        return self.category, self.subcategory, self.details, self.outcome

class DualFlapLayout(knock_knock.twin_prime_layout.Layout):
    @property
    def inferred_amplicon_length(self):
        return self.read_length

class NoOverlapPairLayout(Layout):
    individual_layout_class = Layout

    def __init__(self, alignments, target_info):
        self.alignments = alignments
        self.target_info = target_info
        self._flipped = False

        self.layouts = {
            'R1': type(self).individual_layout_class(alignments['R1'], target_info),
            'R2': type(self).individual_layout_class(alignments['R2'], target_info, flipped=True),
        }

        self._inferred_amplicon_length = -1

    @property
    def inferred_amplicon_length(self):
        return self._inferred_amplicon_length

    @memoized_property
    def concordant_nonoverlapping(self):
        R1 = self.layouts['R1']
        R2 = self.layouts['R2']

        R1.categorize()

        if R1.category in ['wild type', 'intended edit', 'partial edit']:
            R1_als = [R1.target_edge_alignments['right']]
            R2_als = [R2.target_edge_alignments['right']]

        elif R1.nonspecific_amplification:
            R1_als = R1.nonspecific_amplification['covering_als']
        
            R2_als = []
        
            to_cover = R2.whole_read_minus_edges(2)
        
            for al in R2.supplemental_alignments:
                if (to_cover - hits.interval.get_covered(al)).total_length == 0:
                    R2_als.append(al)
        
        else:
            R1_als = []
            R2_als = []

        gap, amplicon_length, pairs = self.find_pair_with_shortest_gap(R1_als, R2_als)

        if gap is not None:
            results = {
                'gap': gap,
                'amplicon_length': amplicon_length,
                'pairs': pairs,
            }
        else:
            results = None

        return results

    def categorize(self):
        R1 = self.layouts['R1']
        R2 = self.layouts['R2']

        R1.categorize()

        self.category = R1.category
        self.subcategory = R1.subcategory
        self.details = R1.details
        self.outcome = R1.outcome

        self.relevant_alignments = {
            'R1': R1.relevant_alignments,
            'R2': R2.alignments,
        }

        if self.concordant_nonoverlapping is not None:
            results = self.concordant_nonoverlapping
            self.relevant_alignments['R2'] = [R2 for R1, R2 in results['pairs']]
            self._inferred_amplicon_length = results['amplicon_length']

        return self.category, self.subcategory, self.details, self.outcome

    def plot(self,
             relevant=True,
             manual_alignments=None,
             **kwargs,
            ):

        plot_kwargs = kwargs.copy()
        plot_kwargs.setdefault('alignment_registration', 'left')

        als_to_plot = self.alignments

        if manual_alignments is not None:
            als_to_plot = manual_alignments

        elif relevant:
            self.categorize()
            als_to_plot = self.relevant_alignments

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 self.target_info,
                                                                 flip_target=self.sequencing_direction == '-',
                                                                 inferred_amplicon_length=self.inferred_amplicon_length,
                                                                 features_to_show=self.plot_parameters['features_to_show'],
                                                                 label_overrides=self.plot_parameters['label_overrides'],
                                                                 feature_heights=self.plot_parameters['feature_heights'],
                                                                 **plot_kwargs,
                                                                )

        return diagram

    def characterize_gap_between_closest_alignments(self, R1_al, R2_al):
        gap = 'unknown'
        amplicon_length = -1

        if R1_al is not None and R2_al is not None:
            R1_strand = hits.sam.get_strand(R1_al)
            R2_strand = hits.sam.get_strand(R2_al)

            if R1_al.reference_name == self.target_info.pegRNA_names_by_side_of_read.get('left') and R2_al.reference_name == self.target_info.target:
                gap = 'unknown'
                amplicon_length = -1

            elif (R1_al.reference_name != R2_al.reference_name) or (R1_strand == R2_strand):
                gap = None
                amplicon_length = None

            else:
                if R1_strand == '+':
                    gap_start = R1_al.reference_end - 1
                    gap_end = R2_al.reference_start

                elif R1_strand == '-':
                    gap_start = R2_al.reference_end - 1
                    gap_end = R1_al.reference_start

                gap = gap_end - gap_start - 1

                R1_query_end = hits.interval.get_covered(R1_al).end
                R2_query_end = hits.interval.get_covered(R2_al).end

                amplicon_length = gap + (R1_query_end + 1) + (R2_query_end + 1)

        return gap, amplicon_length

    def find_pair_with_shortest_gap(self, R1_als, R2_als):
        concordant_pairs = []

        for R1_al in R1_als:
            for R2_al in R2_als:
                gap, amplicon_length = self.characterize_gap_between_closest_alignments(R1_al, R2_al)
                if (gap is not None) and ((gap == 'unknown') or (-20 <= gap <= 2000)):
                    concordant_pairs.append((gap, amplicon_length, (R1_al, R2_al)))

        if len(concordant_pairs) == 0:
            min_gap = None
            amplicon_length = None
            pairs = []
        else:
            min_gap, amplicon_length = min((gap, amplicon_length) for gap, amplicon_length, pair in concordant_pairs)
            pairs = [pair for gap, amplicon_length, pair in concordant_pairs if gap == min_gap]

        return min_gap, amplicon_length, pairs