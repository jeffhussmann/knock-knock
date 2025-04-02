import copy

import hits.interval
import hits.sam
import hits.utilities

import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout

from knock_knock.outcome import *

memoized_property = hits.utilities.memoized_property
memoized_with_args = hits.utilities.memoized_with_args

class Layout(knock_knock.prime_editing_layout.Layout):
    category_order = [
        ('RTed sequence',
            ('n/a',
            ),
        ),
        ('targeted genomic sequence',
            ('edited',
             'unedited',
             'unknown editing status'
            ),
        ),
        ('nonspecific amplification',
            ('hg38',
             'mm10',
             'primer dimer',
             'short unknown',
            ),
        ),
        ('minimal alignment to intended target',
            ('n/a',
            ),
        ),
        ('phiX',
            ('phiX',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
            ),
        ),
    ]

    @property
    def inferred_amplicon_length(self):
        return self.read_length

    @memoized_property
    def gap_covering_alignments(self):
        return []

    @memoized_property
    def partial_gap_perfect_alignments(self):
        return []

    def realign_edges_to_primers(self, read_side):
        return None

    @memoized_property
    def perfect_right_edge_alignment(self):
        return None

    @memoized_property
    def primer_alignments(self):
        primer_alignments = super().primer_alignments
        primer_alignments['right'] = None
        return primer_alignments

    @memoized_property
    def target_nts_past_primer(self):
        target_nts_past_primer = super().target_nts_past_primer

        target_nts_past_primer['right'] = 0

        return target_nts_past_primer

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
        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.aligns_to_phiX:
            self.category = 'phiX'
            self.subcategory = 'phiX'

            self.relevant_alignments = [self.longest_phiX_alignment]

        elif self.starts_at_expected_location and self.minimal_cover is not None:
            if self.minimal_cover == 'first target':
                self.category = 'targeted genomic sequence'
                self.subcategory = 'unedited'
                edge = hits.sam.reference_edges(self.extension_chains_by_side['left']['alignments']['first target'])[3]
                self.Details = Details(target_edge=edge, mismatches=self.non_pegRNA_mismatches)

            elif self.minimal_cover in ['pegRNA', 'first pegRNA']:
                self.category = 'RTed sequence'
                self.subcategory = 'n/a'
                edge = self.extension_chain_edges['left']
                self.Details = Details(pegRNA_edge=edge, mismatches=self.non_pegRNA_mismatches)

            elif self.minimal_cover == 'second target':
                self.category = 'targeted genomic sequence'
                self.subcategory = 'edited'
                edge = hits.sam.reference_edges(self.extension_chains_by_side['left']['alignments']['second target'])[3]
                self.Details = Details(target_edge=edge, mismatches=self.non_pegRNA_mismatches)
            
            else:
                raise ValueError

            self.relevant_alignments = self.parsimonious_extension_chain_alignments

        elif self.query_length_covered_by_on_target_alignments <= 30:
            self.register_minimal_alignments_detected()

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'

            self.relevant_alignments = self.uncategorized_relevant_alignments

        self.details = str(self.Details)

        self.categorized = True

        return self.category, self.subcategory, self.details, self.Details

    @memoized_property
    def plot_parameters(self):
        ti = self.target_info

        plot_parameters = super().plot_parameters

        for virtual_primer in {(ti.target, name) for name in ti.primers if name != ti.sequencing_start_feature_name}:
            plot_parameters['features_to_show'].remove(virtual_primer)

        return plot_parameters

class NoOverlapPairLayout(Layout, knock_knock.layout.NoOverlapPairCategorizer):
    individual_layout_class = Layout

    @property
    def inferred_amplicon_length(self):
        return self._inferred_amplicon_length

    @memoized_property
    def query_length_covered_by_on_target_alignments(self):
        return sum(layout.query_length_covered_by_on_target_alignments for layout in self.layouts.values())

    @memoized_property
    def concordant_nonoverlapping(self):
        '''
        R1 is nonspecific
            R2 is covered by concordant
        
        R1 is targeted genomic and ends before edit:
            R2 contains entire edit
            R2 starts after edit

        R1 is targeted genomic and ends after edit
            R2 is covered by concordant           

        R1 is RTed
            R2 has a relevant side extension chain and ends in RTed
            R2 starts after edit

        '''

        R1 = self.layouts['R1']
        R2 = self.layouts['R2']

        R1_cover = R1.minimal_cover
        R2_cover = R2.minimal_cover

        if R1.nonspecific_amplification:
            R1.register_nonspecific_amplification()

            category = 'nonspecific amplification'
            subcategory = R1.subcategory

            R1_als = R1.nonspecific_amplification['covering_als']
        
            R2_als = []
        
            to_cover = R2.whole_read_minus_edges(2)
        
            for al in R2.supplemental_alignments:
                if (to_cover - hits.interval.get_covered(al)).total_length == 0:
                    R2_als.append(al)

        elif R1_cover is None or R2_cover is None:
            R1_als = []
            R2_als = []

        elif R1.minimal_cover == 'first target':
            R1_als = [R1.extension_chains_by_side['left']['alignments']['first target']]

            if R2_cover == 'first target':
                R2_als = [R2.extension_chains_by_side['left']['alignments']['first target']]

                category = 'targeted genomic sequence'

                if R1.extension_chain_covers_both_HAs:
                    R1.categorize()
                    subcategory = R1.subcategory
                else:
                    subcategory = 'unknown editing status'

            else:
                category = 'uncategorized'
                subcategory = 'uncategorized'

                R2_als = R2.uncategorized_relevant_alignments

        elif R1_cover == 'pegRNA':
            R1_als = [R1.extension_chains_by_side['left']['alignments']['pegRNA']]

            if R2_cover == 'pegRNA':
                R2_als = [R2.extension_chains_by_side['left']['alignments']['pegRNA']]

                category = 'targeted genomic sequence'
                subcategory = 'edited'

            elif R2_cover == 'first target':
                R2_als = [R2.extension_chains_by_side['left']['alignments']['first target']]

                category = 'targeted genomic sequence'
                subcategory = 'unknown editing status'

            else:
                raise NotImplementedError

        elif R1_cover == 'second target':
            R1_als = [R1.extension_chains_by_side['left']['alignments']['second target']]

            if R2_cover == 'first target':
                R2_als = [R2.extension_chains_by_side['left']['alignments']['first target']]

                category = 'targeted genomic sequence'
                subcategory = 'edited'

            else:
                category = 'uncategorized'
                subcategory = 'uncategorized'

                R2_als = R2.uncategorized_relevant_alignments

        else:
            R1_als = []
            R2_als = []

        gap, amplicon_length, pairs = self.find_pair_with_shortest_gap(R1_als, R2_als)

        if gap is not None:
            results = {
                'category': category,
                'subcategory': subcategory,
                'gap': gap,
                'amplicon_length': amplicon_length,
                'pairs': pairs,
            }
        else:
            results = None

        return results

    def register_targeted_genomic_sequence(self):
        R1 = self.layouts['R1']
        R2 = self.layouts['R2']

        R2_al = R2.extension_chains_by_side['left']['alignments']['first target']
        edge = hits.sam.reference_edges(R2_al)[5]

        self.category = self.concordant_nonoverlapping['category']
        self.subcategory = self.concordant_nonoverlapping['subcategory']

        self.Details = Details(target_edge=edge, mismatches=R1.non_pegRNA_mismatches)

        self.relevant_alignments = {
            'R1': R1.parsimonious_extension_chain_alignments,
            'R2': R2.parsimonious_extension_chain_alignments,
        }

    def register_nonspecific_amplification(self):
        R1 = self.layouts['R1']

        self.category = self.concordant_nonoverlapping['category']
        self.subcategory = self.concordant_nonoverlapping['subcategory']

        self.relevant_alignments = {
            'R1': R1.target_edge_alignments_list + [R1 for R1, R2 in self.concordant_nonoverlapping['pairs']],
            'R2': [R2 for R1, R2 in self.concordant_nonoverlapping['pairs']],
        }

    def categorize(self):
        self.relevant_alignments = self.alignments

        if self.concordant_nonoverlapping:
            results = self.concordant_nonoverlapping

            if results['category'] == 'targeted genomic sequence':
                self.register_targeted_genomic_sequence()

            elif results['category'] == 'nonspecific amplification':
                self.register_nonspecific_amplification()

            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'

            self._inferred_amplicon_length = results['amplicon_length']

        elif self.query_length_covered_by_on_target_alignments <= 30:
            self.category = 'minimal alignment to intended target'
            self.subcategory = 'n/a'

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'

        self.details = str(self.Details)

        return self.category, self.subcategory, self.details, self.Details

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

        label_overrides = plot_kwargs.pop('label_overrides', self.plot_parameters['label_overrides'].copy())

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 self.target_info,
                                                                 highlight_SNPs=True,
                                                                 flip_target=self.sequencing_direction == '-',
                                                                 inferred_amplicon_length=self.inferred_amplicon_length,
                                                                 features_to_show=self.plot_parameters['features_to_show'],
                                                                 label_overrides=label_overrides,
                                                                 feature_heights=self.plot_parameters['feature_heights'],
                                                                 **plot_kwargs,
                                                                )

        return diagram

    def characterize_gap_between_closest_alignments(self, R1_al, R2_al):
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
                if (gap is not None) and ((gap == 'unknown') or (-10 <= gap <= 2000)):
                    concordant_pairs.append((gap, amplicon_length, (R1_al, R2_al)))

        if len(concordant_pairs) == 0:
            min_gap = None
            amplicon_length = None
            pairs = []
        else:
            min_gap, amplicon_length = min((gap, amplicon_length) for gap, amplicon_length, pair in concordant_pairs)
            pairs = [pair for gap, amplicon_length, pair in concordant_pairs if gap == min_gap]

        return min_gap, amplicon_length, pairs

class TwinPrimeLayout(Layout, knock_knock.twin_prime_layout.Layout):
    # MRO puts twin_prime_layout.Layout before prime_editing_layout.Layout
    pass

class NoOverlapPairTwinPrimeLayout(NoOverlapPairLayout, TwinPrimeLayout):
    individual_layout_class = TwinPrimeLayout