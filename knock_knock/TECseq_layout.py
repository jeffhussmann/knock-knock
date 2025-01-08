import itertools

import hits.interval
import hits.sam
import hits.utilities

import knock_knock.prime_editing_layout

memoized_property = hits.utilities.memoized_property

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
            ('n/a',
             'primer dimer',
            ),
        ),
        ('minimal alignment to intended target',
            ('n/a',
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
    def primer_alignments(self):
        primer_alignments = super().primer_alignments
        primer_alignments['right'] = None
        return primer_alignments

    @memoized_property
    def minimal_cover(self):
        covered = self.extension_chain['query_covered_incremental']

        minimal_cover = None

        for key in ['first target', 'pegRNA', 'second target']:
            if (key in covered) and (self.not_covered_by_primers - covered[key]).is_empty:
                minimal_cover = key
                break

        return minimal_cover

    def categorize(self):
        self.details = 'n/a'
        self.outcome = None

        side = self.target_info.pegRNA_side

        if self.minimal_cover is not None:
            if self.minimal_cover == 'first target':
                self.category = 'targeted genomic sequence'
                self.subcategory = 'unedited'
                self.details = str(self.extension_chain_edges[side])

            elif self.minimal_cover == 'pegRNA':
                self.category = 'RTed sequence'
                self.subcategory = 'n/a'
                self.details = str(self.extension_chain_edges[side])

            elif self.minimal_cover == 'second target':
                self.category = 'targeted genomic sequence'
                self.subcategory = 'edited'
                self.details = str(self.extension_chain_edges[side])
            
            else:
                raise ValueError

            self.relevant_alignments = [al for al in self.target_alignments + self.pegRNA_alignments if not self.is_pegRNA_protospacer_alignment(al)]

        elif self.nonspecific_amplification:
            self.category = 'nonspecific amplification'
            self.subcategory = 'n/a'

        elif self.query_length_covered_by_on_target_alignments <= 30:
            self.register_minimal_alignments_detected()

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome

    @memoized_property
    def plot_parameters(self):
        ti = self.target_info

        plot_parameters = super().plot_parameters

        for virtual_primer in {(ti.target, name) for name in ti.primers if name != ti.sequencing_start_feature_name}:
            plot_parameters['features_to_show'].remove(virtual_primer)

        return plot_parameters

class NoOverlapPairLayout(Layout):
    def __init__(self, alignments, target_info):
        self.alignments = alignments
        self.target_info = target_info

        self.layouts = {which: Layout(als, target_info) for which, als in self.alignments.items()}

        for layout in self.layouts.values():
            layout.categorize()

        self._inferred_amplicon_length = -1

    @property
    def inferred_amplicon_length(self):
        return self._inferred_amplicon_length

    @memoized_property
    def query_length_covered_by_on_target_alignments(self):
        return sum(layout.query_length_covered_by_on_target_alignments for layout in self.layouts.values())

    @memoized_property
    def nonspecific_amplification(self):
        R1 = self.layouts['R1']
        R2 = self.layouts['R2']
        
        R2_covering_als = []
    
        to_cover = R2.whole_read_minus_edges(2)
    
        for al in R2.supplemental_alignments:
            if (to_cover - hits.interval.get_covered(al)).total_length == 0:
                R2_covering_als.append(al)
                
        best_pairs = []
                    
        valid_pairs = []

        if R1.nonspecific_amplification:
            R1_als = R1.nonspecific_amplification['covering_als']
        else:
            R1_als = []
    
        for R1_al, R2_al in itertools.product(R1_als, R2_covering_als):
            
            if R1_al.reference_name != R2_al.reference_name:
                continue
    
            R1_strand = hits.sam.get_strand(R1_al)
            R2_strand = hits.sam.get_strand(R2_al)
    
            if R1_strand != R2_strand:
                # should be in opposite orientation if concordant
    
                if R1_strand == '+':
                    start = R1_al.reference_end - 1
                    end = R2_al.reference_start
        
                elif R1_strand == '-':
                    start = R2_al.reference_end - 1
                    end = R1_al.reference_start

                ref_gap = end - start - 1

                R1_query_end = hits.interval.get_covered(R1_al).end
                R2_query_end = hits.interval.get_covered(R2_al).end

                length = ref_gap + (R1_query_end + 1) + (R2_query_end + 1)
        
                if 0 < length < 2000:
                    valid_pairs.append((length, {'R1': R1_al, 'R2': R2_al}))
    
        if valid_pairs:
            min_length = min(length for length, _ in valid_pairs)
    
            best_pairs = (min_length, [als for length, als in valid_pairs if length == min_length])

        return best_pairs

    @memoized_property
    def targeted_genomic_sequence(self):
        R1 = self.layouts['R1']
        R2 = self.layouts['R2']
        
        R2_covering_als = []
    
        to_cover = R2.whole_read_minus_edges(2)
    
        for al in R2.target_alignments:
            if (to_cover - hits.interval.get_covered(al)).total_length == 0:
                R2_covering_als.append(al)

        results = None

        if len(R2_covering_als) == 1:
            R2_al = R2_covering_als[0]
                
            if R1.category == 'targeted genomic sequence':

                R1_al = R1.extension_chain['alignments']['first target']
    
                R1_strand = hits.sam.get_strand(R1_al)
                R2_strand = hits.sam.get_strand(R2_al)
    
                if R1_strand != R2_strand:
                    # should be in opposite orientation if concordant
    
                    if R1_strand == '+':
                        start = R1_al.reference_end - 1
                        end = R2_al.reference_start
            
                    elif R1_strand == '-':
                        start = R2_al.reference_end - 1
                        end = R1_al.reference_start

                    ref_gap = end - start - 1

                    R1_query_end = hits.interval.get_covered(R1_al).end
                    R2_query_end = hits.interval.get_covered(R2_al).end

                    length = ref_gap + (R1_query_end + 1) + (R2_query_end + 1)
        
                    results = {
                        'length': length,
                        'edge': R2.convert_target_alignment_edge_to_nick_coordinate(al, 'end' if R1_strand == '+' else 'start'),
                        'relevant_alignments': {'R1': list(R1.extension_chain['alignments'].values()), 'R2': [R2_al]},
                    }
    
        return results

    def categorize(self):
        self.details = 'n/a'
        self.outcome = None
        self.relevant_alignments = self.alignments

        if self.nonspecific_amplification:
            length, _ = self.nonspecific_amplification
            self.category = 'nonspecific amplification'
            self.subcategory = 'n/a'
            self._inferred_amplicon_length = length

        elif self.targeted_genomic_sequence:
            results = self.targeted_genomic_sequence
            self.category = 'targeted genomic sequence'
            self.subcategory = 'unknown editing status'
            self.details = str(results['edge'])
            self._inferred_amplicon_length = results['length']
            self.relevant_alignments = results['relevant_alignments']

        elif self.query_length_covered_by_on_target_alignments <= 30:
            self.category = 'minimal alignment to intended target'
            self.subcategory = 'n/a'

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'

        return self.category, self.subcategory, self.details, self.outcome

    def plot(self,
             relevant=True,
             manual_alignments=None,
             **kwargs,
            ):

        als_to_plot = self.alignments

        if manual_alignments is not None:
            als_to_plot = manual_alignments

        if relevant:
            self.categorize()
            als_to_plot = self.relevant_alignments

        ti = self.target_info

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 self.target_info,
                                                                 flip_target=ti.sequencing_direction == '-',
                                                                 inferred_amplicon_length=self.inferred_amplicon_length,
                                                                 features_to_show=self.plot_parameters['features_to_show'],
                                                                 label_overrides=self.plot_parameters['label_overrides'],
                                                                 feature_heights=self.plot_parameters['feature_heights'],
                                                                 **kwargs,
                                                                )

        return diagram
