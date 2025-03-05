import copy
from collections import defaultdict

import hits.interval
import hits.utilities

from hits import sam, interval

import knock_knock.layout
import knock_knock.twin_prime_layout
from knock_knock.outcome import *

memoized_property = hits.utilities.memoized_property

class Layout(knock_knock.twin_prime_layout.Layout):
    category_order = [
        ('wild type',
            ('clean',
             'mismatches',
             'short indel far from cut',
            ),
        ),
        ('intended edit',
            ('replacement',
             'partial replacement',
             'deletion',
            ),
        ),
        ('partial replacement',
            ('left pegRNA',
             'right pegRNA',
             'both pegRNAs',
             'single pegRNA (ambiguous)',
            ),
        ),
        ('intended integration',
            ('single',
             'multiple',
             'messy',
            ),
        ),
        ('unintended rejoining of RT\'ed sequence',
            ('left RT\'ed, right RT\'ed',
             'left RT\'ed, right not RT\'ed',
             'left not RT\'ed, right RT\'ed',
             'left RT\'ed, right not seen',
             'left not seen, right RT\'ed',
            ),
        ),
        ('unintended rejoining of overlap-extended sequence',
            ('left RT\'ed + overlap-extended, right RT\'ed + overlap-extended',
             'left RT\'ed + overlap-extended, right RT\'ed',
             'left RT\'ed, right RT\'ed + overlap-extended',
             'left RT\'ed + overlap-extended, right not RT\'ed',
             'left not RT\'ed, right RT\'ed + overlap-extended',
             'left RT\'ed + overlap-extended, right not seen',
             'left not seen, right RT\'ed + overlap-extended',
            ),
        ),
        ('multistep unintended rejoining of RT\'ed sequence',
            ('left RT\'ed, right RT\'ed',
             'left RT\'ed, right indel',
             'left indel, right RT\'ed',
            ),
        ),
        ('flipped pegRNA incorporation',
            ('left pegRNA',
             'right pegRNA',
             'both pegRNAs',
            ),
        ),
        ('deletion',
            ('clean',
             'mismatches',
             'multiple',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('insertion',
            ('clean',
             'mismatches',
            ),
        ),
        ('edit + indel',
            ('deletion',
             'insertion',
             'duplication',
            ),
        ),
        ('multiple indels',
            ('multiple indels',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'hg38',
             'macFas5',
             'mm10',
             'bosTau7',
             'e_coli',
             'phiX',
            ),
        ),
        ('inversion',
            ('inversion',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
             'low quality', 
             'no alignments detected',
            ),
        ),
        ('malformed',
            ('doesn\'t have both primers',
            ),
        ),
        ('nonspecific amplification',
            ('hg19',
             'hg38',
             'macFas5',
             'T2T-MFA8v1.0',
             'mm10',
             'bosTau7',
             'e_coli',
             'b_subtilis',
             'phiX',
             'primer dimer',
             'short unknown',
             'extra sequence',
            ),
        ),
        ('phiX',
            ('phiX',
            ),
        ),
    ]

    @property
    def inferred_amplicon_length(self):
        if self.primer_alignments['left'] is not None:
            left = interval.get_covered(self.primer_alignments['left']).start
        else:
            left = 0

        if self.primer_alignments['right'] is not None:
            right = interval.get_covered(self.primer_alignments['right']).end
        else:
            right = self.whole_read.end

        inferred_amplicon_length = right - left + 1

        if inferred_amplicon_length < 0:
            inferred_amplicon_length = len(self.seq)

        return inferred_amplicon_length
            
    @memoized_property
    def is_low_quality(self):
        return False

    def overlaps_primer(self, al, strand):
        primer = self.target_info.primers_by_strand[strand]
        overlaps = sam.overlaps_feature(al, primer, require_same_strand=False)

        return overlaps

    @memoized_property
    def longest_primer_alignments(self):
        by_strand = {
            strand: max([al for al in self.target_alignments if self.overlaps_primer(al, strand)], key=lambda al: al.query_alignment_length, default=None)
            for strand in ['+', '-']
        }
        
        return by_strand

    @memoized_property
    def sequencing_direction(self):
        strands = set(sam.get_strand(al) for al in self.longest_primer_alignments.values() if al is not None)
        
        if len(strands) == 1:    
            strand = list(strands)[0]
        else:
            strand = None
            
        return strand

    @property
    def flipped(self):
        return self.sequencing_direction == '-'

    @memoized_property
    def target_edge_alignments(self):
        if self.sequencing_direction == '-':
            side_to_strand = {
                'left': '-',
                'right': '+',
            }
        else:
            side_to_strand = {
                'left': '+',
                'right': '-',
            }
            
        target_edge_als = {
            side: self.longest_primer_alignments[side_to_strand[side]]
            for side in ['left', 'right']
        }

        return target_edge_als

    @memoized_property
    def between_primers(self):
        if self.primer_alignments['left'] is not None:
            start = interval.get_covered(self.primer_alignments['left']).start
        else:
            start = self.whole_read.start

        if self.primer_alignments['right'] is not None:
            end = interval.get_covered(self.primer_alignments['right']).end
        else:
            end = self.whole_read.end

        return interval.Interval(start, end)

    def overlaps_outside_primers(self, al):
        return (hits.interval.get_covered(al) - self.between_primers).total_length > 0

    @memoized_property
    def starts_at_expected_location(self):
        edge_al = self.target_edge_alignments['left']
        return edge_al is not None

    @memoized_property
    def donor_alignments(self):
        ''' Donor meaning integrase donor '''
        if self.target_info.donor is not None:
            valid_names = [self.target_info.donor]
        else:
            valid_names = []

        donor_als = [
            al for al in self.alignments
            if al.reference_name in valid_names
        ]

        donor_als = self.split_and_extend_alignments(donor_als)
        
        return donor_als

    @memoized_property
    def pegRNA_integrase_sites(self):
        all_sites = self.target_info.integrase_sites
        by_pegRNA = {}
        for pegRNA_name in self.target_info.pegRNA_names:
            by_pegRNA[pegRNA_name] = {}

            for (ref_name, feature_name), feature in all_sites.items():
                if ref_name == pegRNA_name:
                    component = feature_name.split('_')[-2]
                    by_pegRNA[pegRNA_name][component] = feature

        return by_pegRNA

    @memoized_property
    def donor_integrase_sites(self):
        all_sites = self.target_info.integrase_sites

        by_component = defaultdict(list)

        for (ref_name, feature_name), feature in all_sites.items():
            if ref_name == self.target_info.donor:
                component = feature_name.split('_')[-2]
                by_component[component].append(feature)

        by_component = dict(by_component)

        return by_component

    def find_donor_alignment_extending_pegRNA_alignment(self, pegRNA_al, from_side):
        donor_als = []

        pegRNA_CD_name = self.pegRNA_integrase_sites[pegRNA_al.reference_name]['CD'].ID
        
        for donor_al in self.donor_alignments:
            for donor_CD_feature in self.donor_integrase_sites['CD']:
                donor_CD_name = donor_CD_feature.ID

                if from_side == 'left':
                    left_al = pegRNA_al
                    left_feature = pegRNA_CD_name

                    right_al = donor_al
                    right_feature = donor_CD_name

                elif from_side == 'right':
                    left_al = donor_al
                    left_feature = donor_CD_name

                    right_al = pegRNA_al
                    right_feature = pegRNA_CD_name
                
                else:
                    raise ValueError

                extension_results = self.are_mutually_extending_from_shared_feature(left_al,
                                                                                    left_feature,
                                                                                    right_al,
                                                                                    right_feature,
                                                                                   )
                if extension_results['status'] == 'definite':
                    donor_als.append(donor_al)
                    
        return max(donor_als, key=lambda al: al.query_alignment_length, default=None)

    def find_pegRNA_alignment_extending_donor_alignment(self, donor_al, from_side):
        extending_pegRNA_als = []

        for pegRNA_name, pegRNA_als in self.pegRNA_alignments_by_pegRNA_name.items():
            pegRNA_CD_name = self.pegRNA_integrase_sites[pegRNA_name]['CD'].ID

            for pegRNA_al in pegRNA_als:

                for donor_CD_feature in self.donor_integrase_sites['CD']:
                    donor_CD_name = donor_CD_feature.ID

                    if from_side == 'left':
                        left_al = donor_al
                        left_feature = donor_CD_name

                        right_al = pegRNA_al
                        right_feature = pegRNA_CD_name

                    elif from_side == 'right':
                        left_al = pegRNA_al
                        left_feature = pegRNA_CD_name

                        right_al = donor_al
                        right_feature = donor_CD_name

                    else:
                        raise ValueError

                    extension_results = self.are_mutually_extending_from_shared_feature(left_al,
                                                                                        left_feature,
                                                                                        right_al,
                                                                                        right_feature,
                                                                                       )
                    if extension_results['status'] == 'definite':
                        extending_pegRNA_als.append(pegRNA_al)

        return max(extending_pegRNA_als, key=lambda al: al.query_alignment_length, default=None)

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

    def manual_anchors(self, alignments_to_plot):
        return {}

    def interesting_and_uninteresting_indels(self, als):
        # For now, ignore most indels.

        interesting = []
        uninteresting = []

        indels = self.extract_indels_from_alignments(als)

        for indel, near_cut in indels:
            if indel.kind == 'D' and indel.length > 20:
                interesting.append(indel)

        return interesting, uninteresting

    @memoized_property
    def donor_extension_chains_by_side(self):
        chains = copy.deepcopy(self.extension_chains_by_side)

        for side in ['left', 'right']:
            if chains[side]['description'] == "RT'ed":
                first_pegRNA_al = chains[side]['alignments']['first pegRNA']
                donor_al = self.find_donor_alignment_extending_pegRNA_alignment(first_pegRNA_al, side)

                if donor_al is not None:
                    chains[side]['description'] = 'one-sided integration'

                    chains[side]['alignments']['donor'] = donor_al

                    second_pegRNA_al = self.find_pegRNA_alignment_extending_donor_alignment(donor_al, side)

                    if second_pegRNA_al is not None:
                        chains[side]['description'] = 'integration'
                        chains[side]['alignments']['second pegRNA'] = second_pegRNA_al

                        overlap_extended_target_al, _, _ = self.find_target_alignment_extending_pegRNA_alignment(second_pegRNA_al, 'PBS')
                    
                        if overlap_extended_target_al is not None:
                            chains[side]['alignments']['second target'] = overlap_extended_target_al

            if 'donor' in chains[side]['alignments']:
                al_order = [
                    'first target',
                    'first pegRNA',
                    'donor',
                    'second pegRNA',
                    'second target',
                ]

                query_covered = interval.get_disjoint_covered([])
                query_covered_incremental = {'none': query_covered}

                for al_order_i, al_key in enumerate(al_order):
                    if al_key in chains[side]['alignments']:
                        als_up_to = [chains[side]['alignments'][key] for key in al_order[:al_order_i + 1]]
                        query_covered = interval.get_disjoint_covered(als_up_to)
                        query_covered_incremental[al_key] = query_covered

                chains[side].update({
                    'query_covered': query_covered,
                    'query_covered_incremental': query_covered_incremental,
                })

        return chains

    @memoized_property
    def nonspecific_amplification_of_donor(self):
        covering_donor_als = []

        if self.primer_alignments['left'] is not None and self.donor_alignments:
            covered = interval.get_disjoint_covered(self.donor_alignments)
            if (self.not_covered_by_primers - covered).total_length == 0:
                covering_donor_als = self.donor_alignments

        return covering_donor_als

    @memoized_property
    def is_intended_integration(self):
        chains = self.donor_extension_chains_by_side
        return chains['left']['description'] == 'integration' or chains['right']['description'] == 'integration'

    def register_intended_integration(self):
        chains = self.donor_extension_chains_by_side

        self.category = 'intended integration'

        if chains['left']['query_covered'] == chains['right']['query_covered']:
            self.subcategory = 'single'
        elif chains['left']['description'] == 'integration' and chains['right']['description'] == 'integration':
            self.subcategory = 'multiple'
        else:
            self.subcategory = 'messy'

        self.outcome = Outcome('n/a')

    def categorize(self):
        self.outcome = None

        if self.primer_alignments['left'] is None or self.primer_alignments['right'] is None:
            self.category = 'malformed'
            self.subcategory = 'doesn\'t have both primers'
            self.details = 'n/a/'

        elif self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.aligns_to_phiX:
            self.category = 'phiX'
            self.subcategory = 'phiX'
            self.details = 'n/a'

            self.relevant_alignments = [self.longest_phiX_alignment]

        elif self.no_alignments_detected:
            self.register_uncategorized()

        elif self.is_intended_or_partial_replacement:
            self.register_intended_replacement()

        elif self.is_intended_deletion:
            self.category = 'intended edit'
            self.subcategory = 'deletion'
            self.outcome = ProgrammedEditOutcome(self.pegRNA_SNV_string,
                                                 self.non_pegRNA_mismatches_outcome,
                                                 self.non_programmed_edit_mismatches_outcome,
                                                 [self.target_info.pegRNA_programmed_deletion],
                                                )
            self.relevant_alignments = self.intended_edit_relevant_alignments

        elif self.is_intended_integration:
            self.register_intended_integration()

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            if len(interesting_indels) == 0:
                if self.starts_at_expected_location:
                    # Need to check in case the intended replacements only involves minimal changes. 
                    if self.is_intended_or_partial_replacement:
                        self.register_intended_replacement()

                    else:
                        self.category = 'wild type'

                        if len(self.non_pegRNA_mismatches) == 0 and len(uninteresting_indels) == 0:
                            self.subcategory = 'clean'
                            self.outcome = Outcome('n/a')

                        elif len(uninteresting_indels) == 1:
                            self.subcategory = 'short indel far from cut'

                            indel = uninteresting_indels[0]
                            if indel.kind == 'D':
                                self.outcome = DeletionOutcome(indel)
                            elif indel.kind == 'I':
                                self.outcome = InsertionOutcome(indel)
                            else:
                                raise ValueError(indel.kind)

                        elif len(uninteresting_indels) > 1:
                            self.register_uncategorized()

                        else:
                            self.subcategory = 'mismatches'
                            self.outcome = MismatchOutcome(self.non_pegRNA_mismatches)

                        self.relevant_alignments = [target_alignment]

                else:
                    self.register_uncategorized()

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.non_pegRNA_mismatches) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    if indel == self.target_info.pegRNA_programmed_deletion:
                        self.category = 'intended edit'
                        self.subcategory = 'deletion'
                        self.relevant_alignments = [target_alignment] + self.pegRNA_extension_als_list

                    else:
                        self.category = 'deletion'
                        self.relevant_alignments = self.target_edge_alignments_list

                    self.outcome = DeletionOutcome(indel)

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.outcome = InsertionOutcome(indel)
                    self.relevant_alignments = [target_alignment]

            else: # more than one indel
                self.register_uncategorized()

        elif self.inversion:
            self.category = 'inversion'
            self.subcategory = 'inversion'
            self.details = 'n/a'

            self.relevant_alignments = self.target_edge_alignments_list + self.inversion

        elif self.is_multistep_unintended_rejoining:
            self.register_multistep_unintended_rejoining()

        elif len(self.has_any_flipped_pegRNA_al) > 0:
            self.category = 'flipped pegRNA incorporation'

            if len(self.has_any_flipped_pegRNA_al) == 1:
                side = sorted(self.has_any_flipped_pegRNA_al)[0]
                self.subcategory = f'{side} pegRNA'

            elif len(self.has_any_flipped_pegRNA_al) == 2:
                self.subcategory = f'both pegRNAs'

            else:
                raise ValueError(len(self.has_any_flipped_pegRNA_al))

            self.outcome = Outcome('n/a')
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.original_target_alignment_has_only_relevant_indels:
            self.register_indels_in_original_alignment()

        else:
            self.register_uncategorized()

        self.relevant_alignments = self.target_edge_alignments_list + [
            al for al in self.target_alignments + sam.make_noncontained(self.non_protospacer_pegRNA_alignments) + sam.make_noncontained(self.donor_alignments)
            if not self.overlaps_outside_primers(al)
        ] + self.extra_alignments

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome

    def plot(self, **kwargs):
        ti = self.target_info

        features_to_show = set()
        label_overrides = {}
        label_offsets = {}

        refs_to_draw = {
            ti.target,
            *ti.pegRNA_names
        }

        if 'features_to_show' in kwargs:
            features_to_show.update(kwargs.pop('features_to_show'))

        diagram = super().plot(features_to_show=features_to_show,
                               #alignment_registration='centered on primers',
                               #split_at_indels=False,
                               label_overrides=label_overrides,
                               label_offsets=label_offsets,
                               refs_to_draw=refs_to_draw,
                               donor_below=False,
                               #draw_sequence=False,
                               high_resolution_parallelograms=False,
                               annotate_overlap=False,
                               flip_target=self.sequencing_direction == '-',
                               flip_donor=self.sequencing_direction == '-',
                               **kwargs,
                              )

        return diagram
