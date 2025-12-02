import itertools

import numpy as np

import knock_knock.architecture
import knock_knock.outcome
import knock_knock.visualize.architecture

from hits import interval, sam, sw
from hits.utilities import memoized_property

from knock_knock.outcome import Details

class Architecture(knock_knock.architecture.Categorizer):
    category_order = [
        (
            'WT',
            (
                'WT',
            ),
        ),
        (
            'simple indel',
            (
                'insertion',
                'deletion',
                'deletion <50 nt',
                'deletion >=50 nt',
            ),
        ),
        (
            'complex indel',
            (
                'complex indel',
                'multiple indels',
            ),
        ),
        (
            'HDR',
            (
                'HDR',
            ),
        ),
        (
            'blunt misintegration',
            (
                "5' HDR, 3' blunt",
                "5' blunt, 3' HDR",
                "5' blunt, 3' blunt",
                "5' blunt, 3' imperfect",
                "5' imperfect, 3' blunt",
            ),
        ),
        (
            'incomplete HDR',
            (
                "5' HDR, 3' imperfect",
                "5' imperfect, 3' HDR",
            ),
        ),
        (
            'donor fragment',
            (
                "5' imperfect, 3' imperfect",
            ),
        ),
        (
            'complex misintegration',
            (
                'complex misintegration',
            ),
        ),
        (
            'concatenated misintegration',
            (
                'HDR',
                '5\' blunt',
                '3\' blunt',
                '5\' and 3\' blunt',
                'incomplete',
            ),
        ),
        (
            'non-homologous donor',
            (
                'simple',
                'complex',
            ),
        ),
        (
            'genomic insertion',
            (
                'hg38',
                'hg19',
                'mm10',
                'e_coli',
            ),
        ),
        (
            'uninformative',
            (
                'uninformative',
            ),
        ),
        (
            'uncategorized',
            (
                'uncategorized',
                'donor with indel',
                'mismatch(es) near cut',
                'multiple indels near cut',
                'donor specific present',
                'other',
            ),
        ),
        (
            'unexpected source',
            (
                'flipped',
                'e coli',
                'uncategorized',
            ),
        ),
        (
            'nonspecific amplification',
            (
                'hg38',
                'hg19',
                'mm10',
            ),
        ),
        (
            'malformed layout',
            (
                'extra copy of primer',
                'missing a primer',
                'too short',
                'primer far from read edge',
                'primers not in same orientation',
                'no alignments detected',
            ),
        ),
        (
            'bad sequence',
            (
                'non-overlapping',
            ),
        ),
    ]

    non_relevant_categories = [
        'phiX',
        'nonspecific amplification',
        'minimal alignment to intended target',
        'malformed layout',
        'uninformative',
    ]

    def __init__(self, alignments, editing_strategy, error_corrected=False, platform='illumina'):
        super().__init__(alignments, editing_strategy, error_corrected=error_corrected, platform=platform)

        self.error_corrected = error_corrected

        if self.platform == 'illumina':
            self.max_indel_allowed_in_donor = 1
        elif self.platform == 'pacbio':
            self.max_indel_allowed_in_donor = 3

        self.ignore_target_outside_amplicon = True

    @memoized_property
    def target_alignments(self):
        return self.refined_target_alignments

    @memoized_property
    def donor_alignments(self):
        if self.editing_strategy.donor is None:
            return []

        original_als = [al for al in self.alignments if al.reference_name == self.editing_strategy.donor]
        processed_als = []

        for al in original_als:
            split_als = self.comprehensively_split_alignment(al)
            processed_als.extend(split_als)

        return processed_als
    
    @memoized_property
    def nonhomologous_donor_alignments(self):
        if self.editing_strategy.nonhomologous_donor is None:
            return []

        original_als = [al for al in self.alignments if al.reference_name == self.editing_strategy.nonhomologous_donor]
        processed_als = []

        for al in original_als:
            split_als = self.comprehensively_split_alignment(al)
            processed_als.extend(split_als)

        return processed_als

    @memoized_property
    def nonredundant_supplemental_alignments(self):
        primary_als = self.parsimonious_and_gap_alignments + self.nonhomologous_donor_alignments + self.extra_alignments
        covered = interval.get_disjoint_covered(primary_als)

        supp_als_to_keep = []

        for al in self.supplemental_alignments:
            if interval.get_covered(al) - covered:
                supp_als_to_keep.append(al)

        supp_als_to_keep = sorted(supp_als_to_keep, key=lambda al: al.query_alignment_length, reverse=True)
        return supp_als_to_keep

    @memoized_property
    def target_and_donor_alignments(self):
        return self.target_alignments + self.donor_alignments

    @memoized_property
    def HA_names_by_side_of_target(self):
        HA_names = {
            target_side: self.editing_strategy.homology_arms[target_side]['target'].ID
            for target_side in [5, 3]
        }
        return HA_names

    @memoized_property
    def HA_names_by_side_of_read(self):
        read_side_to_HA_name = {}
        
        for read_side, target_side in self.read_side_to_target_side.items():
            read_side_to_HA_name[read_side] = self.HA_names_by_side_of_target[target_side]
            
        return read_side_to_HA_name

    @memoized_property
    def extension_chain_link_specifications(self):
        links = [
            (self.target_alignments, ('first target', 'second target')),
                self.HA_names_by_side_of_target[5],
            (self.donor_alignments, ('donor', 'donor')),
                self.HA_names_by_side_of_target[3],
            (self.target_alignments, ('second target', 'first target')),
        ]

        last_al_to_description = {
            'none': 'no target',
            'first target': 'not HDR\'ed',
            'donor': 'incomplete HDR',
            'second target': 'HDR',
        }

        specs = {'HDR': (links, last_al_to_description)}

        return specs
    
    @memoized_property
    def is_intended_edit(self):
        chains = self.reconciled_extension_chains()['HDR']

        return (
            chains['left'].description == 'HDR' and
            chains['right'].description == 'HDR' and 
            chains['left'].query_covered == chains['right'].query_covered
        )

    def register_single_read_covering_target_alignment(self):
        target_alignment = self.single_read_covering_target_alignment

        interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

        if self.platform in ['pacbio', 'ont', 'nanopore']:
            deletions = [indel for indel in interesting_indels if indel.kind == 'D']
            insertions = [indel for indel in interesting_indels if indel.kind == 'I']

            mismatches = []
        else:
            deletions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'D']
            insertions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'I']

            _, mismatches = self.summarize_mismatches_in_alignments([target_alignment])

        self.details = Details(
            deletions=deletions,
            insertions=insertions,
            mismatches=mismatches,
        )

        self.relevant_alignments = [target_alignment]

        if len(self.mismatches_near_cut) > 0:
            self.category = 'uncategorized'
            self.subcategory = 'mismatch(es) near cut'

        elif len(interesting_indels) == 0:
            self.category = 'WT'
            self.subcategory = 'WT'

        elif len(interesting_indels) == 1:
            self.category = 'simple indel'

            indel = interesting_indels[0]

            if indel.kind == 'D':
                if indel.length < 50:
                    self.subcategory = 'deletion <50 nt'
                else:
                    self.subcategory = 'deletion >=50 nt'

            elif indel.kind == 'I':
                self.subcategory = 'insertion'

        else: # more than one indel
            self.category = 'uncategorized'
            self.subcategory = 'multiple indels near cut'

    @memoized_property
    def is_uniformative(self):
        ''' For non-amplicon strategies, reads may not span the edited region. '''
        num_valid_flanking_alignments = sum(al is not None for al in self.target_flanking_alignments.values())
        return (len(self.editing_strategy.primers) == 0) and (num_valid_flanking_alignments != 2)

    def register_uniformative(self):
        self.category = 'uninformative'
        self.subcategory = 'uninformative'

        self.relevant_alignments = self.parsimonious_and_gap_alignments

    def categorize(self):
        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        if self.seq is None or len(self.seq) <= self.min_relevant_length:
            self.category = 'malformed layout'
            self.subcategory = 'too short'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif all(al.is_unmapped for al in self.target_and_donor_alignments):
            self.category = 'malformed layout'
            self.subcategory = 'no alignments detected'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.extra_copy_of_primer:
            self.category = 'malformed layout'
            self.subcategory = 'extra copy of primer'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.is_uniformative:
            self.register_uniformative()

        elif self.single_read_covering_target_alignment:
            self.register_single_read_covering_target_alignment()

        elif self.is_intended_edit:
            self.category = 'HDR'
            self.subcategory = 'HDR'
            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.integration_summary == 'donor':
            junctions = set(self.junction_summary_per_side.values())

            if self.gap_covered_by_target_alignment:
                self.category = 'complex indel'
                self.subcategory = 'complex indel'
                self.relevant_alignments = self.parsimonious_and_gap_alignments

            elif junctions == set(['imperfect']):
                if self.not_covered_by_simple_integration.total_length >= 2:
                    self.category = 'complex misintegration'
                    self.subcategory = 'complex misintegration'
                else:
                    self.category = 'donor fragment'
                    self.subcategory = f'5\' {self.junction_summary_per_side[5]}, 3\' {self.junction_summary_per_side[3]}'

                self.relevant_alignments = self.parsimonious_and_gap_alignments

            else:
                self.subcategory = f'5\' {self.junction_summary_per_side[5]}, 3\' {self.junction_summary_per_side[3]}'

                if 'blunt' in junctions:
                    self.category = 'blunt misintegration'

                elif junctions == set(['imperfect', 'HDR']):
                    if self.not_covered_by_simple_integration.total_length >= 2:
                        self.category = 'complex misintegration'
                        self.subcategory = 'complex misintegration'
                    else:
                        self.category = 'incomplete HDR'

                else:
                    self.category = 'complex misintegration'
                    self.subcategory = 'complex misintegration'

                self.relevant_alignments = self.parsimonious_and_gap_alignments
        
        # TODO: check here for HA extensions into donor specific
        elif self.gap_covered_by_target_alignment:
            self.category = 'complex indel'
            self.subcategory = 'complex indel'
            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.integration_interval.total_length <= 5:
            if self.target_to_at_least_cut[5] and self.target_to_at_least_cut[3]:
                self.category = 'simple indel'
                self.subcategory = 'insertion'

            else:
                self.category = 'complex indel'
                self.subcategory = 'complex indel'

            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.integration_summary == 'concatamer':
            if self.editing_strategy.donor_type == 'plasmid':
                self.category = 'complex misintegration'
                self.subcategory = 'complex misintegration'
            else:
                self.category = 'concatenated misintegration'
                self.subcategory = self.junction_summary

            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.nonhomologous_donor_integration is not None:
            self.category = 'non-homologous donor'
            self.subcategory = 'simple'

            NH_al = self.nonhomologous_donor_alignments[0]
            NH_strand = sam.get_strand(NH_al)
            MH_nts = self.NH_donor_microhomology
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonhomologous_donor_alignments

        elif self.nonspecific_amplification is not None:
            self.register_nonspecific_amplification()

        elif self.genomic_insertion is not None:
            self.register_genomic_insertion()

        elif self.partial_nonhomologous_donor_integration is not None:
            self.category = 'non-homologous donor'
            self.subcategory = 'complex'
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonhomologous_donor_alignments + self.nonredundant_supplemental_alignments
        
        elif self.any_donor_specific_present:
            self.category = 'complex misintegration'
            self.subcategory = 'complex misintegration'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.integration_summary in ['donor with indel', 'other', 'unexpected length', 'unexpected source']:
            self.category = 'uncategorized'
            self.subcategory = self.integration_summary

            self.relevant_alignments = self.uncategorized_relevant_alignments

        else:
            print(self.integration_summary)

        return self.category, self.subcategory, self.details
    
    @memoized_property
    def gap_alignments(self):
        gap_als = []

        gap = self.gap_between_primer_alignments
        if len(gap) >= 4:
            seq_bytes = self.seq.encode()
            for on in ['target', 'donor']:
                aligner = self.editing_strategy.seed_and_extender[on]
                als = aligner(seq_bytes, gap.start, gap.end, self.name)
                als = sorted(als, key=lambda al: al.query_alignment_length, reverse=True)
                # For same reasoning as in target_alignments, only consider als that overlap the amplicon interval.
                if on == 'target':
                    als = [al for al in als if (self.editing_strategy.amplicon_interval & interval.get_covered_on_ref(al))]

                gap_als.extend(als[:10])

        return gap_als

    @memoized_property
    def not_covered_by_initial_alignments(self):
        uncovered = self.between_primers - interval.get_disjoint_covered(self.target_and_donor_alignments)
        return uncovered

    @memoized_property
    def not_covered_by_refined_alignments(self):
        uncovered = self.between_primers - interval.get_disjoint_covered(self.parsimonious_and_gap_alignments)
        return uncovered

    @memoized_property
    def not_covered_by_simple_integration(self):
        ''' Length of read not covered by target flanking alignments and the
        longest parsimonious donor integration alignment.
        '''

        als = [
            self.donor_specific_integration_alignments[0],
            self.target_flanking_alignments['left'],
            self.target_flanking_alignments['right'],
        ]

        uncovered = self.between_primers - interval.get_disjoint_covered(als)

        return uncovered

    @memoized_property
    def sw_gap_alignments(self):
        gap_covers = []

        if self.platform == 'illumina':
            strat = self.editing_strategy

            
            target_interval = strat.amplicon_interval
            
            for gap in self.not_covered_by_initial_alignments:
                if gap.total_length == 1:
                    continue

                start = max(0, gap.start - 5)
                end = min(len(self.seq) - 1, gap.end + 5)
                extended_gap = interval.Interval(start, end)

                als = sw.align_read(self.read,
                                    [(strat.target, strat.target_sequence),
                                    ],
                                    4,
                                    strat.header,
                                    N_matches=False,
                                    max_alignments_per_target=5,
                                    read_interval=extended_gap,
                                    ref_intervals={strat.target: target_interval},
                                    mismatch_penalty=-2,
                                )

                als = [sw.extend_alignment(al, strat.reference_sequence_bytes[strat.target]) for al in als]
                
                gap_covers.extend(als)

                if strat.donor is not None:
                    als = sw.align_read(self.read,
                                        [(strat.donor, strat.donor_sequence),
                                        ],
                                        4,
                                        strat.header,
                                        N_matches=False,
                                        max_alignments_per_target=5,
                                        read_interval=extended_gap,
                                        mismatch_penalty=-2,
                                    )

                    als = [sw.extend_alignment(al, strat.reference_sequence_bytes[strat.donor]) for al in als]
                    
                    gap_covers.extend(als)

        return gap_covers

    @memoized_property
    def possibly_imperfect_gap_alignments(self):
        gap_als = []
        if self.gap_between_primer_alignments.total_length >= 10:
            for al in self.target_alignments:
                if (self.integration_interval - interval.get_covered(al)).total_length <= 2:
                    gap_als.append(al)

        return gap_als

    @memoized_property
    def all_target_gap_alignments(self):
        all_gap_als = self.gap_alignments + self.possibly_imperfect_gap_alignments
        return [al for al in all_gap_als if al.reference_name == self.editing_strategy.target]

    @memoized_property
    def gap_covered_by_target_alignment(self):
        return len(self.all_target_gap_alignments) > 0

    @memoized_property
    def extra_copy_of_primer(self):
        ''' Check if more than one alignment containing either primer were found. '''
        return any(len(als) > 1 for als in self.all_primer_alignments.values())
    
    @memoized_property
    def missing_a_primer(self):
        ''' Check if either primer was not found in any alignment. '''
        return any(len(als) == 0 for als in self.all_primer_alignments.values())

    @memoized_property
    def target_flanking_alignments(self):
        target_flanking_alignments = super().target_flanking_alignments

        # To unambiguously be from the target, require extending outside of the homology arms.
        for target_side in [5, 3]:
            HA = self.editing_strategy.homology_arms[target_side]['target']

            if target_side == 5:
                crop_start = 0
                crop_end = HA.start - 1
            else:
                crop_start = HA.end + 1
                crop_end = len(self.editing_strategy.target_sequence)

            read_side = self.target_side_to_read_side[target_side]
            cropped_al = sam.crop_al_to_ref_int(target_flanking_alignments[read_side], crop_start, crop_end)

            if cropped_al is None:
                target_flanking_alignments[read_side] = None

        return target_flanking_alignments
        
    @memoized_property
    def covered_by_target_flanking_alignments(self):
        return interval.get_disjoint_covered(self.target_flanking_alignments_list)

    def overlaps_donor_specific(self, al):
        strat = self.editing_strategy
        if strat.donor is None:
            return False
        elif al.reference_name != strat.donor:
            return False
        else:
            covered = interval.get_covered_on_ref(al)
            overlap = covered & strat.donor_specific_intervals
            return overlap.total_length > 0

    @memoized_property
    def any_donor_specific_present(self):
        als_to_check = self.parsimonious_and_gap_alignments
        return any(self.overlaps_donor_specific(al) for al in als_to_check)

    @memoized_property
    def has_integration(self):
        return (self.single_read_covering_target_alignment is None) and (self.not_covered_by_target_flanking_alignments.total_length > 0)

    @memoized_property
    def mismatches_near_cut(self):
        merged_primer_al = self.single_read_covering_target_alignment
        if merged_primer_al is None:
            return []
        else:
            mismatches = []
            tuples = sam.aligned_tuples(merged_primer_al, self.editing_strategy.target_sequence)
            for true_read_i, read_b, ref_i, ref_b, qual in tuples:
                if ref_i is not None and true_read_i is not None:
                    if read_b != ref_b and ref_i in self.editing_strategy.around_cuts(5):
                        mismatches.append(ref_i)

            return mismatches

    @memoized_property
    def parsimonious_alignments(self):
        return interval.make_parsimonious(self.target_and_donor_alignments)

    @memoized_property
    def parsimonious_and_gap_alignments(self):
        ''' identification of gap_alignments requires further processing of parsimonious alignments '''
        return sam.make_nonredundant(interval.make_parsimonious(self.parsimonious_alignments + self.sw_gap_alignments))

    @memoized_property
    def parsimonious_target_alignments(self):
        return [al for al in self.parsimonious_and_gap_alignments if al.reference_name == self.editing_strategy.target]

    @memoized_property
    def parsimonious_donor_alignments(self):
        return [al for al in self.parsimonious_and_gap_alignments if al.reference_name == self.editing_strategy.donor]

    @memoized_property
    def closest_donor_alignment_to_edge(self):
        ''' Identify the alignments to the donor closest to edge of the read
        that has the PAM-proximal and PAM-distal amplicon primer. '''
        donor_als = self.parsimonious_donor_alignments

        if self.sequencing_direction is None or len(donor_als) == 0:
            closest = {5: None, 3: None}
        else:
            closest = {}

            left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
            right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)

            if self.sequencing_direction == '+':
                closest[5] = left_most
                closest[3] = right_most
            else:
                closest[5] = right_most
                closest[3] = left_most

        return closest

    @memoized_property
    def clean_handoff(self):
        ''' Check if target sequence cleanly transitions to donor sequence at
        each junction between the two.
        '''

        clean_handoff = {
            target_side: 'donor' in self.reconciled_extension_chains()['HDR'][self.target_side_to_read_side[target_side]].alignments
            for target_side in [5, 3]
        }

        return clean_handoff
    
    @memoized_property
    def edge_r(self):
        ''' Where in the donor are the edges of the integration? '''
        all_edge_rs = {
            5: [],
            3: [],
        }

        for al in self.parsimonious_donor_alignments:
            cropped = sam.crop_al_to_query_int(al, self.integration_interval.start, self.integration_interval.end)
            if cropped is None:
                continue
            start = cropped.reference_start
            end = cropped.reference_end - 1
            all_edge_rs[5].append(start)
            all_edge_rs[3].append(end)

        edge_r = {}

        if all_edge_rs[5]:
            edge_r[5] = min(all_edge_rs[5])
        else:
            edge_r[5] = None

        if all_edge_rs[3]:
            edge_r[3] = max(all_edge_rs[3])
        else:
            edge_r[3] = None

        return edge_r

    @memoized_property
    def donor_integration_is_blunt(self):
        donor_length = len(self.editing_strategy.donor_sequence)

        reaches_end = {
            5: self.edge_r[5] is not None and self.edge_r[5] <= 1,
            3: self.edge_r[3] is not None and self.edge_r[3] >= donor_length - 2,
        }

        short_gap = {}
        for side in [5, 3]:
            read_side = self.target_side_to_read_side[side]

            primer_al = self.target_flanking_alignments[read_side]
            donor_al = self.closest_donor_alignment_to_edge[side]
            overlap = knock_knock.architecture.junction_microhomology(self.editing_strategy.reference_sequences, primer_al, donor_al)

            short_gap[side] = overlap > -10

        is_blunt = {side: reaches_end[side] and short_gap[side] for side in [5, 3]}

        return is_blunt

    @memoized_property
    def integration_interval(self):
        ''' because cut site might not exactly coincide with boundary between
        HAs, the relevant part of query to call integration depends on whether
        a clean HDR handoff is detected at each edge '''

        if not self.has_integration:
            return interval.Interval.empty()

        HAs = self.editing_strategy.homology_arms
        cut_after = self.editing_strategy.cut_after

        flanking_al = {}
        mask_start = {5: -np.inf}
        mask_end = {3: np.inf}
        for side in [5, 3]:
            if self.clean_handoff[side]:
                flanking_al[side] = self.closest_donor_alignment_to_edge[side]
            else:
                flanking_al[side] = self.target_flanking_alignments[self.target_side_to_read_side[side]]

        if self.clean_handoff[5] or cut_after is None:
            mask_end[5] = HAs[5]['donor'].end
        else:
            mask_end[5] = cut_after

        if self.clean_handoff[3] or cut_after is None:
            mask_start[3] = HAs[3]['donor'].start
        else:
            mask_start[3] = cut_after + 1

        covered = {
            side: (sam.crop_al_to_ref_int(flanking_al[side], mask_start[side], mask_end[side])
                   if flanking_al[side] is not None else None
                  )
            for side in [5, 3]
        }

        if self.sequencing_direction == '+':
            if covered[5] is not None:
                start = interval.get_covered(covered[5]).end + 1
            else:
                start = 0

            if covered[3] is not None:
                end = interval.get_covered(covered[3]).start - 1
            else:
                end = len(self.seq) - 1

        elif self.sequencing_direction == '-':
            if covered[5] is not None:
                end = interval.get_covered(covered[5]).start - 1
            else:
                end = len(self.seq) - 1

            if covered[3] is not None:
                start = interval.get_covered(covered[3]).end + 1
            else:
                start = 0

        return interval.Interval(start, end)

    @memoized_property
    def gap_between_primer_alignments(self):
        if self.target_flanking_alignments['left'] is None or self.target_flanking_alignments['right'] is None:
            return interval.Interval.empty()

        left_covered = interval.get_covered(self.target_flanking_alignments['left'])
        right_covered = interval.get_covered(self.target_flanking_alignments['right'])

        between_primers = interval.Interval(left_covered.start, right_covered.end)

        gap = between_primers - left_covered - right_covered
        
        return gap

    @memoized_property
    def target_to_at_least_cut(self):
        cut_after = self.editing_strategy.cut_after

        target_to_at_least_cut = {
            5: (al := self.target_flanking_alignments_by_target_side[5]) is not None and al.reference_end - 1 >= cut_after,
            3: (al := self.target_flanking_alignments_by_target_side[3]) is not None and al.reference_start <= (cut_after + 1),
        }

        return target_to_at_least_cut

    @memoized_property
    def junction_summary_per_side(self):
        per_side = {}

        for side in [5, 3]:
            if self.clean_handoff[side]:
                per_side[side] = 'HDR'
            elif self.donor_integration_is_blunt[side]:
                per_side[side] = 'blunt'
            else:
                per_side[side] = 'imperfect'

        return per_side
                
    @memoized_property
    def junction_summary(self):
        per_side = self.junction_summary_per_side

        if (per_side[5] == 'HDR' and
            per_side[3] == 'HDR'):

            summary = 'HDR'

        elif (per_side[5] == 'blunt' and
              per_side[3] == 'HDR'):

            summary = "5' blunt"
        
        elif (per_side[5] == 'HDR' and
              per_side[3] == 'blunt'):

            summary = "3' blunt"
        
        elif (per_side[5] == 'blunt' and
              per_side[3] == 'blunt'):

            summary = "5' and 3' blunt"

        else:
            summary = 'incomplete'

        # blunt isn't a meaningful concept for plasmid donors
        if self.editing_strategy.donor_type == 'plasmid':
            if 'blunt' in summary:
                summary = 'incomplete'

        return summary

    @memoized_property
    def donor_specific_integration_alignments(self):
        integration_donor_als = []

        for al in self.parsimonious_donor_alignments:
            if self.overlaps_donor_specific(al):
                covered = interval.get_covered(al)
                if (self.integration_interval.total_length > 0) and ((self.integration_interval - covered).total_length == 0):
                    # If a single donor al covers the whole integration, use just it.
                    integration_donor_als = [al]
                    break
                else:
                    covered_integration = self.integration_interval & interval.get_covered(al)
                    # Ignore als that barely extend past the homology arms.
                    if len(covered_integration) >= 5:
                        integration_donor_als.append(al)

        return sorted(integration_donor_als, key=lambda al: al.query_alignment_length, reverse=True)

    @memoized_property
    def integration_summary(self):
        if len(self.donor_specific_integration_alignments) == 0:
            summary = 'other'

        elif len(self.donor_specific_integration_alignments) == 1:
            donor_al = self.donor_specific_integration_alignments[0]
            covered_by_donor = interval.get_covered(donor_al)
            uncovered_length = (self.integration_interval - covered_by_donor).total_length

            if uncovered_length > 10:
                summary = 'other'
            else:
                max_indel_length = sam.max_block_length(donor_al, {sam.BAM_CDEL, sam.BAM_CINS})
                if max_indel_length > self.max_indel_allowed_in_donor:
                    summary = 'donor with indel'
                else:
                    summary = 'donor'

        else:
            if self.cleanly_concatanated_donors > 1:
                summary = 'concatamer'

            else:
                #TODO: check for plasmid extensions around the boundary
                summary = 'other'

        return summary
    
    @memoized_property
    def cleanly_concatanated_donors(self):
        strat = self.editing_strategy

        HAs = strat.homology_arms
        p_donor_als = self.parsimonious_donor_alignments

        if len(p_donor_als) <= 1:
            return 0

        # TEMPORARY
        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            # The donor doesn't share homology arms with the target.
            return 0

        if self.sequencing_direction == '+':
            key = lambda al: interval.get_covered(al).start
            reverse = False
        else:
            key = lambda al: interval.get_covered(al).end
            reverse = True

        five_to_three = sorted(p_donor_als, key=key, reverse=reverse)
        junctions_clean = []

        for before, after in zip(five_to_three[:-1], five_to_three[1:]):
            before_int = interval.get_covered(before)
            after_int = interval.get_covered(after)

            adjacent = interval.are_adjacent(before_int, after_int)
            overlap_slightly = len(before_int & after_int) <= 2

            missing_before = len(strat.donor_sequence) - before.reference_end
            missing_after = after.reference_start 

            clean = (adjacent or overlap_slightly) and (missing_before <= 1) and (missing_after <= 1)

            junctions_clean.append(clean)

        if all(junctions_clean):
            return len(junctions_clean) + 1
        else:
            return 0
    
    @memoized_property
    def indels(self):
        indels = []

        al = self.single_read_covering_target_alignment

        if al is not None:
            for i, (kind, length) in enumerate(al.cigar):
                if kind == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before

                    indel = knock_knock.outcome.DegenerateDeletion([starts_at], length)

                elif kind == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    indel = knock_knock.outcome.DegenerateInsertion([starts_after], [insertion])
                    
                else:
                    continue

                indels.append(indel)

        return indels

    @memoized_property
    def genomic_insertion(self):
        min_gap_length = 20
        
        covered_by_normal = interval.get_disjoint_covered(self.target_and_donor_alignments)
        unexplained_gaps = self.between_primers - covered_by_normal

        long_unexplained_gaps = [gap for gap in unexplained_gaps if len(gap) >= min_gap_length]

        if len(long_unexplained_gaps) != 1:
            covering_als = None
        elif self.gap_alignments:
            # gap aligns to the target in the amplicon region
            covering_als = None
        else:
            gap = long_unexplained_gaps[0]

            covering_als = []
            for al in self.supplemental_alignments:
                covered = interval.get_covered(al)
                if (gap - covered).total_length <= 3:
                    edit_distance = sam.edit_distance_in_query_interval(al, gap)
                    error_rate = edit_distance / len(gap)
                    if error_rate < 0.1:
                        covering_als.append(al)
                    
            if len(covering_als) == 0:
                covering_als = None

        return covering_als

    def register_genomic_insertion(self):
        insertion_al = self.min_edit_distance_genomic_insertions[0]

        organism, original_al = self.editing_strategy.remove_organism_from_alignment(insertion_al)

        self.category = 'genomic insertion'
        self.subcategory = organism

        alignments = self.parsimonious_and_gap_alignments + self.parsimonious_donor_alignments + self.min_edit_distance_genomic_insertions
        self.relevant_alignments = interval.make_parsimonious(alignments)

    @memoized_property
    def one_sided_covering_als(self):
        all_covering_als = {
            'nonspecific_amplification': None,
            'genomic_insertion': None,
            'h': None,
            'nh': None,
        }
        
        if self.sequencing_direction == '+':
            primer_al = self.target_flanking_alignments[5]
        elif self.sequencing_direction == '-':
            primer_al = self.target_flanking_alignments[3]
        else:
            return all_covering_als

        covered = interval.get_covered(primer_al)

        close_to_start = primer_al is not None and covered.start <= 10

        if not close_to_start:
            return all_covering_als

        # from supplementary alignments

        has_extra = self.extra_query_in_primer_als['left'] >= 20

        if has_extra:
            kind = 'genomic_insertion'
            primer_interval = interval.get_covered(primer_al)
            primer_interval.start = 0
        else:
            kind = 'nonspecific_amplification'
            primer_interval = self.just_primer_interval['left']
            
        need_to_cover = self.whole_read - primer_interval
        covering_als = []
        for supp_al in self.supplemental_alignments:
            if (need_to_cover - interval.get_covered(supp_al)).total_length <= 10:
                covering_als.append(supp_al)
                
        if covering_als:
            all_covering_als[kind] = covering_als

        # from donor and nh-donor als

        primer_interval = interval.get_covered(primer_al)
        primer_interval.start = 0
            
        need_to_cover = self.whole_read - primer_interval
        for kind, all_als in [('h', self.parsimonious_donor_alignments),
                              ('nh', self.nonhomologous_donor_alignments),
                             ]:
            covering_als = []
            for al in all_als:
                if (need_to_cover - interval.get_covered(al)).total_length <= 10:
                    covering_als.append(al)
                
            if covering_als:
                all_covering_als[kind] = covering_als

        return all_covering_als
    
    @memoized_property
    def nonhomologous_donor_integration_alignments(self):
        min_gap_length = 10
        gap = self.gap_between_primer_alignments
        
        covered_by_normal = interval.get_disjoint_covered(self.target_and_donor_alignments)
        unexplained_gap = gap - covered_by_normal

        if unexplained_gap.total_length < min_gap_length:
            return [], []
        elif self.gap_alignments:
            # gap aligns to the target in the amplicon region
            return [], []
        else:
            full_covering_als = []
            partial_covering_als = []

            for al in self.nonhomologous_donor_alignments:
                covered = interval.get_covered(al)
                if (gap - covered).total_length <= 2:
                    full_covering_als.append(al)
                
                if (covered & unexplained_gap).total_length >= 2:
                    partial_covering_als.append(al)
                    
            return full_covering_als, partial_covering_als

    @memoized_property
    def nonhomologous_donor_integration(self):
        full_covering_als, partial_covering_als = self.nonhomologous_donor_integration_alignments
        if len(full_covering_als) > 0:
            return full_covering_als
        else:
            return None

    @memoized_property
    def partial_nonhomologous_donor_integration(self):
        full_covering_als, partial_covering_als = self.nonhomologous_donor_integration_alignments
        if self.nonhomologous_donor_integration is not None:
            return None
        elif len(partial_covering_als) == 0:
            return None
        else:
            return partial_covering_als

    @memoized_property
    def min_edit_distance_genomic_insertions(self):
        covering_als = self.genomic_insertion
        if covering_als is None:
            return None
        else:
            edit_distances = [sam.edit_distance_in_query_interval(al) for al in covering_als]
            min_distance = min(edit_distances)
            best_als = [al for al, distance in zip(covering_als, edit_distances) if distance == min_distance]
            return best_als

    @memoized_property
    def extra_query_in_primer_als(self):
        not_primer_length = {'left': 0, 'right': 0}

        if self.sequencing_direction is None:
            return not_primer_length

        for target_side in [5, 3]:
            if (target_side == 5 and self.sequencing_direction == '+') or (target_side == 3 and self.sequencing_direction == '-'):
                read_side = 'left'
            elif (target_side == 3 and self.sequencing_direction == '+') or (target_side == 5 and self.sequencing_direction == '-'):
                read_side = 'right'

            al = self.target_flanking_alignments[target_side]
            if al is None:
                not_primer_length[read_side] = 0
                continue

            not_primer_interval = self.whole_read - self.just_primer_interval[read_side]
            not_primer_al = sam.crop_al_to_query_int(al, not_primer_interval.start, not_primer_interval.end)
            if not_primer_al is None:
                not_primer_length[read_side] = 0
            else:
                not_primer_length[read_side] = not_primer_al.query_alignment_length

        return not_primer_length

    @memoized_property
    def just_primer_interval(self):
        primer_interval = {'left': None, 'right': None}

        if self.sequencing_direction is None:
            return primer_interval

        for target_side in [5, 3]:
            if (target_side == 5 and self.sequencing_direction == '+') or (target_side == 3 and self.sequencing_direction == '-'):
                read_side = 'left'
            elif (target_side == 3 and self.sequencing_direction == '+') or (target_side == 5 and self.sequencing_direction == '-'):
                read_side = 'right'

            al = self.target_flanking_alignments[target_side]
            if al is None:
                primer_interval[read_side] = None
                continue

            primer = self.editing_strategy.primers_by_side_of_target[target_side]
            just_primer_al = sam.crop_al_to_ref_int(al, primer.start, primer.end)
            start, end = sam.query_interval(just_primer_al)
            if read_side == 'left':
                primer_interval[read_side] = interval.Interval(0, end)
            elif read_side == 'right':
                primer_interval[read_side] = interval.Interval(start, len(self.seq) - 1)

        return primer_interval

    @memoized_property
    def uncategorized_relevant_alignments(self):
        sources = [
            self.parsimonious_and_gap_alignments,
            self.nonhomologous_donor_alignments,
            self.extra_alignments,
        ]
        flattened = [al for source in sources for al in source]
        parsimonious = sam.make_nonredundant(interval.make_parsimonious(flattened))

        covered = interval.get_disjoint_covered(parsimonious)
        supp_als = []

        def novel_length(supp_al):
            return (interval.get_covered(supp_al) - covered).total_length

        supp_als = interval.make_parsimonious(self.nonredundant_supplemental_alignments)
        supp_als = sorted(supp_als, key=novel_length, reverse=True)[:10]

        final = parsimonious + supp_als

        return final

    @memoized_property
    def templated_insertion_relevant_alignments(self):
        return sam.make_nonredundant(interval.make_parsimonious(self.parsimonious_target_alignments + self.all_target_gap_alignments))

    @memoized_property
    def donor_microhomology(self):
        if len(self.parsimonious_donor_alignments) == 1:
            donor_al = self.parsimonious_donor_alignments[0]
        else:
            donor_al = None
            
        MH_nts = {side: knock_knock.architecture.junction_microhomology(self.editing_strategy.reference_sequences, self.target_flanking_alignments[side], donor_al) for side in [5, 3]}

        return MH_nts

    @memoized_property
    def NH_donor_microhomology(self):
        if self.nonhomologous_donor_integration:
            nh_al = self.nonhomologous_donor_alignments[0]
        else:
            nh_al = None

        MH_nts = {side: knock_knock.architecture.junction_microhomology(self.editing_strategy.reference_sequences, self.target_flanking_alignments[side], nh_al) for side in [5, 3]}

        return MH_nts


    @memoized_property
    def query_interval_to_plot(self):
        if self.has_target_flanking_alignments_on_both_sides:
            left, right = self.target_flanking_intervals['read']['left'].start, self.target_flanking_intervals['read']['right'].end

            query_interval = (left, right)

        else:
            query_interval = None

        return query_interval

    def plot(self, relevant=True, manual_alignments=None, **manual_diagram_kwargs):
        label_overrides = {}
        label_offsets = {}
        feature_heights = {}

        if relevant and not self.categorized:
            self.categorize()

        strat = self.editing_strategy
        features_to_show = {*strat.features_to_show}

        for name in strat.protospacer_names:
            label_overrides[name] = 'protospacer'
            label_offsets[name] = 1

        refs_to_flip = set()
        refs_to_draw = {strat.target}

        if strat.donor is not None:
            refs_to_draw.add(strat.donor)

        flip_target = (self.sequencing_direction == '-')

        if flip_target:
            refs_to_flip.add(strat.target)

        if len(strat.homology_arms) > 0:
            HAs = strat.homology_arms[5]
            opposite_of_target = (HAs['target'].strand != HAs['donor'].strand)

            if ((not flip_target) and opposite_of_target) or (flip_target and (not opposite_of_target)):
                refs_to_flip.add(strat.donor)

        refs_to_label = refs_to_draw

        label_overrides.update({feature_name: None for feature_name in strat.PAM_features})

        features_to_show.update({(strat.target, name) for name in strat.protospacer_names})
        features_to_show.update({(strat.target, name) for name in strat.PAM_features})

        if self.has_any_target_flanking_alignment > 0:
            manual_anchors = {self.editing_strategy.target: self.target_flanking_alignments_list}
        else:
            manual_anchors = {}

        if manual_alignments is not None:
            als_to_plot = manual_alignments
        elif relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.target_and_donor_alignments

        diagram_kwargs = dict(
            draw_sequence=(self.read_length < 1000),
            split_at_indels=False,
            features_to_show=features_to_show,
            label_offsets=label_offsets,
            label_overrides=label_overrides,
            refs_to_draw=refs_to_draw,
            refs_to_flip=refs_to_flip,
            refs_to_label=refs_to_label,
            inferred_amplicon_length=self.inferred_amplicon_length,
            highlight_programmed_substitutions=True,
            feature_heights=feature_heights,
            platform=self.platform,
            high_resolution_parallelograms=(self.read_length < 1000),
            manual_anchors=manual_anchors,
            query_interval=self.query_interval_to_plot,
        )

        for k, v in diagram_kwargs.items():
            manual_diagram_kwargs.setdefault(k, v)

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 strat,
                                                                 architecture=self,
                                                                 **manual_diagram_kwargs,
                                                                )

        return diagram

class NonoverlappingPairArchitecture:
    def __init__(self, als, editing_strategy):
        self.editing_strategy = editing_strategy
        self.layouts = {
            'R1': Architecture(als['R1'], editing_strategy, platform='illumina'),
            'R2': Architecture(als['R2'], editing_strategy, platform='illumina'),
        }
        if self.layouts['R1'].name != self.layouts['R2'].name:
            raise ValueError
        
        self.name = self.layouts['R1'].name
        self.query_name = self.name

    @memoized_property
    def bridging_alignments(self):
        bridging_als = {
            'h': {'R1': None, 'R2': None},
            'nh': {'R1': None, 'R2': None},
        }
        
        for which in ['R1', 'R2']:
            if self.layouts[which].has_integration:
                for kind in ['h', 'nh']:
                    als = self.layouts[which].one_sided_covering_als[kind]
                    if als is not None and len(als) == 1:
                        bridging_als[kind][which] = als[0]

        bridging_als.update(self.best_genomic_al_pairs)
        
        return bridging_als

    @memoized_property
    def target_sides(self):
        target_sides = {}

        for which in ['R1', 'R2']:
            primer_als = self.layouts[which].target_flanking_alignments
            sides = set(s for s, al in primer_als.items() if al is not None)
            if len(sides) == 1:
                side = sides.pop()
            else:
                side = None

            target_sides[which] = side

        return target_sides

    @memoized_property
    def sequencing_direction(self):
        if self.target_sides['R1'] == 5 and self.target_sides['R2'] == 3:
            strand = '+'
        elif self.target_sides['R2'] == 5 and self.target_sides['R1'] == 3:
            strand = '-'
        else:
            strand = None

        return strand

    @memoized_property
    def best_genomic_al_pairs(self):
        best_pairs = {}
        for kind in ['nonspecific_amplification', 'genomic_insertion']:
            best_pairs[kind] = {'R1': None, 'R2': None}
            
            als = {which: self.layouts[which].one_sided_covering_als[kind] for which in ['R1', 'R2']}
            if als['R1'] is None or als['R2'] is None:
                continue
                
            valid_pairs = {}
            for R1_al, R2_al in itertools.product(als['R1'], als['R2']):
                if R1_al.reference_name != R2_al.reference_name:
                    continue

                if sam.get_strand(R1_al) == '+':
                    if sam.get_strand(R2_al) != '-':
                        # should be in opposite orientation if concordant
                        continue
                    start = R1_al.reference_start
                    end = R2_al.reference_end
                elif sam.get_strand(R1_al) == '-':
                    if sam.get_strand(R2_al) != '+':
                        continue
                    start = R2_al.reference_start
                    end = R1_al.reference_end

                length = end - start

                if 0 < length < 2000:
                    # Note: multiple valid pairs with same length are discarded.
                    valid_pairs[length] = {'R1': R1_al, 'R2': R2_al}

            if valid_pairs:
                length = min(valid_pairs)

                best_pairs[kind] = valid_pairs[length]
                
        return best_pairs

    def register_genomic_insertion(self):
        als = self.best_genomic_al_pairs['genomic_insertion']

        R1_al = als['R1']
        R2_al = als['R2']

        organism, original_al = self.editing_strategy.remove_organism_from_alignment(R1_al)

        # TODO: these need to be cropped.

        target_ref_bounds = {
            'left': sam.reference_edges(self.layouts['R1'].target_flanking_alignments[5])[3],
            'right': sam.reference_edges(self.layouts['R2'].target_flanking_alignments[3])[3],
        }

        insertion_ref_bounds = {
            'left': sam.reference_edges(R1_al)[5],
            'right': sam.reference_edges(R2_al)[5],
        }

        insertion_query_bounds = {
            'left': sam.query_interval(R1_al)[0],
            'right': self.inferred_amplicon_length - 1 - sam.query_interval(R2_al)[0],
        }

        outcome = knock_knock.outcome.LongTemplatedInsertionOutcome(organism,
                                                original_al.reference_name,
                                                sam.get_strand(R1_al),
                                                insertion_ref_bounds['left'],
                                                insertion_ref_bounds['right'],
                                                insertion_query_bounds['left'],
                                                insertion_query_bounds['right'],
                                                target_ref_bounds['left'],
                                                target_ref_bounds['right'],
                                                -1,
                                                -1,
                                                -1,
                                                -1,
                                                '',
                                               )

        self.outcome = outcome

        self.category = 'genomic insertion'
        self.subcategory = organism

    def register_nonspecific_amplification(self):
        als = self.best_genomic_al_pairs['nonspecific_amplification']

        al = als['R1']

        organism, original_al = self.editing_strategy.remove_organism_from_alignment(al)

        self.category = 'nonspecific amplification'
        self.subcategory = organism

    @memoized_property
    def bridging_als_missing_from_end(self):
        missing = {k: {'R1': None, 'R2': None} for k in self.bridging_alignments}

        for kind in self.bridging_alignments:
            for which in ['R1', 'R2']:
                al = self.bridging_alignments[kind][which]
                if al is not None:
                    covered = interval.get_covered(al)
                    missing[kind][which] = len(self.layouts[which].seq) - 1 - covered.end

        return missing

    @memoized_property
    def bridging_als_reach_internal_edges(self):
        missing = self.bridging_als_missing_from_end
        reach_edges = {}
        for kind in self.bridging_alignments:
            reach_edges[kind] = all(m is not None and m <= 5 for m in missing[kind].values())

        return reach_edges

    @memoized_property
    def junctions(self):
        junctions = {
            'R1': 'uncategorized',
            'R2': 'uncategorized',
            5: 'uncategorized',
            3: 'uncategorized',
        }

        for side in ['R1', 'R2']:
            target_side = self.target_sides.get(side)
            if target_side is not None:
                junction = self.layouts[side].junction_summary_per_side[target_side]
                junctions[side] = junction
                junctions[target_side] = junction

        return junctions

    @property
    def possible_inferred_amplicon_length(self):
        length = len(self.layouts['R1'].seq) + len(self.layouts['R2'].seq) + self.gap
        return length

    @memoized_property
    def bridging_strand(self):
        strand = {}

        for kind in self.bridging_alignments:
            strand[kind] = None
            
            als = self.bridging_alignments[kind]
            if als['R1'] is None or als['R2'] is None:
                continue

            # Note: R2 should be opposite orientation as R1
            flipped_als = [als['R1'], sam.flip_alignment(als['R2'])]
            strands = {sam.get_strand(al) for al in flipped_als}
            if len(strands) > 1:
                continue
            else:
                strand[kind] = strands.pop()

        return strand

    @memoized_property
    def successful_bridging_kind(self):
        successful = set()
        
        for kind in self.bridging_alignments:
            if self.bridging_strand[kind] is not None and self.bridging_als_reach_internal_edges[kind]:
                successful.add(kind)

                
        if len(successful) == 0:
            return None
        elif len(successful) > 1:
            if 'h' in successful:
                return 'h'
            else:
                raise ValueError(self.name, successful)
        else:
            return successful.pop()
    
    @memoized_property
    def gap(self):
        kind = self.successful_bridging_kind
        if kind is None:
            return 100
        
        als = self.bridging_alignments[kind]
        unaligned_gap = sum(self.bridging_als_missing_from_end[kind].values())
        if self.bridging_strand[kind] == '+':
            # If there is no gap, R1 reference_end (which points one past actual end)
            # will be the same as R2 reference_start.
            aligned_gap = als['R2'].reference_start - als['R1'].reference_end
        elif self.bridging_strand[kind] == '-':
            aligned_gap = als['R1'].reference_start - als['R2'].reference_end

        return aligned_gap - unaligned_gap

    @memoized_property
    def uncategorized_relevant_alignments(self):
        als = {which: l.uncategorized_relevant_alignments for which, l in self.layouts.items()}

        return als

    def categorize(self):
        kind = self.successful_bridging_kind
        if kind == 'h' and self.possible_inferred_amplicon_length > 0:
            self.inferred_amplicon_length = self.possible_inferred_amplicon_length

            self.relevant_alignments = {
                'R1': self.layouts['R1'].parsimonious_target_alignments + self.layouts['R1'].parsimonious_donor_alignments,
                'R2': self.layouts['R2'].parsimonious_target_alignments + self.layouts['R2'].parsimonious_donor_alignments,
            }

            junctions = set(self.junctions.values())

            if 'blunt' in junctions and 'uncategorized' not in junctions:
                self.category = 'blunt misintegration'
                self.subcategory = f'5\' {self.junctions[5]}, 3\' {self.junctions[3]}'

            elif junctions == set(['imperfect', 'HDR']):
                self.category = 'incomplete HDR'
                self.subcategory = f'5\' {self.junctions[5]}, 3\' {self.junctions[3]}'

            elif junctions == set(['imperfect']):
                self.category = 'complex misintegration'
                self.subcategory = 'complex misintegration'

            else:
                self.inferred_amplicon_length = -1
                self.category = 'bad sequence'
                self.subcategory = 'non-overlapping'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif kind == 'nh' and self.possible_inferred_amplicon_length > 0:
            self.inferred_amplicon_length = self.possible_inferred_amplicon_length

            self.category = 'non-homologous donor'
            self.subcategory = 'simple'
            self.relevant_alignments = {
                'R1': self.layouts['R1'].parsimonious_target_alignments + self.layouts['R1'].nonhomologous_donor_alignments,
                'R2': self.layouts['R2'].parsimonious_target_alignments + self.layouts['R2'].nonhomologous_donor_alignments,
            }
            
        elif kind == 'nonspecific_amplification' and self.possible_inferred_amplicon_length > 0:
            R1_primer = self.layouts['R1'].target_flanking_alignments[5]
            R2_primer = self.layouts['R2'].target_flanking_alignments[3]

            if R1_primer is not None and R2_primer is not None:
                self.inferred_amplicon_length = self.possible_inferred_amplicon_length

                self.register_nonspecific_amplification()

                bridging_als = self.bridging_alignments['nonspecific_amplification']
                self.relevant_alignments = {
                    'R1': [R1_primer, bridging_als['R1']],
                    'R2': [R2_primer, bridging_als['R2']],
                }

            else:
                self.inferred_amplicon_length = -1
                self.category = 'bad sequence'
                self.subcategory = 'non-overlapping'
                self.relevant_alignments = self.uncategorized_relevant_alignments
        
        elif kind == 'genomic_insertion' and self.possible_inferred_amplicon_length > 0:
            R1_primer = self.layouts['R1'].target_flanking_alignments[5]
            R2_primer = self.layouts['R2'].target_flanking_alignments[3]

            if R1_primer is not None and R2_primer is not None:
                self.inferred_amplicon_length = self.possible_inferred_amplicon_length

                self.register_genomic_insertion()

                bridging_als = self.bridging_alignments['genomic_insertion']
                self.relevant_alignments = {
                    'R1': [R1_primer, bridging_als['R1']],
                    'R2': [R2_primer, bridging_als['R2']],
                }
            else:
                self.inferred_amplicon_length = -1
                self.category = 'bad sequence'
                self.subcategory = 'non-overlapping'
                self.relevant_alignments = self.uncategorized_relevant_alignments
            
        else:
            self.inferred_amplicon_length = -1
            self.category = 'bad sequence'
            self.subcategory = 'non-overlapping'
            self.relevant_alignments = self.uncategorized_relevant_alignments
        
        #if self.strand == '-':
        #    self.relevant_alignments = {
        #        'R1': self.relevant_alignments['R2'],
        #        'R2': self.relevant_alignments['R1'],
        #    }
            
        return self.category, self.subcategory, self.details
    