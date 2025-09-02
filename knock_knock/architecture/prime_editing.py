from collections import Counter, defaultdict

import numpy as np
import pysam

from hits import interval, sam, sw, utilities
from hits.utilities import memoized_property
import hits.visualize

import knock_knock.architecture
import knock_knock.pegRNAs
import knock_knock.visualize.architecture

from knock_knock.outcome import Details, DuplicationJunction

class Architecture(knock_knock.architecture.Categorizer):
    category_order = [
        (
            'wild type',
            (
                'clean',
                'short indel far from cut',
                'mismatches',
            ),
        ),
        (
            'intended edit',
            (
                'substitution',
                'deletion',
                'deletion + substitution',
                'insertion',
                'insertion + substitution',
                'combination',
            ),
        ),
        (
            'partial edit',
            (
                'partial incorporation',
                'other',
            ),
        ),
        (
            'unintended rejoining of RT\'ed sequence',
            (
                'includes scaffold',
                'includes scaffold, no substitution',
                'includes scaffold, with deletion',
                'includes scaffold, no substitution, with deletion',
                'no scaffold',
                'no scaffold, no substitution',
                'no scaffold, with deletion',
                'no scaffold, no substitution, with deletion',
                'doesn\'t include insertion',
            ),
        ),
        (
            'deletion',
            (
                'clean',
                'mismatches',
            ),
        ),
        (
            'duplication',
            (
                'simple',
                'iterated',
                'complex',
            ),
        ),
        (
            'insertion',
            (
                'clean',
                'mismatches',
            ),
        ),
        (
            'edit + indel',
            (
                'deletion',
                'insertion',
                'duplication',
            ),
        ),
        (
            'multiple indels',
            (
                'multiple indels',
                'duplication + deletion',
                'duplication + insertion',
            ),
        ),
        (
            'genomic insertion',
            (
                'hg19',
                'hg38',
                'macFas5',
                'mm10',
                'bosTau7',
                'e_coli',
                'phiX',
            ),
        ),
        (
            'inversion',
            (
                'inversion',
            ),
        ),
        (
            'incorporation of extra sequence',
            (
                'has RT\'ed extension',
                'no RT\'ed extension',
            ),
        ),
        (
            'complex incorporation of RT\'ed sequence',
            (
                'n/a',
            ),
        ),
        (
            'minimal alignment to intended target',
            (
                'n/a',
            ),
        ),
        (
            'uncategorized',
            (
                'uncategorized',
                'low quality',
            ),
        ),
        (
            'nonspecific amplification',
            (
                'hg19',
                'hg38',
                'macFas5',
                'T2T-MFA8v1.0',
                'mm10',
                'bosTau7',
                'e_coli',
                'b_subtilis',
                'primer dimer',
                'short unknown',
                'extra sequence',
            ),
        ),
        (
            'phiX',
            (
                'phiX',
            ),
        ),
    ]

    non_relevant_categories = [
        'phiX',
        'nonspecific amplification',
        'minimal alignment to intended target',
    ]

    def __init__(self, alignments, editing_strategy, error_corrected=False, mode='illumina', flipped=False):
        super().__init__(alignments, editing_strategy)

        self.ins_size_to_split_at = 1
        self.del_size_to_split_at = 1

        self.error_corrected = error_corrected
        self.mode = mode
        self._flipped = flipped

    @property
    def flipped(self):
        return self._flipped

    @memoized_property
    def intended_edit_type(self):
        if len(self.editing_strategy.pegRNA_names) != 1:
            edit_type = None
        else:
            edit_type = self.editing_strategy.pegRNAs[0].edit_type

        return edit_type

    @property
    def sequencing_direction(self):
        sequencing_direction = self.editing_strategy.sequencing_direction

        if self.flipped:
            sequencing_direction = sam.opposite_strand[sequencing_direction]

        return sequencing_direction

    @memoized_property
    def pegRNA_side(self):
        pegRNA_side = self.editing_strategy.pegRNA_side

        if self.flipped:
            pegRNA_side = {'left': 'right', 'right': 'left'}[pegRNA_side]

        return pegRNA_side

    @memoized_property
    def non_pegRNA_side(self):
        non_pegRNA_side = self.editing_strategy.non_pegRNA_side

        if self.flipped:
            non_pegRNA_side = {'left': 'right', 'right': 'left'}[non_pegRNA_side]

        return non_pegRNA_side

    @memoized_property
    def pegRNA_name_to_side_of_read(self):
        if self.flipped:
            pegRNA_name_to_side_of_read = {
                name:  {'left': 'right', 'right': 'left'}[side]
                for name, side in self.editing_strategy.pegRNA_name_to_side_of_read.items()
            }
        else:
            pegRNA_name_to_side_of_read = self.editing_strategy.pegRNA_name_to_side_of_read

        return pegRNA_name_to_side_of_read

    @memoized_property
    def pegRNA_names_by_side_of_read(self):
        return utilities.reverse_dictionary(self.pegRNA_name_to_side_of_read)

    @memoized_property
    def primers_by_side_of_read(self):
        if self.flipped:
            side_to_side = {
                'left': 'right',
                'right': 'left',
            }
        else:
            side_to_side = {
                'left': 'left',
                'right': 'right',
            }

        primers = {
            side: self.editing_strategy.primers_by_side_of_read[side_to_side[side]]
            for side in ['left', 'right']
        }

        return primers

    # Accessing, refining, and augmenting alignments.
    # Care needs to be taken to avoid circular dependencies.
    # In general, methods should be added below any of their dependencies.

    @memoized_property
    def initial_target_alignments(self):
        t_als = [
            al for al in self.alignments
            if al.reference_name == self.editing_strategy.target
        ]
        
        return t_als

    @memoized_property
    def initial_pegRNA_alignments(self):
        if self.editing_strategy.pegRNA_names is None:
            als = []
        else:
            als = [
                al for al in self.alignments
                if al.reference_name in self.editing_strategy.pegRNA_names
            ]
        
        return als

    @memoized_property
    def donor_alignments(self):
        ''' Donor meaning integrase donor '''
        if self.editing_strategy.donor is not None:
            valid_names = [self.editing_strategy.donor]
        else:
            valid_names = []

        donor_als = [
            al for al in self.alignments
            if al.reference_name in valid_names
        ]

        donor_als = self.split_and_extend_alignments(donor_als)
        
        return donor_als

    @memoized_property
    def extra_alignments(self):
        # Any alignments not to the target, pegRNAs, or donor.

        strat = self.editing_strategy

        non_extra_ref_names = {strat.target}

        if strat.pegRNA_names is not None:
            non_extra_ref_names.update(strat.pegRNA_names)

        if strat.donor is not None:
            non_extra_ref_names.add(strat.donor)

        extra_ref_names = {n for n in strat.reference_sequences if n not in non_extra_ref_names}

        als = [al for al in self.alignments if al.reference_name in extra_ref_names]

        return als
    
    @memoized_property
    def supplemental_alignments(self):
        supp_als = [
            al for al in self.alignments
            if (not al.is_unmapped) and al.reference_name not in self.primary_ref_names
        ]

        if self.mode == 'nanopore':
            # TODO: use property value here?
            ins_size_to_split_at = 10
        else:
            ins_size_to_split_at = 2

        split_als = []
        for supp_al in supp_als:
            split_als.extend(sam.split_at_large_insertions(supp_al, ins_size_to_split_at))
        
        few_mismatches = []
        # Alignments generated with STAR will have MD tags, but alignments
        # generated with blastn will not. TODO: pass ref seq along for non-MD-tag
        # alignments.
        for al in split_als:
            if al.has_tag('MD'):
                if sam.total_edit_distance(al) / al.query_alignment_length < 0.2:
                    few_mismatches.append(al)
            else:
                few_mismatches.append(al)

        # Convert relevant supp als to target als.
        # Any that overlap the amplicon interval should not be considered supplemental_als
        # to prevent the intended amplicon from being considered nonspecific amplification.

        strat = self.editing_strategy

        if strat.reference_name_in_genome_source:
            target_als_outside_amplicon = []
            other_reference_als = []

            target_interval = interval.Interval(0, len(strat.target_sequence) - 1)

            for al in few_mismatches:

                if al.reference_name != strat.reference_name_in_genome_source:
                    other_reference_als.append(al)

                else:
                    conversion_results = strat.convert_genomic_alignment_to_target_coordinates(al)

                    if conversion_results:
                        converted_interval = interval.Interval(conversion_results['start'], conversion_results['end'])
    
                        if not interval.are_overlapping(converted_interval, target_interval):
                            other_reference_als.append(al)

                        else:
                            al_dict = al.to_dict()

                            al_dict['ref_name'] = strat.target
                            
                            converted_al = pysam.AlignedSegment.from_dict(al_dict, strat.header)

                            converted_al.reference_start = conversion_results['start']

                            converted_al.is_reverse = (conversion_results['strand'] == '-')

                            overlaps_amplicon = interval.get_covered_on_ref(converted_al) & strat.amplicon_interval

                            if not overlaps_amplicon:
                                target_als_outside_amplicon.append(converted_al)

            final_als = other_reference_als + target_als_outside_amplicon

        else:
            final_als = few_mismatches

        return final_als

    def split_and_extend_alignments(self, als):
        all_split_als = []

        for al in als:
            split_als = knock_knock.architecture.comprehensively_split_alignment(al,
                                                               self.editing_strategy,
                                                               self.mode,
                                                               self.ins_size_to_split_at,
                                                               self.del_size_to_split_at,
                                                               programmed_substitutions=self.editing_strategy.pegRNA_programmed_alternative_bases,
                                                              )

            seq_bytes = self.editing_strategy.reference_sequence_bytes[al.reference_name]

            extended = [sw.extend_alignment(split_al, seq_bytes) for split_al in split_als]

            all_split_als.extend(extended)
        
        return sam.make_nonredundant(all_split_als)
    
    @memoized_property
    def split_target_alignments(self):
        initial_als = self.initial_target_alignments

        return self.split_and_extend_alignments(initial_als)

    @memoized_property
    def split_pegRNA_alignments(self):
        all_split_als = self.split_and_extend_alignments(self.initial_pegRNA_alignments)

        return sam.make_nonredundant(all_split_als)

    @memoized_property
    def non_protospacer_pegRNA_alignments(self):
        return [al for al in self.pegRNA_alignments if not self.is_pegRNA_protospacer_alignment(al)]

    @memoized_property
    def initial_gap(self):
        # Would like to use self.not_covered_by_primers here, but unfortunately that is a
        # circular dependence. As a crude replacement, just exclude the first 10 nts of the read.
        exclude_start = interval.Interval(10, self.read_length)
        return exclude_start & self.not_covered_by_split_target_or_pegRNA_alignments & self.not_covered_by_extra_alignments

    @memoized_property
    def partial_gap_perfect_alignments(self):
        def is_relevant(al):
            if al.reference_name == self.editing_strategy.target:
                return (interval.get_covered_on_ref(al) & self.editing_strategy.amplicon_interval).total_length > 0
            else:
                return True

        als = []

        targets = ['target'] + self.editing_strategy.pegRNA_names

        for target_name in targets:
            for gap in self.initial_gap:
                # Note: interval end is the last base, but seed_and_extend wants one past
                start = gap.start
                end = gap.end + 1

                from_start_gap_als = []
                while (end > start) and not from_start_gap_als:
                    end -= 1
                    from_start_gap_als = self.seed_and_extend(target_name, start, end)
                    from_start_gap_als = [al for al in from_start_gap_als if is_relevant(al)]
                    
                start = gap.start
                end = gap.end + 1
                from_end_gap_als = []
                while (end > start) and not from_end_gap_als:
                    start += 1
                    from_end_gap_als = self.seed_and_extend(target_name, start, end)
                    from_end_gap_als = [al for al in from_end_gap_als if is_relevant(al)]

                als = from_start_gap_als + from_end_gap_als

                for al in als:
                    if al.is_reverse:
                        al.query_qualities = self.qual[::-1]
                    else:
                        al.query_qualities = self.qual

        als = [al for al in als if al.query_alignment_length >= 5]

        return als

    @memoized_property
    def gap_covering_alignments(self):
        strat = self.editing_strategy

        gap_covers = []
        
        target_interval = strat.amplicon_interval
        
        for gap in self.initial_gap:
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

            if strat.pegRNA_names is not None:
                for pegRNA_name in strat.pegRNA_names:
                    als = sw.align_read(self.read,
                                        [(pegRNA_name, strat.reference_sequences[pegRNA_name]),
                                        ],
                                        4,
                                        strat.header,
                                        N_matches=False,
                                        max_alignments_per_target=5,
                                        read_interval=extended_gap,
                                        mismatch_penalty=-2,
                                       )

                    als = [sw.extend_alignment(al, strat.reference_sequence_bytes[pegRNA_name]) for al in als]
                    
                    gap_covers.extend(als)

        return gap_covers

    @memoized_property
    def target_alignments(self):
        strat = self.editing_strategy

        gap_als = [al for al in self.gap_covering_alignments + self.partial_gap_perfect_alignments if al.reference_name == strat.target]
        all_als = self.split_target_alignments + gap_als

        for side in [5, 3]:
            primer_al = self.realign_edges_to_primers(side)
            if primer_al is not None:
                all_als.append(primer_al)

        if (self.whole_read_minus_edges(2) - interval.get_disjoint_covered(all_als)).total_length > 0:
            if self.perfect_right_edge_alignment is not None:
                all_als.append(self.perfect_right_edge_alignment)

        all_als = sam.make_nonredundant(all_als)
        all_als = interval.make_parsimonious(all_als)
        merged_als = sam.merge_any_adjacent_pairs(all_als, strat.reference_sequences)
        split_als = self.split_and_extend_alignments(merged_als)

        return split_als

    @memoized_property
    def pegRNA_alignments(self):
        strat = self.editing_strategy

        if strat.pegRNA_names is not None:
            gap_als = [al for al in self.gap_covering_alignments + self.partial_gap_perfect_alignments if al.reference_name in strat.pegRNA_names]
            all_als = self.split_pegRNA_alignments + gap_als

            all_als = sam.make_nonredundant(all_als)
            all_als = sam.make_noncontained(all_als)

            # Supplement with manually-generated extensions of target edge alignments.
            for side in ['left', 'right']:
                if self.pegRNA_names_by_side_of_read.get(side) is not None and self.target_edge_alignments.get(side) is not None:
                    al = self.generate_extended_pegRNA_PBS_alignment(self.target_edge_alignments[side], side)
                    if al is not None:
                        all_als.append(al)

            merged_als = sam.merge_any_adjacent_pairs(all_als, strat.reference_sequences)
            split_als = self.split_and_extend_alignments(merged_als)
        else:
            split_als = []

        return split_als

    @memoized_property
    def pegRNA_alignments_by_pegRNA_name(self):
        if self.editing_strategy.pegRNA_names is None:
            pegRNA_alignments = None
        else:
            pegRNA_alignments = {
                pegRNA_name: [
                    al for al in self.pegRNA_alignments
                    if al.reference_name == pegRNA_name
                ]
                for pegRNA_name in self.editing_strategy.pegRNA_names
            }
        
        return pegRNA_alignments
    
    @memoized_property
    def target_edge_alignments(self):
        edge_alignments = {
            'left': [],
            'right':[],
        }

        # Re-merge any deletions.
        merged_als = sam.merge_any_adjacent_pairs(self.target_alignments,
                                                  self.editing_strategy.reference_sequences,
                                                  max_deletion_length=2,
                                                 )

        for al in merged_als:
            if sam.get_strand(al) != self.sequencing_direction:
                continue

            covered = interval.get_covered(al)

            if covered.total_length >= 10:
                if covered.start <= 5 or (self.overlaps_primer(al, 'left') and covered.start < 25):
                    edge_alignments['left'].append(al)
                
                if covered.end >= len(self.seq) - 1 - 5 or self.overlaps_primer(al, 'right'):
                    edge_alignments['right'].append(al)

        for edge in ['left', 'right']:
            if len(edge_alignments[edge]) == 0:
                edge_alignments[edge] = None
            else:
                edge_alignments[edge] = max(edge_alignments[edge], key=lambda al: al.query_alignment_length)

        return edge_alignments

    @memoized_property
    def target_edge_alignments_list(self):
        return [al for al in self.target_edge_alignments.values() if al is not None]

    def overlaps_primer(self, al, side):
        primer = self.primers_by_side_of_read[side]

        overlaps = sam.overlaps_feature(al, primer, require_same_strand=False)
        correct_strand = sam.get_strand(al) == self.sequencing_direction 

        return overlaps and correct_strand

    def pegRNA_alignment_extends_target_alignment(self, pegRNA_al, target_al, shared_feature):
        if pegRNA_al is None or target_al is None:
            return None, None, None

        strat = self.editing_strategy

        pegRNA_name = pegRNA_al.reference_name
        target_PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
        HA_RT_name = f'HA_RT_{pegRNA_name}'

        pegRNA_side = self.pegRNA_name_to_side_of_read[pegRNA_name]

        if shared_feature == 'PBS':
            if pegRNA_side == 'left':
                left_al = target_al
                right_al = pegRNA_al

                left_feature_name = target_PBS_name
                right_feature_name = 'PBS'

                pegRNA_al_key = 'right'
                target_al_key = 'left'

            else:
                right_al = target_al
                left_al = pegRNA_al

                right_feature_name = target_PBS_name
                left_feature_name = 'PBS'

                pegRNA_al_key = 'left'
                target_al_key = 'right'

        elif shared_feature == 'RTT':
            left_feature_name = HA_RT_name
            right_feature_name = HA_RT_name

            if pegRNA_side == 'left':
                left_al = pegRNA_al
                right_al = target_al

                pegRNA_al_key = 'left'
                target_al_key = 'right'

            else:
                right_al = pegRNA_al
                left_al = target_al

                pegRNA_al_key = 'right'
                target_al_key = 'left'

        else:
            raise ValueError(shared_feature)

        def contribution_test(al):
            return al.reference_name in strat.pegRNA_names and self.specific_to_pegRNA(al)

        extension_results = self.are_mutually_extending_from_shared_feature(left_al, left_feature_name, right_al, right_feature_name, contribution_test=contribution_test)

        cropped_pegRNA_extension_al = extension_results['cropped_alignments'][pegRNA_al_key]
        cropped_target_al = extension_results['cropped_alignments'][target_al_key]

        # For non-definite PBS-extending als, insist that the pegRNA alignment is not entirely
        # contained in the PBS - i.e. overlaps the RTT.
        if shared_feature == 'PBS' and not sam.overlaps_feature(cropped_pegRNA_extension_al, strat.features[pegRNA_name, 'RTT'], require_same_strand=False):
            extension_results['status'] = 'spurious'

        return extension_results['status'], cropped_pegRNA_extension_al, cropped_target_al

    def find_target_alignment_extending_pegRNA_alignment(self, pegRNA_al, shared_feature, require_definite=True):
        if pegRNA_al is None:
            return None, None, None

        target_als = self.target_alignments + self.target_edge_alignments_list
        manually_extended_al = self.generate_extended_target_PBS_alignment(pegRNA_al)
        if manually_extended_al is not None:
            target_als = target_als + [manually_extended_al]

        by_status = defaultdict(list)

        for target_al in sorted(target_als, key=lambda al: al.query_alignment_length, reverse=True):
            status, cropped_pegRNA_al, cropped_target_al = self.pegRNA_alignment_extends_target_alignment(pegRNA_al, target_al, shared_feature=shared_feature)
            by_status[status].append((target_al, cropped_pegRNA_al, cropped_target_al))

        eligible = by_status['definite']
        if not require_definite:
            eligible.extend(by_status['possible'])

        if len(eligible) > 0:
            relevant_target_al, cropped_pegRNA_al, cropped_target_al = max(eligible, key=lambda tuple_: tuple_[0].query_alignment_length) 
        else:
            relevant_target_al, cropped_pegRNA_al, cropped_target_al = None, None, None
        
        return relevant_target_al, cropped_pegRNA_al, cropped_target_al

    def find_pegRNA_alignment_extending_target_edge_al(self, side, shared_feature, require_definite=True):
        target_edge_al = self.target_edge_alignments[side]
        if target_edge_al is None:
            return None, None, None

        strat = self.editing_strategy

        if len(strat.pegRNA_names) == 0:
            return None, None, None
        elif len(strat.pegRNA_names) == 1:
            pegRNA_name = strat.pegRNA_names[0]
        elif len(strat.pegRNA_names) == 2:
            pegRNA_name = self.pegRNA_names_by_side_of_read[side]
        else:
            raise ValueError(strat.pegRNA_names)

        candidate_als = self.pegRNA_alignments_by_pegRNA_name[pegRNA_name]

        by_status = defaultdict(list)
        
        for pegRNA_al in sorted(candidate_als, key=lambda al: al.query_alignment_length, reverse=True):
            status, cropped_pegRNA_al, cropped_target_al = self.pegRNA_alignment_extends_target_alignment(pegRNA_al, target_edge_al, shared_feature=shared_feature)
            by_status[status].append((pegRNA_al, cropped_pegRNA_al, cropped_target_al))

            if status == 'definite':
                by_status['possible'].append((pegRNA_al, cropped_pegRNA_al, cropped_target_al))
                
        relevant_pegRNA_al, cropped_pegRNA_al, cropped_target_al = None, None, None

        if require_definite:
            eligible = by_status['definite']
        else:
            eligible = by_status['possible']

        if len(eligible) > 0:
            relevant_pegRNA_al, cropped_pegRNA_al, cropped_target_al = max(eligible, key=lambda tuple_: tuple_[0].query_alignment_length)
        else:
            relevant_pegRNA_al, cropped_pegRNA_al, cropped_target_al = None, None, None
        
        return relevant_pegRNA_al, cropped_pegRNA_al, cropped_target_al

    def characterize_extension_chain_on_side(self, side, require_definite=True):
        als = {}

        target_edge_al = self.target_edge_alignments[side]
        
        if target_edge_al is not None:
            als['first target'] = target_edge_al
        
            if side == self.pegRNA_side:
                shared_feature = 'PBS'
            else:
                shared_feature = 'RTT'

            pegRNA_al, _, cropped_target_al = self.find_pegRNA_alignment_extending_target_edge_al(side, shared_feature, require_definite=require_definite)

            if pegRNA_al is not None:
                als['first target'] = cropped_target_al
                als['pegRNA'] = pegRNA_al
                
                if side == self.pegRNA_side:
                    shared_feature = 'RTT'
                else:
                    shared_feature = 'PBS'

                extended_target_al, _, _ = self.find_target_alignment_extending_pegRNA_alignment(pegRNA_al, shared_feature, require_definite=require_definite)
                    
                if extended_target_al is not None:
                    als['second target'] = extended_target_al
                        
        al_order = [
            'first target',
            'pegRNA',
            'second target',
        ]

        query_covered = interval.get_disjoint_covered([])
        query_covered_incremental = {'none': query_covered}

        for al_order_i in range(len(al_order)):
            al_key = al_order[al_order_i]
            if al_key in als:
                als_up_to = [als[key] for key in al_order[:al_order_i + 1]]
                query_covered = interval.get_disjoint_covered(als_up_to)
                query_covered_incremental[al_key] = query_covered

        results = {
            'query_covered': query_covered,
            'query_covered_incremental': query_covered_incremental,
            'alignments': als,
        }

        return results

    @memoized_property
    def extension_chains_by_side(self):
        chains = {side: self.characterize_extension_chain_on_side(side) for side in ['left', 'right']}

        # Check whether any members of an extension chain on one side are not
        # necessary to make it to the other chain. (Warning: could imagine a
        # scenario in which it would be possible to remove from either the
        # left or right chain.)

        al_order = [
            'none',
            'first target',
            'pegRNA',
            'second target',
        ]

        last_al_to_description = {
            'none': 'no target',
            'first target': 'not RT\'ed',
            'pegRNA': 'RT\'ed',
            'second target': 'RT\'ed + annealing-extended',
        }

        possible_covers = set()

        if chains['left']['query_covered'] != chains['right']['query_covered']:
            for left_key in al_order:
                if left_key in chains['left']['alignments']:
                    for right_key in al_order:
                        if right_key in chains['right']['alignments']:
                            covered_left = chains['left']['query_covered_incremental'][left_key]
                            covered_right = chains['right']['query_covered_incremental'][right_key]

                            # Check if left and right overlap or abut each other.
                            if covered_left.end >= covered_right.start - 1:
                                possible_covers.add((left_key, right_key))

        last_parsimonious_key = {}

        if possible_covers:
            last_parsimonious_key['left'], last_parsimonious_key['right'] = max(possible_covers, key=lambda pair: (al_order.index(pair[0]), al_order.index(pair[1])))
        else:
            for side, chain in chains.items():
                maximal_covers = [k for k, covered in chain['query_covered_incremental'].items() if covered == chain['query_covered']]
                last_parsimonious_key[side] = min(maximal_covers, key=al_order.index, default='none')

        for side in ['left', 'right']:
            key = last_parsimonious_key[side]

            chains[side]['description'] = last_al_to_description[key]

            last_index = al_order.index(key)
            chains[side]['parsimonious_alignments'] = [al for key, al in chains[side]['alignments'].items() if al_order.index(key) <= last_index]

            chains[side]['query_covered'] = chains[side]['query_covered_incremental'][key]

        # If one chain is absent and the other chain covers the whole read
        # (except possibly 2 nts at either edge), classify the missing side
        # as 'not seen'.

        not_covered_by_primers_minus_edges = self.not_covered_by_primers & self.whole_read_minus_edges(2)

        if not_covered_by_primers_minus_edges in chains['left']['query_covered']:
            if chains['right']['description'] == 'no target':
                chains['right']['description'] = 'not seen'

        if not_covered_by_primers_minus_edges in chains['right']['query_covered']:
            if chains['left']['description'] == 'no target':
                chains['left']['description'] = 'not seen'

        return chains

    @memoized_property
    def extension_chain(self):
        return self.extension_chains_by_side[self.pegRNA_side]

    @memoized_property
    def possible_extension_chains_by_side(self):
        chains = {side: self.characterize_extension_chain_on_side(side, require_definite=False) for side in ['left', 'right']}

        al_order = [
            'none',
            'first target',
            'pegRNA',
            'second target',
        ]

        last_al_to_description = {
            'none': 'no target',
            'first target': 'not RT\'ed',
            'pegRNA': 'RT\'ed',
            'second target': 'RT\'ed + annealing-extended',
        }

        last_parsimonious_key = {}

        for side in ['left', 'right']:
            last_parsimonious_key[side] = max(chains[side]['alignments'], key=al_order.index, default='none')

        for side in ['left', 'right']:
            key = last_parsimonious_key[side]

            chains[side]['description'] = last_al_to_description[key]

            last_index = al_order.index(key)
            chains[side]['parsimonious_alignments'] = [al for key, al in chains[side]['alignments'].items() if al_order.index(key) <= last_index]

            chains[side]['query_covered'] = chains[side]['query_covered_incremental'][key]

        # If one chain is absent and the other chain covers the whole read
        # (except possibly 2 nts at either edge), classify the missing side
        # as 'not seen'.

        not_covered_by_primers_minus_edges = self.not_covered_by_primers & self.whole_read_minus_edges(2)

        if not_covered_by_primers_minus_edges in chains['left']['query_covered']:
            if chains['right']['description'] == 'no target':
                chains['right']['description'] = 'not seen'

        if not_covered_by_primers_minus_edges in chains['right']['query_covered']:
            if chains['left']['description'] == 'no target':
                chains['left']['description'] = 'not seen'

        return chains

    @memoized_property
    def possible_extension_chain(self):
        return self.possible_extension_chains_by_side[self.pegRNA_side]

    def get_extension_chain_junction_microhomology(self, require_definite=True):
        last_als = {}

        for side in ['left', 'right']:
            if require_definite:
                chain = self.extension_chains_by_side[side]
            else:
                chain = self.possible_extension_chains_by_side[side]

            if chain['description'] in ['not seen', 'no target']:
                last_al = None

            else:
                if side == self.pegRNA_side:
                    if chain['description'] == 'RT\'ed + annealing-extended':
                        last_al = chain['alignments']['second target']
                    elif chain['description'] == 'not RT\'ed':
                        last_al = chain['alignments']['first target']
                    elif chain['description'] == 'RT\'ed':
                        last_al = chain['alignments']['pegRNA']
                else:
                    # For non-pegRNA side, only consider target.
                    last_al = chain['alignments']['first target']

            last_als[side] = last_al

        if None in last_als.values():
            mh = None
        
        else:
            mh = knock_knock.architecture.junction_microhomology(self.editing_strategy.reference_sequences, last_als['left'], last_als['right'])

            mh = max(mh, 0)

        return mh

    @memoized_property
    def extension_chain_junction_microhomology(self):
        return self.get_extension_chain_junction_microhomology()

    @memoized_property
    def possible_extension_chain_junction_microhomology(self):
        return self.get_extension_chain_junction_microhomology(require_definite=False)

    def get_extension_chain_edge(self, side, require_definite=True):
        ''' Get the position of the far edge of an extension chain
        in the relevant coordinate system.
        '''
        strat = self.editing_strategy

        pegRNA_name = strat.pegRNA_names[0]

        PBS_end = strat.features[pegRNA_name, 'PBS'].end

        target_PBS_name = strat.PBS_names_by_side_of_read[self.pegRNA_side]
        target_PBS = strat.features[strat.target, target_PBS_name]

        if require_definite:
            chain = self.extension_chains_by_side[side]
        else:
            chain = self.possible_extension_chains_by_side[side]

        if chain['description'] in ['not seen', 'no target']:
            relevant_edge = None

        elif side == self.pegRNA_side:
            if chain['description'] == 'RT\'ed':
                al = chain['alignments']['pegRNA']

                relevant_edge = PBS_end - al.reference_start

            else:
                if chain['description'] == 'RT\'ed + annealing-extended':
                    al = chain['alignments']['second target']
                else:
                    al = chain['alignments']['first target']

                if target_PBS.strand == '+':
                    relevant_edge = (al.reference_end - 1) - target_PBS.end
                else:
                    # TODO: confirm that there are no off-by-one errors here.
                    relevant_edge = target_PBS.start - al.reference_start

        else:
            al = chain['alignments']['first target']

            # By definition, the nt on the PAM-distal side of the nick
            # is zero in the coordinate system, and postive values go towards
            # the PAM.

            # 24.12.05: having trouble reconciling comment above with code. 
            # Should it be "PAM-proximal side of the nick"?

            if target_PBS.strand == '+':
                relevant_edge = al.reference_start - (target_PBS.end + 1)
            else:
                # TODO: confirm that there are no off-by-one errors here.
                relevant_edge = (target_PBS.start - 1) - (al.reference_end - 1)

        return relevant_edge

    def convert_target_alignment_edge_to_nick_coordinate(self, al, start_or_end):
        strat = self.editing_strategy
        target_PBS = strat.features[strat.target, strat.pegRNA.PBS_name]

        if start_or_end == 'start':
            reference_edge = al.reference_start
        elif start_or_end == 'end':
            reference_edge = al.reference_end - 1
        
        if target_PBS.strand == '+':
            converted_edge = reference_edge - target_PBS.end
        else:
            converted_edge = target_PBS.start - reference_edge

        return converted_edge

    @memoized_property
    def extension_chain_edges(self):
        return {side: self.get_extension_chain_edge(side) for side in ['left', 'right']}

    @memoized_property
    def possible_extension_chain_edges(self):
        return {side: self.get_extension_chain_edge(side, require_definite=False) for side in ['left', 'right']}

    @memoized_property
    def pegRNA_extension_als_list(self):
        extension_als = []

        if len(self.editing_strategy.pegRNA_names) == 1:
            if 'pegRNA' in self.extension_chain['alignments']:
                extension_als.append(self.extension_chain['alignments']['pegRNA'])

        elif len(self.editing_strategy.pegRNA_names) == 2:
            for side, extension_chain in self.extension_chains_by_side.items():
                if 'first pegRNA' in extension_chain['alignments']:
                    extension_als.append(extension_chain['alignments']['first pegRNA'])
                if 'second pegRNA' in extension_chain['alignments']:
                    extension_als.append(extension_chain['alignments']['second pegRNA'])

        extension_als = sam.make_nonredundant(extension_als)

        return extension_als

    @memoized_property
    def pegRNA_extension_als_from_either_side_list(self):
        ''' For when an extension al might not get called on both sides
        because of e.g. an indel disrupting the target edge alignment
        it would need to extend.
        '''
        pegRNA_als = []
        for side in ['left', 'right']:
            for key in ['pegRNA', 'first pegRNA', 'second pegRNA']:
                if (pegRNA_al := self.extension_chains_by_side[side]['alignments'].get(key)) is not None:
                    pegRNA_als.append(pegRNA_al)

        return pegRNA_als

    @memoized_property
    def is_intended_deletion(self):
        is_intended_deletion = False
        target_alignment = None

        def is_intended(indel):
            return indel.kind == 'D' and indel == self.editing_strategy.pegRNA_programmed_deletion

        if self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
        elif self.original_target_covering_alignment:
            target_alignment = self.original_target_covering_alignment

        if target_alignment is not None:
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            # "Uninteresting indels" are 1-nt deletions that don't overlap a window of 5 nts on either side of a cut site.
            # Need to check these in case this is true of the intended deletion.

            intended_deletions = [indel for indel in interesting_indels + uninteresting_indels if is_intended(indel)]
            interesting_not_intended_deletions = [indel for indel in interesting_indels if not is_intended(indel)]

            if len(intended_deletions) == 1 and len(interesting_not_intended_deletions) == 0:
                is_intended_deletion = True

        return is_intended_deletion

    @memoized_property
    def is_intended_edit(self):
        if self.intended_edit_type is None:
            return False

        elif self.intended_edit_type == 'deletion':
            # Outcomes that are very close to but not exactly an intended deletion
            # can produce full extension chains. 
            return self.is_intended_deletion

        else:
            full_chain = set(self.extension_chain['alignments']) == {'first target', 'pegRNA', 'second target'}

            covered = self.extension_chain['query_covered']
            # Allow failure to explain the last few nts of the read.
            need_to_cover = self.whole_read_minus_edges(2) & (self.covered_by_primers | self.not_covered_by_primers)
            uncovered = need_to_cover - covered

            chain_covers_whole_read = full_chain and uncovered.total_length == 0

            not_wild_type = not (self.intended_edit_type == 'substitution(s)' and self.pegRNA_substitution_string == self.no_incorporation_pegRNA_substitution_string)

            return chain_covers_whole_read and not_wild_type and self.matches_all_programmed_insertion_features

    @memoized_property
    def flipped_pegRNA_als(self):
        ''' Identify flipped pegRNA alignments that pair the pegRNA protospacer with target protospacer. '''

        strat = self.editing_strategy

        flipped_als = {}

        for side, pegRNA_name in self.pegRNA_names_by_side_of_read.items():
            flipped_als[side] = []

            # Note: can't use parsimonious here.
            pegRNA_als = self.pegRNA_alignments_by_pegRNA_name[pegRNA_name]
            target_al = self.target_edge_alignments[side]

            ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)

            scaffold_feature = strat.features[pegRNA_name, 'scaffold']

            for pegRNA_al in pegRNA_als:
                if self.share_feature(target_al, ps_name, pegRNA_al, 'protospacer'):
                    if sam.feature_overlap_length(pegRNA_al, scaffold_feature) >= 10:
                        flipped_als[side].append(pegRNA_al)
                        
        return flipped_als

    @memoized_property
    def single_read_covering_target_alignment(self):
        need_to_cover = self.whole_read_minus_edges(2) & self.not_covered_by_primers

        merged_als = sam.merge_any_adjacent_pairs(self.target_alignments, self.editing_strategy.reference_sequences)

        covering_als = []

        for al in merged_als:
            if (need_to_cover - interval.get_covered(al)).total_length == 0:
                covering_als.append(al)
                
        if len(covering_als) > 0:
            covering_al = covering_als[0]
        else:
            covering_al = None
            
        return covering_al

    @memoized_property
    def original_target_covering_alignment(self):
        ''' Reads that cover the whole amplicon on the target but
        contain many sequencing errors may get split in such
        a way that single_read_covering_target_alignment doesn't end
        up re-assembling them.
        '''
        need_to_cover = self.not_covered_by_primers

        merged_original_als = sam.merge_any_adjacent_pairs(self.primary_alignments, self.editing_strategy.reference_sequences)
        
        original_covering_als = [al for al in merged_original_als
                                 if al.reference_name == self.editing_strategy.target and 
                                 (need_to_cover - interval.get_covered(al)).total_length == 0
                                ]

        if len(original_covering_als) == 1:
            target_covering_alignment = original_covering_als[0]
        else:
            target_covering_alignment = None

        return target_covering_alignment

    def query_missing_from_alignment(self, al):
        if al is None:
            return None
        else:
            split_als = sam.split_at_large_insertions(al, 5)
            covered = interval.get_disjoint_covered(split_als)
            ignoring_edges = interval.Interval(covered.start, covered.end)

            missing_from = {
                'start': covered.start,
                'end': len(self.seq) - covered.end - 1,
                'middle': (ignoring_edges - covered).total_length,
            }

            return missing_from

    def alignment_covers_read(self, al):
        missing_from = self.query_missing_from_alignment(al)

        # Non-indel-containing alignments can more safely be considered to have truly
        # reached an edge if they make it to a primer since the primer-overlapping part
        # of the alignment is less likely to be noise.
        no_indels = len(self.extract_indels_from_alignments([al])) == 0

        if missing_from is None:
            return False
        else:
            not_too_much = {
                'start': (missing_from['start'] <= 5) or (no_indels and self.overlaps_primer(al, 'left')),
                'end': (missing_from['end'] <= 5) or (no_indels and self.overlaps_primer(al, 'right')),
                'middle': (missing_from['middle'] <= 5),
            }

            starts_at_expected_location = self.overlaps_primer(al, 'left')

            return all(not_too_much.values()) and starts_at_expected_location

    @memoized_property
    def starts_at_expected_location(self):
        edge_al = self.target_edge_alignments['left']
        return edge_al is not None and self.overlaps_primer(edge_al, 'left')

    def summarize_mismatches_in_alignments(self, relevant_alignments):
        ''' Record bases seen at programmed substitution positions relative to target +. '''
        substitutions = self.editing_strategy.pegRNA_substitutions

        target = self.editing_strategy.target
        position_to_substitution_name = {}

        if substitutions is not None:
            for ref_name in substitutions:
                for name in substitutions[ref_name]:
                    position_to_substitution_name[ref_name, substitutions[ref_name][name]['position']] = name

        read_bases_at_substitution_locii = {name: [] for name in position_to_substitution_name.values()}

        non_pegRNA_mismatches = []

        for al in relevant_alignments:
            is_pegRNA_al = al.reference_name in self.editing_strategy.pegRNA_names

            ref_seq = self.editing_strategy.reference_sequences[al.reference_name]

            for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, ref_seq):
                if true_read_i is None or ref_i is None:
                    continue

                if (al.reference_name, ref_i) in position_to_substitution_name:
                    substitution_name = position_to_substitution_name[al.reference_name, ref_i]

                    # read_b is relative to al.reference_name + strand.
                    # If target, done.
                    # If pegRNA, flip if necessary
                    if substitutions[al.reference_name][substitution_name]['strand'] == '-':
                        read_b = utilities.reverse_complement(read_b)

                    # For combination edits, a target alignment may spuriously
                    # extend across an substitution, creating a disagreement with a pegRNA
                    # alignment. If this happens, gave precedence to the pegRNA
                    # alignment.

                    read_bases_at_substitution_locii[substitution_name].append((read_b, qual, is_pegRNA_al))
                else:
                    substitution_name = None

                if al.reference_name == target and read_b != ref_b:
                    if substitution_name is None:
                        matches_pegRNA = False
                    else:
                        pegRNA_base = substitutions[target][substitution_name]['alternative_base']

                        matches_pegRNA = (pegRNA_base == read_b)

                    if not matches_pegRNA:
                        mismatch = knock_knock.outcome.Mismatch(ref_i, read_b)
                        non_pegRNA_mismatches.append(mismatch)

        non_pegRNA_mismatches = knock_knock.outcome.Mismatches(non_pegRNA_mismatches)

        return read_bases_at_substitution_locii, non_pegRNA_mismatches

    @memoized_property
    def mismatches_summary(self):
        # Don't want to consider probably spurious alignments to parts of the query that
        # should have been trimmed. 

        relevant_alignments  = [
            al for al in self.target_alignments
            if (interval.get_covered(al) & self.not_covered_by_primers).total_length >= 10
        ]

        relevant_alignments.extend(self.pegRNA_extension_als_list)

        return self.summarize_mismatches_in_alignments(relevant_alignments)

    def specific_to_pegRNA(self, al):
        ''' Does al contain a pegRNA-specific substitution? '''
        if al is None or al.is_unmapped:
            return False

        strat = self.editing_strategy

        ref_name = al.reference_name
        ref_seq = strat.reference_sequences[al.reference_name]

        contains_substitution = False

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, ref_seq):
            # Note: read_b and ref_b are as if the read is the forward strand
            pegRNA_base = strat.simple_pegRNA_substitutions.get((ref_name, ref_i))

            if pegRNA_base is not None and pegRNA_base == read_b:
                contains_substitution = True

        return contains_substitution

    @memoized_property
    def pegRNA_substitution_locii_summary(self):
        substitutions = self.editing_strategy.pegRNA_substitutions

        substitution_names_seen = set()

        if substitutions is None:
            string_summary = ''
        else:
            pegRNA_substitution_locii, _ = self.mismatches_summary

            target = self.editing_strategy.target
            
            genotype = {}

            for substitution_name in sorted(substitutions[target]):
                bs_from_pegRNA = {b for b, q, from_pegRNA in pegRNA_substitution_locii[substitution_name] if from_pegRNA}
                all_bs = {b for b, q, from_pegRNA in pegRNA_substitution_locii[substitution_name]}

                if len(bs_from_pegRNA) > 0:
                    bs = bs_from_pegRNA
                else:
                    bs = all_bs

                if len(bs) == 0:
                    genotype[substitution_name] = '-'
                elif len(bs) != 1:
                    genotype[substitution_name] = 'N'
                else:
                    b = list(bs)[0]

                    if b == substitutions[target][substitution_name]['base']:
                        genotype[substitution_name] = '_'
                    else:
                        genotype[substitution_name] = b

                        pegRNA_base = substitutions[target][substitution_name]['alternative_base']
                    
                        if b == pegRNA_base:
                            substitution_names_seen.add(substitution_name)

            string_summary = ''.join(genotype[substitution_name] for substitution_name in sorted(substitutions[target]))

        has_pegRNA_substitution = len(substitution_names_seen) > 0

        pegRNAs_that_explain_all_substitutions = set()
        for pegRNA_name in self.editing_strategy.pegRNA_names:
            if all(substitution_name in substitutions[pegRNA_name] for substitution_name in substitution_names_seen):
                pegRNAs_that_explain_all_substitutions.add(pegRNA_name)

        return has_pegRNA_substitution, pegRNAs_that_explain_all_substitutions, string_summary

    @memoized_property
    def has_pegRNA_substitution(self):
        has_pegRNA_substitution, _, _ = self.pegRNA_substitution_locii_summary
        return has_pegRNA_substitution

    @memoized_property
    def pegRNAs_that_explain_all_substitutions(self):
        _, pegRNAs_that_explain_all_substitutions, _ = self.pegRNA_substitution_locii_summary
        return pegRNAs_that_explain_all_substitutions

    @memoized_property
    def pegRNA_substitution_string(self):
        _, _, string_summary = self.pegRNA_substitution_locii_summary
        return string_summary

    @memoized_property
    def full_incorporation_pegRNA_substitution_string(self):
        ''' value of self.pegRNA_substitution_string expected if all substitutions are incorporated '''
        strat = self.editing_strategy
        substitutions = strat.pegRNA_substitutions

        full_incorporation = []

        if substitutions is not None:
            for substitution_name in sorted(substitutions[strat.target]):
                pegRNA_base = substitutions[strat.target][substitution_name]['alternative_base']
                full_incorporation.append(pegRNA_base)

        full_incorporation = ''.join(full_incorporation)

        return full_incorporation

    @memoized_property
    def no_incorporation_pegRNA_substitution_string(self):
        ''' value of self.pegRNA_substitution_string expected if no substitutions are incorporated '''
        strat = self.editing_strategy
        substitutions = strat.pegRNA_substitutions

        no_incorporation = []

        if substitutions is not None:
            for substitution_name in sorted(substitutions[strat.target]):
                no_incorporation.append('_')

        no_incorporation = ''.join(no_incorporation)

        return no_incorporation

    @memoized_property
    def pegRNA_insertion_feature_summaries(self):
        summaries = {}
        
        for feature in self.editing_strategy.pegRNA_programmed_insertion_features:
            programmed_insertion_sequence = feature.sequence(self.editing_strategy.reference_sequences)
            observed_sequences = []
            for al in self.pegRNA_extension_als_list:
                cropped_al = sam.crop_al_to_feature(al, feature)
                if cropped_al is not None:
                    # programmed sequence is always on minus strand, and
                    # query_alignment_sequence is always reported as if on plus
                    observed_sequence = utilities.reverse_complement(cropped_al.query_alignment_sequence)

                    observed_sequences.append(observed_sequence)

            def close_enough(observed):
                if len(observed) != len(programmed_insertion_sequence):
                    return False
                else:
                    return sum(a == b for a, b in zip(observed, programmed_insertion_sequence)) >= 0.9 * len(observed)
                    
            matches = (len(observed_sequences) > 0) and all(map(close_enough, observed_sequences))
            
            summaries[feature.ID] = (programmed_insertion_sequence, observed_sequences, matches)
            
        return summaries

    @memoized_property
    def matches_all_programmed_insertion_features(self):
        summaries = self.pegRNA_insertion_feature_summaries
        return len(summaries) == 0 or all(matches for _, _, matches in summaries.values())

    @memoized_property
    def indels(self):
        merged_target_als = sam.merge_any_adjacent_pairs(self.target_alignments, self.editing_strategy.reference_sequences)
        return self.extract_indels_from_alignments(merged_target_als)

    def alignment_scaffold_overlap(self, al):
        strat = self.editing_strategy

        if al.reference_name not in strat.pegRNA_names:
            scaffold_overlap = 0

        else:

            pegRNA_name = al.reference_name
            pegRNA_seq = strat.reference_sequences[pegRNA_name]

            scaffold_feature = strat.features[pegRNA_name, 'scaffold']
            cropped = sam.crop_al_to_ref_int(al, scaffold_feature.start, scaffold_feature.end)
            if cropped is None:
                scaffold_overlap = 0
            else:
                scaffold_overlap = cropped.query_alignment_length

                # Try to filter out junk alignments.
                edits = sam.edit_distance_in_query_interval(cropped, ref_seq=pegRNA_seq)
                if edits / scaffold_overlap > 0.2:
                    scaffold_overlap = 0

                # Insist on overlapping HA_RT to prevent false positive from protospacer alignment.            
                if self.HA_RT is not None and not sam.overlaps_feature(al, self.HA_RT, require_same_strand=False):
                    scaffold_overlap = 0

        return scaffold_overlap

    @memoized_property
    def max_scaffold_overlap(self):
        return max([self.alignment_scaffold_overlap(al) for al in self.initial_pegRNA_alignments], default=0)

    @memoized_property
    def HA_RT(self):
        pegRNA_name = self.editing_strategy.pegRNA_names[0]
        return self.editing_strategy.features.get((pegRNA_name, f'HA_RT_{pegRNA_name}'))

    def interesting_and_uninteresting_indels(self, als):
        indels = self.extract_indels_from_alignments(als)

        interesting = []
        uninteresting = []

        for indel, near_cut in indels:
            if near_cut:
                append_to = interesting
            else:
                if indel.kind == 'D' and indel.length == 1:
                    append_to = uninteresting
                else:
                    append_to = interesting

            append_to.append(indel)

        return interesting, uninteresting

    def extract_indels_from_alignments(self, als):
        strat = self.editing_strategy

        around_cut_interval = strat.around_cuts(5)

        primer_intervals = interval.make_disjoint([interval.Interval.from_feature(p) for p in strat.primers.values()])

        indels = []
        for al in als:
            for i, (cigar_op, length) in enumerate(al.cigar):
                if cigar_op == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before
                    ends_at = starts_at + length - 1

                    indel_interval = interval.Interval(starts_at, ends_at)

                    indel = knock_knock.outcome.DegenerateDeletion([starts_at], length)

                elif cigar_op == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    indel_interval = interval.Interval(starts_after, starts_after)

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    indel = knock_knock.outcome.DegenerateInsertion([starts_after], [insertion])
                    
                else:
                    continue

                near_cut = len(indel_interval & around_cut_interval) > 0
                entirely_in_primer = indel_interval in primer_intervals

                indel = self.editing_strategy.expand_degenerate_indel(indel)
                indels.append((indel, near_cut, entirely_in_primer))

        # Ignore any indels entirely contained in primers.

        indels = [(indel, near_cut) for indel, near_cut, entirely_in_primer in indels if not entirely_in_primer]

        return indels

    @memoized_property
    def indels_string(self):
        reps = [str(indel) for indel in self.indels]
        string = ' '.join(reps)
        return string

    @memoized_property
    def covered_by_split_target_or_pegRNA_alignments(self):
        relevant_als = self.split_target_alignments + self.split_pegRNA_alignments
        covered = interval.get_disjoint_covered(relevant_als)
        return covered

    @memoized_property
    def not_covered_by_split_target_or_pegRNA_alignments(self):
        return self.whole_read_minus_edges(2) - self.covered_by_split_target_or_pegRNA_alignments

    @memoized_property
    def not_covered_by_primary_alignments(self):
        covered = interval.get_disjoint_covered(self.primary_alignments)
        return self.whole_read - covered

    @memoized_property
    def not_covered_by_extra_alignments(self):
        covered = interval.get_disjoint_covered(self.extra_alignments)
        return self.whole_read - covered

    @memoized_property
    def not_covered_by_target_edge_alignments(self):
        als = self.target_edge_alignments_list
        uncovered = (self.whole_read - interval.get_disjoint_covered(als)) & self.not_covered_by_primers
        return uncovered

    @memoized_property
    def not_covered_by_donor_alignments(self):
        als = self.donor_alignments
        uncovered = (self.whole_read - interval.get_disjoint_covered(als)) & self.not_covered_by_primers
        return uncovered

    @memoized_property
    def query_length_covered_by_on_target_alignments(self):
        return (self.not_covered_by_primers & self.covered_by_split_target_or_pegRNA_alignments).total_length

    @memoized_property
    def nonredundant_supplemental_alignments(self):
        nonredundant = []
        
        for al in self.supplemental_alignments:
            # phiX alignments are handled elsewhere
            if 'phiX' in al.reference_name:
                continue

            covered = interval.get_covered(al)

            novel_covered = (
                covered & 
                self.not_covered_by_split_target_or_pegRNA_alignments &
                self.not_covered_by_extra_alignments &
                self.not_covered_by_donor_alignments
            )

            if novel_covered:
                nonredundant.append(al)

        return nonredundant

    def seed_and_extend(self, on, query_start, query_end):
        extender = self.editing_strategy.seed_and_extender[on]
        return extender(self.seq_bytes, query_start, query_end, self.query_name)
    
    @memoized_property
    def perfect_right_edge_alignment(self):
        min_length = 5
        
        edge_al = None

        def is_valid(al):
            long_enough = al.query_alignment_length >= min_length
            overlaps_amplicon_interval = interval.are_overlapping(interval.get_covered_on_ref(al), self.editing_strategy.amplicon_interval)
            return long_enough and overlaps_amplicon_interval

        for length in range(20, 3, -1):
            start = max(0, len(self.seq) - length)
            end = len(self.seq)

            als = self.seed_and_extend('target', start, end)

            valid = [al for al in als if is_valid(al)]
            
        if len(valid) > 0:
            edge_al = max(valid, key=lambda al: al.query_alignment_length)

            if edge_al.is_reverse:
                edge_al.query_qualities = self.qual[::-1]
            else:
                edge_al.query_qualities = self.qual

        return edge_al

    @memoized_property
    def primer_alignments(self):
        primers = self.primers_by_side_of_read

        primer_alignments = {}

        for side in ['left', 'right']:
            al = self.target_edge_alignments[side]
            primer_alignments[side] = sam.crop_al_to_feature(al, primers[side])

        return primer_alignments

    @memoized_property
    def covered_by_primers(self):
        return interval.get_disjoint_covered(self.primer_alignments.values())

    @memoized_property
    def not_covered_by_primers(self):
        ''' More complicated than function name suggests. If there are primer
        alignments, returns the query interval between but not covered by them to enable ignoring
        the region outside of them, which is likely to be the result of incorrect trimming.
        '''
        if self.covered_by_primers.is_empty:
            not_covered_by_primers = self.whole_read

        elif self.primer_alignments['left'] and not self.primer_alignments['right']:
            not_covered_by_primers = interval.Interval(self.covered_by_primers.end + 1, self.whole_read.end)

        elif self.primer_alignments['right'] and not self.primer_alignments['left']:
            not_covered_by_primers = interval.Interval(self.whole_read.start, self.covered_by_primers.start - 1)

        else:
            not_covered_by_primers = interval.Interval(self.covered_by_primers.start, self.covered_by_primers.end) - self.covered_by_primers 

        if not_covered_by_primers.is_empty:
            not_covered_by_primers = interval.Interval.empty()

        return not_covered_by_primers

    @memoized_property
    def non_primer_nts(self):
        return self.not_covered_by_primers.total_length

    @memoized_property
    def target_nts_past_primer(self):
        target_nts_past_primer = {}

        for side in ['left', 'right']:
            target_past_primer = interval.get_covered(self.target_edge_alignments[side]) - interval.get_covered(self.primer_alignments[side])
            target_nts_past_primer[side] = target_past_primer.total_length 

        return target_nts_past_primer

    @memoized_property
    def nonspecific_amplification(self):
        ''' Nonspecific amplification if any of following apply:
         
         - read is empty after adapter trimming
         
         - read is short after adapter trimming, in which case inference of
            nonspecific amplification per se is less clear but
            sequence is unlikely to be informative of any other process
         
         - read starts with an alignment to the expected primer, but
            this alignment does not extend substantially past the primer, and
            the rest of the read is covered by a single alignment to some other
            source that either reaches the end of the read or reaches an
            an alignment to the other primer that does not extend 
            substantially past the primer.
         
         - read starts with an alignment to the expected primer, but all
            alignments to the target collectively leave a substantial part
            of the read uncovered, and a single alignment to some other
            source covers the entire read with minimal edit distance.
         
         - read starts and ends with alignments to the expected primers, these
           alignments are spanned by a single alignment to some other source, and
           the inferred amplicon length is more than 20 nts different from the expected
           WT amplicon. This covers the case where an amplififcation product has enough
           homology around the primer for additional sequence to align to the target. 
        
        '''
        results = {}

        valid = False

        min_relevant_length = self.editing_strategy.min_relevant_length
        if min_relevant_length is None:
            min_relevant_length = self.editing_strategy.combined_primer_length + 10

        need_to_cover = self.whole_read_minus_edges(2) & self.not_covered_by_primers

        results['covering_als'] = []

        if len(self.seq) <= min_relevant_length or self.non_primer_nts <= 10:
            valid = True

        if need_to_cover.total_length > 0 and self.primer_alignments['left'] is not None:

            if self.target_nts_past_primer['left'] <= 10 and self.target_nts_past_primer['right'] <= 10:
                # Exclude phiX reads, which can rarely have spurious alignments to the forward primer
                # close to the start of the read that overlap the forward primer.
                relevant_alignments = self.supplemental_alignments + self.split_pegRNA_alignments + self.extra_alignments
                relevant_alignments = [al for al in relevant_alignments if 'phiX' not in al.reference_name]

                for al in relevant_alignments:
                    covered_by_al = interval.get_covered(al)
                    if (need_to_cover - covered_by_al).total_length == 0:
                        results['covering_als'].append(al)

            else:
                target_als = [al for al in self.primary_alignments if al.reference_name == self.editing_strategy.target]
                not_covered_by_any_target_als = self.not_covered_by_primers - interval.get_disjoint_covered(target_als)

                has_substantial_uncovered = not_covered_by_any_target_als.total_length >= 100
                has_substantial_length_discrepancy = (
                    abs(self.inferred_amplicon_length - self.editing_strategy.amplicon_length) >= 20 and
                    self.primer_alignments['right'] is not None
                )

                if has_substantial_uncovered or has_substantial_length_discrepancy:
                    ref_seqs = {**self.editing_strategy.reference_sequences}

                    # Exclude phiX reads, which can rarely have spurious alignments to the forward primer
                    # close to the start of the read that overlap the forward primer.
                    relevant_alignments = self.supplemental_alignments
                    relevant_alignments = [al for al in relevant_alignments if 'phiX' not in al.reference_name]

                    for al in relevant_alignments:
                        covered_by_al = interval.get_covered(al)
                        if (need_to_cover - covered_by_al).total_length == 0:
                            cropped_al = sam.crop_al_to_query_int(al, self.not_covered_by_primers.start, self.not_covered_by_primers.end)
                            total_edits = sum(knock_knock.architecture.edit_positions(cropped_al, ref_seqs, use_deletion_length=True))
                            if total_edits <= 5:
                                results['covering_als'].append(al)


            if len(results['covering_als']) > 0:
                if not any('phiX' in al.reference_name for al in results['covering_als']):
                    valid = True

        if valid:
            results['target_edge_als'] = self.target_edge_alignments_list
        else:
            results = None

        return results

    @memoized_property
    def extension_chain_covers_both_HAs(self):
        strat = self.editing_strategy

        if strat.pegRNA is None:
            covers = True

        else:
            HA_names = [
                f'HA_PBS_{strat.pegRNA.name}',
                f'HA_RT_{strat.pegRNA.name}',
            ]

            intervals = []

            for HA_name in HA_names:
                HA_feature = strat.features.get((strat.target, HA_name))
                if HA_feature is not None:
                    intervals.append(interval.Interval.from_feature(HA_feature))

            need_to_cover = interval.DisjointIntervals(intervals)

            chain_als = self.extension_chains_by_side['left']['alignments']
            chain_target_als = [chain_als[key] for key in ['first target', 'second target'] if key in chain_als]
            covered = interval.get_disjoint_covered_on_ref(chain_target_als)

            covers = need_to_cover in covered
        
        return covers

    @memoized_property
    def longest_phiX_alignment(self):
        # Note: hard-coding of phiX reference is not ideal here.
        # Should really use doubled phiX
        phiX_alignments = [al for al in self.supplemental_alignments if al.reference_name == 'phiX_phix']

        if len(phiX_alignments) == 0:
            longest_phiX_alignment = None
        else:
            longest_phiX_alignment = max(phiX_alignments, key=lambda al: al.query_alignment_length)

        return longest_phiX_alignment

    @memoized_property
    def aligns_to_phiX(self):
        return self.longest_phiX_alignment is not None and self.longest_phiX_alignment.query_alignment_length >= 50

    @memoized_property
    def parsimonious_extension_chain_alignments(self):
        # For recodes, target als can sometimes be redundant.
        chain_als = []

        for k in ['first target', 'second target']:
            if (al := self.extension_chain['alignments'].get(k)) is not None:
                chain_als.append(al)

        parsimonious_alignments = interval.make_parsimonious(chain_als)

        if (al := self.extension_chain['alignments'].get('pegRNA')) is not None:
            parsimonious_alignments.append(al)

        return parsimonious_alignments

    def register_intended_edit(self, single_target_alignment_without_indels=False):
        self.category = 'intended edit'

        self.relevant_alignments = self.parsimonious_extension_chain_alignments

        if single_target_alignment_without_indels:
            target_als = [al for al in self.relevant_alignments if al.reference_name == self.editing_strategy.target]
            target_als.append(self.single_read_covering_target_alignment)
            other_als = [al for al in self.relevant_alignments if al.reference_name != self.editing_strategy.target]

            self.relevant_alignments = sam.make_noncontained(target_als) + other_als

        insertions = []
        deletions = []

        if self.intended_edit_type == 'combination':
            if self.pegRNA_substitution_string == self.full_incorporation_pegRNA_substitution_string and not single_target_alignment_without_indels:
                self.subcategory = 'combination'
            else:
                self.category = 'partial edit'
                self.subcategory = 'partial incorporation'

            if not single_target_alignment_without_indels:
                if self.editing_strategy.pegRNA_programmed_deletion is not None:
                    deletions.append(self.editing_strategy.pegRNA_programmed_deletion)

                if self.editing_strategy.pegRNA_programmed_insertion is not None:
                    insertions.append(self.editing_strategy.pegRNA_programmed_insertion)

        elif self.intended_edit_type == 'insertion':
            self.subcategory = 'insertion'
            insertions.append(self.editing_strategy.pegRNA_programmed_insertion)

        elif self.intended_edit_type == 'deletion':
            self.subcategory = 'deletion'
            deletions.append(self.editing_strategy.pegRNA_programmed_deletion)

        else:
            target_alignment = self.single_read_covering_target_alignment
            
            if target_alignment is None:
                target_alignment = self.original_target_covering_alignment

            if target_alignment is not None:
                _, indels = self.interesting_and_uninteresting_indels([target_alignment])
            else:
                indels = []

            deletions.extend([indel for indel in indels if indel.kind == 'D'])
            insertions.extend([indel for indel in indels if indel.kind == 'I'])

            if self.pegRNA_substitution_string == self.full_incorporation_pegRNA_substitution_string:
                self.subcategory = 'substitution'
            else:
                self.category = 'partial edit'
                self.subcategory = 'partial incorporation'

        if self.platform == 'nanopore':
            mismatches = []
        else:
            mismatches = self.non_pegRNA_mismatches

        self.Details = Details(programmed_substitution_read_bases=self.pegRNA_substitution_string,
                               mismatches=mismatches,
                               non_programmed_edit_mismatches=self.non_programmed_edit_mismatches,
                               deletions=deletions,
                               insertions=insertions,
                              )

    def register_indels_in_original_alignment(self):
        relevant_indels, other_indels = self.indels_in_original_target_covering_alignment

        deletions = [indel for indel in relevant_indels if indel.kind == 'D']
        insertions = [indel for indel in relevant_indels if indel.kind == 'I']

        self.Details = Details(deletions=deletions, insertions=insertions, mismatches=self.non_pegRNA_mismatches)

        if len(relevant_indels) == 1:
            indel = relevant_indels[0]

            if indel == self.editing_strategy.pegRNA_programmed_insertion:
                # Splitting alignments at edit clusters may have prevented an intended
                # insertion with a cluster of low-quality mismatches from 
                # from being recognized as an intended insertion in split form.
                # Catch these here, with the caveat that this may inflate the
                # apparent ratio of intended edits to unintended rejoinings, since
                # there is no equivalent catching of those.
                self.register_intended_edit()

            else:
                if self.has_pegRNA_substitution:
                    if indel.kind == 'D':
                        subcategory = 'deletion'

                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                    self.register_edit_plus_indel(subcategory, [indel])

                else:
                    if indel.kind == 'D':
                        self.category = 'deletion'
                        self.relevant_alignments = [self.original_target_covering_alignment]

                    elif indel.kind == 'I':
                        self.category = 'insertion'
                        self.relevant_alignments = [self.original_target_covering_alignment] + self.non_protospacer_pegRNA_alignments

                    if len(self.mismatches_in_original_target_covering_alignment) > 0:
                        self.subcategory = 'mismatches'
                    else:
                        self.subcategory = 'clean'


        else:
            self.category = 'multiple indels'
            self.subcategory = 'multiple indels'

            self.relevant_alignments = [self.original_target_covering_alignment]

    def register_nonspecific_amplification(self):
        results = self.nonspecific_amplification

        self.category = 'nonspecific amplification'

        if self.non_primer_nts <= 2:
            self.subcategory = 'primer dimer'
            self.relevant_alignments = self.target_edge_alignments_list

        elif len(results['covering_als']) == 0:
            self.subcategory = 'short unknown'
            self.relevant_alignments = sam.make_noncontained(self.uncategorized_relevant_alignments)

        else:
            if self.editing_strategy.pegRNA_names is None:
                pegRNA_names = []
            else:
                pegRNA_names = self.editing_strategy.pegRNA_names

            if any(al.reference_name in pegRNA_names for al in results['covering_als']):
                # amplification off of pegRNA-expressing plasmid
                self.subcategory = 'extra sequence'

            elif any(al in self.extra_alignments for al in results['covering_als']):
                self.subcategory = 'extra sequence'
            
            elif any(al.reference_name not in self.primary_ref_names for al in results['covering_als']):
                organisms = {self.editing_strategy.remove_organism_from_alignment(al)[0] for al in results['covering_als'] if al.reference_name not in self.primary_ref_names}
                organism = sorted(organisms)[0]
                self.subcategory = organism

            elif any(al.reference_name == self.editing_strategy.target for al in results['covering_als']) and self.editing_strategy.genome_source is not None:
                # reference name of supplemental al has been replaced
                self.subcategory = self.editing_strategy.genome_source

            else:
                raise ValueError

            self.relevant_alignments = results['target_edge_als'] + results['covering_als']

    def register_edit_plus_indel(self, subcategory, indels):
        self.category = 'edit + indel'
        self.subcategory = subcategory

        deletions = [indel for indel in indels if indel.kind == 'D']
        insertions = [indel for indel in indels if indel.kind == 'I']

        # Can't just use self.pegRNA_extension_als_list since indel
        # might cut off extension chain on one side.

        als = interval.make_parsimonious(self.split_target_alignments) + self.pegRNA_extension_als_from_either_side_list
        als = sam.merge_any_adjacent_pairs(als, self.editing_strategy.reference_sequences, max_insertion_length=2)
        self.relevant_alignments = als

        if self.intended_edit_type == 'insertion':
            insertions.append(self.editing_strategy.pegRNA_programmed_insertion)

        elif self.intended_edit_type == 'deletion':
            deletions.append(self.editing_strategy.pegRNA_programmed_deletion)

        self.Details = Details(programmed_substitution_read_bases=self.pegRNA_substitution_string,
                               mismatches=self.non_pegRNA_mismatches,
                               non_programmed_edit_mismatches=self.non_programmed_edit_mismatches,
                               deletions=deletions,
                               insertions=insertions,
                              )

    def is_valid_unintended_rejoining(self, chains):
        ''' There is RT'ed sequence, and the extension chains cover the whole read.
        '''
        # Note difference from twin prime here - excludes RT'ed + annealing-extended.
        contains_RTed_sequence = (chains[self.pegRNA_side]['description'] == 'RT\'ed')

        empty = interval.DisjointIntervals([interval.Interval.empty()])
        pegRNA_side_covered = chains[self.pegRNA_side]['query_covered_incremental'].get('pegRNA', empty)
        non_pegRNA_side_covered = chains[self.non_pegRNA_side]['query_covered_incremental'].get('first target', empty)

        combined_covered = pegRNA_side_covered | non_pegRNA_side_covered
        uncovered = self.not_covered_by_primers - combined_covered

        # Allow failure to explain the last few nts of the read.
        uncovered = uncovered & self.whole_read_minus_edges(2)

        return contains_RTed_sequence and uncovered.total_length == 0

    @memoized_property
    def is_unintended_rejoining(self):
        return self.is_valid_unintended_rejoining(self.extension_chains_by_side)

    @memoized_property
    def is_possible_unintended_rejoining(self):
        return self.is_valid_unintended_rejoining(self.possible_extension_chains_by_side)

    def register_unintended_rejoining(self):
        if self.is_unintended_rejoining:
            chain = self.extension_chain
            chains = self.extension_chains_by_side
            chain_edges = self.extension_chain_edges
            chain_junction_MH = self.extension_chain_junction_microhomology

        elif self.is_possible_unintended_rejoining:
            chain = self.possible_extension_chain
            chains = self.possible_extension_chains_by_side
            chain_edges = self.possible_extension_chain_edges
            chain_junction_MH = self.possible_extension_chain_junction_microhomology

        else:
            raise ValueError

        if self.flipped:
            possibly_flipped_side = {
                'left': 'right',
                'right': 'left',
            }
        else:
            possibly_flipped_side = {
                'left': 'left',
                'right': 'right',
            }

        pegRNA_al = chain['alignments']['pegRNA']
        has_pegRNA_substitution = self.specific_to_pegRNA(pegRNA_al)

        self.category = 'unintended rejoining of RT\'ed sequence'

        if self.alignment_scaffold_overlap(pegRNA_al) >= 2:
            self.subcategory = 'includes scaffold'
        else:
            self.subcategory = 'no scaffold'

        if not has_pegRNA_substitution:
            self.subcategory += ', no substitution'

        # TODO: need to add integrase sites here.

        details_kwargs = {}

        if chain_edges[possibly_flipped_side['left']] is not None:
            details_kwargs['left_rejoining_edge'] = chain_edges[possibly_flipped_side['left']]

        if chain_edges[possibly_flipped_side['right']] is not None:
            details_kwargs['right_rejoining_edge'] = chain_edges[possibly_flipped_side['right']]

        if chain_junction_MH is not None:
            details_kwargs['junction_microhomology_length'] = chain_junction_MH

        self.Details = Details(**details_kwargs)

        self.relevant_alignments = []

        for side, key in [
            (self.pegRNA_side, 'first target'),
            (self.pegRNA_side, 'pegRNA'),
            (self.non_pegRNA_side, 'first target'),
        ]:
            if key in chains[side]['alignments']:
                self.relevant_alignments.append(chains[side]['alignments'][key])

    def register_incorporation_of_extra_sequence(self):
        self.category = 'incorporation of extra sequence'

        if any(self.extension_chains_by_side[side]['description'].startswith('RT') for side in ['left', 'right']):
            self.subcategory = 'has RT\'ed extension'
        else:
            self.subcategory = 'no RT\'ed extension'

        alignments = (
            interval.make_parsimonious(self.target_alignments + self.pegRNA_alignments + self.nonredundant_extra_alignments) + 
            self.pegRNA_extension_als_list
        )

        self.relevant_alignments = sam.make_nonredundant(alignments)

    @memoized_property
    def is_low_quality(self):
        num_Ns = Counter(self.seq)['N']

        is_low_quality = (num_Ns > 10) or (self.Q30_fractions['all'] < 0.5) or (self.Q30_fractions['second_half'] < 0.5)

        return is_low_quality

    def register_uncategorized(self):
        self.category = 'uncategorized'

        if self.is_low_quality:
            self.subcategory = 'low quality'
        else:
            self.subcategory = 'uncategorized'

        self.relevant_alignments = self.uncategorized_relevant_alignments

    def register_minimal_alignments_detected(self):
        self.category = 'minimal alignment to intended target'
        self.subcategory = 'n/a'
        self.relevant_alignments = self.uncategorized_relevant_alignments

    @memoized_property
    def pegRNA_alignments_cover_target_gap(self):
        meaningful_gap_covers = []
        
        gap = self.not_covered_by_target_edge_alignments

        if gap.total_length > 0:
            if self.editing_strategy.pegRNA_side == 'left':
                relevant_pegRNA_strand = '-'
            else:
                relevant_pegRNA_strand = '+'
            
            pegRNA_als = self.pegRNA_alignments
            pegRNA_als = [al for al in pegRNA_als if sam.get_strand(al) == relevant_pegRNA_strand]
            pegRNA_als = [al for al in pegRNA_als if not self.is_pegRNA_protospacer_alignment(al)]

            covered_by_pegRNA_alignments = interval.get_disjoint_covered(pegRNA_als)
            gap_not_covered_by_pegRNA_alignments = gap - covered_by_pegRNA_alignments
            if gap_not_covered_by_pegRNA_alignments.total_length == 0:
                meaningful_gap_covers = pegRNA_als
                
        return meaningful_gap_covers

    def register_single_read_covering_target_alignment(self):
        target_alignment = self.single_read_covering_target_alignment
        interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

        deletions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'D']
        insertions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'I']

        if self.platform == 'nanopore':
            mismatches = []
        else:
            mismatches = self.non_pegRNA_mismatches

        self.Details = Details(
            deletions=deletions,
            insertions=insertions,
            mismatches=mismatches,
        )

        if len(interesting_indels) == 0:
            if self.starts_at_expected_location:
                if self.specific_to_pegRNA(self.single_read_covering_target_alignment):
                    self.register_intended_edit(single_target_alignment_without_indels=True)

                else:
                    self.category = 'wild type'

                    if len(self.non_pegRNA_mismatches) == 0 and len(uninteresting_indels) == 0:
                        self.subcategory = 'clean'

                    elif len(uninteresting_indels) == 1:
                        self.subcategory = 'short indel far from cut'

                    elif len(uninteresting_indels) > 1:
                        self.register_uncategorized()

                    else:
                        self.subcategory = 'mismatches'

                    self.relevant_alignments = [target_alignment]

            else:
                self.register_uncategorized()

        elif self.max_scaffold_overlap >= 2 and self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif len(interesting_indels) == 1:
            indel = interesting_indels[0]

            if self.has_pegRNA_substitution:
                if indel.kind == 'D':
                    if self.is_unintended_rejoining:
                        self.register_unintended_rejoining()
                    else:
                        self.register_edit_plus_indel('deletion', [indel])

                else:
                    self.register_uncategorized()

            else: # no pegRNA mismatches
                if len(self.non_pegRNA_mismatches) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    self.category = 'deletion'
                    self.relevant_alignments = self.target_edge_alignments_list

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.relevant_alignments = [target_alignment]

        else: # more than one indel
            if len(self.indels) == 2:

                indels = [indel for indel, near_cut in self.indels]

                if self.editing_strategy.pegRNA_programmed_deletion in indels:
                    indel = [indel for indel in indels if indel != self.editing_strategy.pegRNA_programmed_deletion][0]

                    if indel.kind  == 'D':
                        self.register_edit_plus_indel('deletion', [indel])

                    else:
                        self.register_uncategorized()

                else:
                    self.category = 'multiple indels'
                    self.subcategory = 'multiple indels'

                    self.relevant_alignments = [target_alignment]

            else:
                self.register_uncategorized()

    @memoized_property
    def no_alignments_detected(self):
        return all(al.is_unmapped for al in self.alignments)

    def categorize(self):
        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.no_alignments_detected:
            self.register_minimal_alignments_detected()

        elif self.aligns_to_phiX:
            self.category = 'phiX'
            self.subcategory = 'phiX'

            self.relevant_alignments = [self.longest_phiX_alignment]

        elif self.is_intended_edit:
            self.register_intended_edit()

        elif self.single_read_covering_target_alignment:
            self.register_single_read_covering_target_alignment()

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.is_possible_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.pegRNA_alignments_cover_target_gap:
            self.category = 'complex incorporation of RT\'ed sequence'
            self.subcategory = 'n/a'
            strat = self.editing_strategy
            PBS_al = self.generate_extended_pegRNA_PBS_alignment(self.target_edge_alignments[self.pegRNA_side], self.pegRNA_side)
            als = self.target_edge_alignments_list + interval.make_parsimonious(self.pegRNA_alignments_cover_target_gap)
            if PBS_al is not None:
                als.append(PBS_al)

            als = sam.merge_any_adjacent_pairs(als, strat.reference_sequences, max_deletion_length=2, max_insertion_length=2)
            self.relevant_alignments = als

        elif self.duplication_covers_whole_read:
            subcategory, ref_junctions, indels, als_with_pegRNA_substitutions, merged_als = self.duplication
            self.Details = Details(duplication_junctions=ref_junctions)

            if als_with_pegRNA_substitutions == 0:
                self.category = 'duplication'
                self.subcategory = subcategory
            else:
                self.category = 'edit + indel'
                self.subcategory = 'duplication'

            self.relevant_alignments = self.pegRNA_extension_als_from_either_side_list + merged_als

        elif self.inversion:
            self.category = 'inversion'
            self.subcategory = 'inversion'

            self.relevant_alignments = self.target_edge_alignments_list + self.inversion

        elif self.nonredundant_extra_alignments:
            self.register_incorporation_of_extra_sequence()

        elif self.original_target_alignment_has_no_indels:
            self.category = 'wild type'
            # Assume clean would have been caught before.
            self.subcategory = 'mismatches'
            self.Details = Details(deletions=[], insertions=[], mismatches=self.mismatches_in_original_target_covering_alignment)
            self.relevant_alignments = [self.original_target_covering_alignment]

        elif self.duplication is not None:
            subcategory, ref_junctions, indels, als_with_pegRNA_substitutions, merged_als = self.duplication
            self.relevant_alignments = self.pegRNA_extension_als_list + merged_als

            if len(indels) == 0:
                if als_with_pegRNA_substitutions == 0:
                    self.category = 'duplication'
                    self.subcategory = subcategory
                else:
                    self.category = 'edit + indel'
                    self.subcategory = 'duplication'

                self.relevant_alignments = self.pegRNA_extension_als_from_either_side_list + merged_als

                self.Details = Details(duplication_junctions=ref_junctions)

            elif len(indels) == 1 and indels[0].kind == 'D':
                indel = indels[0]

                if indel == self.editing_strategy.pegRNA_programmed_deletion:
                    self.category = 'edit + indel'
                    self.subcategory = 'duplication'
                else:
                    self.category = 'multiple indels'
                    self.subcategory = 'duplication + deletion'

                self.Details = Details(deletions=indels, duplication_junctions=ref_junctions)

                self.relevant_alignments = self.pegRNA_extension_als_from_either_side_list + merged_als

            elif len(indels) == 1 and indels[0].kind == 'I':
                indel = indels[0]
                self.category = 'multiple indels'
                self.subcategory = 'duplication + insertion'

            else:
                raise ValueError('duplication shouldn\'t have >1 indel') 

        elif self.duplication_plus_edit is not None:
            self.category = 'edit + indel'
            self.subcategory = 'duplication'
            self.relevant_alignments = self.duplication_plus_edit

        elif self.is_deletion_plus_edit is not None:
            deletion = self.is_deletion_plus_edit
            self.register_edit_plus_indel('deletion', [deletion])

        elif self.original_target_alignment_has_only_relevant_indels:
            self.register_indels_in_original_alignment()

        elif self.query_length_covered_by_on_target_alignments <= 30:
            self.register_minimal_alignments_detected()

        else:
            self.register_uncategorized()

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        self.categorized = True

        return self.category, self.subcategory, self.details, self.Details

    @memoized_property
    def nonredundant_extra_alignments(self):
        ''' Alignments from extra sequences that explain a substantial portion
        of the read not covered by target or pegRNA alignemnts.
        '''
        relevant_extra_als = []

        need_to_cover = self.not_covered_by_split_target_or_pegRNA_alignments & self.not_covered_by_target_edge_alignments

        potentially_relevant_als = [
            al for al in self.extra_alignments
            if ((interval.get_covered(al) & need_to_cover).total_length > 0)
            and (sam.total_edit_distance(al, self.editing_strategy.reference_sequences[al.reference_name]) < 5)
        ]

        if len(potentially_relevant_als) > 0:
            covered = interval.get_disjoint_covered(potentially_relevant_als)

            covered_by_extra = covered & need_to_cover

            if covered_by_extra.total_length >= 10:
                relevant_extra_als = interval.make_parsimonious(potentially_relevant_als)

        return relevant_extra_als

    @memoized_property
    def duplications_from_each_read_edge(self):
        strat = self.editing_strategy
        target_als = interval.make_parsimonious(self.target_alignments)
        # Order target als by position on the query from left to right.
        target_als = sorted(target_als, key=interval.get_covered)

        correct_strand_als = [al for al in target_als if sam.get_strand(al) == self.sequencing_direction]

        # Need deletions to be merged.
        merged_als = sam.merge_any_adjacent_pairs(correct_strand_als, strat.reference_sequences)
        
        intervals = [interval.get_covered(al) for al in merged_als]
        
        if len(merged_als) > 0 and self.overlaps_primer(merged_als[0], 'left'):
            no_gaps_through_index = 0
            
            for i in range(1, len(intervals)):
                cumulative_from_left = interval.make_disjoint(intervals[:i + 1])
                
                # If there are no gaps so far
                if len(cumulative_from_left.intervals) == 1:
                    no_gaps_through_index = i
                else:
                    break
                    
            from_left_edge = merged_als[:no_gaps_through_index + 1]
        else:
            from_left_edge = []
            
        if len(merged_als) > 0 and \
           (self.overlaps_primer(merged_als[len(intervals) - 1], 'right') or
            (intervals[-1].end >= self.whole_read.end - 1 and len(intervals[-1]) >= 20)
           ):
            no_gaps_through_index = len(intervals) - 1
            
            for i in range(len(intervals) - 1 - 1, -1, -1):
                cumulative_from_right = interval.make_disjoint(intervals[i:])
                
                # If there are no gaps so far
                if len(cumulative_from_right.intervals) == 1:
                    no_gaps_through_index = i
                else:
                    break
                    
            from_right_edge = merged_als[no_gaps_through_index:]
        else:
            from_right_edge = []
        
        return {'left': from_left_edge, 'right': from_right_edge}

    @memoized_property
    def duplication_plus_edit(self):
        alignments = None

        if self.editing_strategy.pegRNA_names is not None and len(self.editing_strategy.pegRNA_names) > 0:
            duplication_als = self.duplications_from_each_read_edge[self.non_pegRNA_side]
            if len(duplication_als) > 1:
                covered_by_duplication = interval.get_disjoint_covered(duplication_als)

                chain_als = self.extension_chain['alignments']

                if 'pegRNA' in chain_als and 'second target' in chain_als:
                    combined_covered = self.extension_chain['query_covered'] | covered_by_duplication

                    if self.matches_all_programmed_insertion_features:
                        uncovered = self.whole_read_minus_edges(2) - combined_covered
                    
                        if uncovered.total_length == 0:
                            alignments = list(chain_als.values()) + duplication_als

        if alignments is not None:
            target_als = [al for al in alignments if al.reference_name == self.editing_strategy.target]
            target_als = sam.make_noncontained(target_als)
            other_als = [al for al in alignments if al.reference_name != self.editing_strategy.target]

            alignments = target_als + other_als

        return alignments

    @memoized_property
    def is_deletion_plus_edit(self):
        deletion = None

        if self.editing_strategy.pegRNA_names is not None and len(self.editing_strategy.pegRNA_names) > 0:
            target_als = self.duplications_from_each_read_edge[self.non_pegRNA_side]

            if len(target_als) == 1:
                covered_after_deletion = interval.get_disjoint_covered(target_als)

                chain_als = self.extension_chain['alignments']

                if 'pegRNA' in chain_als and 'second target' in chain_als:
                    combined_covered = self.extension_chain['query_covered'] | covered_after_deletion

                    uncovered = self.whole_read_minus_edges(2) - combined_covered
                
                    if uncovered.total_length == 0:
                        not_programmed_indels = [indel for indel, _ in self.extract_indels_from_alignments(target_als) if indel != self.editing_strategy.pegRNA_programmed_deletion]
                        
                        not_programmed_deletions = [indel for indel in not_programmed_indels if indel.kind == 'D']
                        
                        if len(not_programmed_indels) == 1 and len(not_programmed_deletions) == 1:
                            deletion = not_programmed_deletions[0]                        

        return deletion

    @memoized_property
    def duplication(self):
        ''' (duplication, simple)   - a single junction
            (duplication, iterated) - multiple uses of the same junction
            (duplication, complex)  - multiple junctions that are not exactly the same
        '''
        strat = self.editing_strategy
        target_als = interval.make_parsimonious(self.target_alignments)
        # Order target als by position on the query from left to right.
        target_als = sorted(target_als, key=interval.get_covered)

        correct_strand_als = [al for al in target_als if sam.get_strand(al) == self.sequencing_direction]

        merged_als = sam.merge_any_adjacent_pairs(correct_strand_als, strat.reference_sequences)
    
        # See docstring for not_covered_by_primers to make sense of this.
        relevant_interval = self.covered_by_primers | self.not_covered_by_primers
        relevant_als = [al for al in merged_als if (interval.get_covered(al) & relevant_interval).total_length >= 5]

        # TODO: update to use self.not_covered_by_primers
        covereds = []
        for al in relevant_als:
            covered = interval.get_covered(al)
            if covered.total_length >= 20:
                if self.overlaps_primer(al, 'right'):
                    covered.end = self.whole_read.end
                if self.overlaps_primer(al, 'left'):
                    covered.start = 0
            covereds.append(covered)
    
        covered = interval.make_disjoint(covereds)

        uncovered = self.whole_read_minus_edges(2) - covered
        
        if len(relevant_als) == 1 or uncovered.total_length > 0:
            return None
        
        ref_junctions = []

        indels = []

        als_with_pegRNA_substitutions = sum(self.specific_to_pegRNA(al) for al in relevant_als)

        indels = [indel for indel, _ in self.extract_indels_from_alignments(relevant_als)]

        for left_al, right_al in zip(relevant_als, relevant_als[1:]):
            switch_results = sam.find_best_query_switch_after(left_al, right_al, strat.target_sequence, strat.target_sequence, max)

            lefts = tuple(sam.closest_ref_position(q, left_al) for q in switch_results['best_switch_points'])
            rights = tuple(sam.closest_ref_position(q + 1, right_al) for q in switch_results['best_switch_points'])

            # Don't consider duplications of 1 or 2 nts.
            if abs(lefts[0] - rights[0]) <= 2:
                # placeholder, only needs to be of kind 'I'
                indel = knock_knock.outcome.DegenerateInsertion([-1], ['N'])
                indels.append(indel)
                continue

            ref_junction = DuplicationJunction(lefts, rights)
            ref_junctions.append(ref_junction)

        if len(indels) > 1:
            return None

        if len(ref_junctions) == 0:
            return None

        elif len(ref_junctions) == 1:
            subcategory = 'simple'

        elif len(set(ref_junctions)) == 1:
            # There are multiple junctions but they are all identical.
            subcategory = 'iterated'

        else:
            subcategory = 'complex'

        return subcategory, ref_junctions, indels, als_with_pegRNA_substitutions, relevant_als

    @memoized_property
    def duplication_covers_whole_read(self):
        if self.duplication is None:
            return False
        else:
            _, _, indels, _, merged_als = self.duplication
            not_covered = self.whole_read - interval.get_disjoint_covered(merged_als)
            return (not_covered.total_length == 0) and (len(indels) == 0)

    @memoized_property
    def inversion(self):
        need_to_cover = self.not_covered_by_target_edge_alignments
        inversion_als = []
        
        if need_to_cover.total_length >= 5:
            flipped_target_als = [al for al in self.target_alignments if sam.get_strand(al) != self.sequencing_direction]
        
            for al in flipped_target_als:
                covered = interval.get_covered(al)
                if covered.total_length >= 5 and (need_to_cover - covered).total_length == 0:
                    inversion_als.append(al)
                    
        return inversion_als

    @memoized_property
    def indels_in_original_target_covering_alignment(self):
        max_insertion_length = 3

        relevant_indels = []
        other_indels = []

        if self.original_target_covering_alignment is not None:
            for indel, near_cut in self.extract_indels_from_alignments([self.original_target_covering_alignment]):
                if indel.kind == 'D' or (indel.kind == 'I' and indel.length <= max_insertion_length):
                    relevant_indels.append(indel)
                else:
                    other_indels.append(indel)

        return relevant_indels, other_indels

    @memoized_property
    def mismatches_in_original_target_covering_alignment(self):
        # Don't want to consider probably spurious alignments to parts of the query that
        # should have been trimmed. 

        relevant_alignments  = [self.original_target_covering_alignment] + self.pegRNA_extension_als_list

        _, non_pegRNA_mismatches = self.summarize_mismatches_in_alignments(relevant_alignments)

        return non_pegRNA_mismatches

    @memoized_property
    def non_pegRNA_mismatches(self):
        _, non_pegRNA_mismatches = self.mismatches_summary

        # Remove mismatches at programmed posititions. 
        if self.editing_strategy.pegRNA is None:
            programmed_ps = set()
        else:
            programmed_ps = self.editing_strategy.pegRNA.programmed_substitution_target_ps

        non_pegRNA_mismatches = [mismatch for mismatch in non_pegRNA_mismatches if mismatch.position not in programmed_ps]

        return non_pegRNA_mismatches

    @memoized_property
    def non_programmed_edit_mismatches(self):
        return knock_knock.outcome.Mismatches([])

    @memoized_property
    def original_target_alignment_has_no_indels(self):
        if self.original_target_covering_alignment is None:
            return False

        relevant_indels, other_indels = self.indels_in_original_target_covering_alignment

        return len(relevant_indels) == 0 and len(other_indels) == 0

    @memoized_property
    def original_target_alignment_has_only_relevant_indels(self):
        if self.original_target_covering_alignment is None:
            return False

        relevant_indels, other_indels = self.indels_in_original_target_covering_alignment

        return len(relevant_indels) > 0 and len(other_indels) == 0

    def generate_extended_target_PBS_alignment(self, pegRNA_al):
        pegRNA_name = pegRNA_al.reference_name
        target_PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
        return self.extend_alignment_from_shared_feature(pegRNA_al, 'PBS', self.editing_strategy.target, target_PBS_name)

    def generate_extended_pegRNA_PBS_alignment(self, target_al, side):
        if target_al is None:
            extended_al = None
        else:
            pegRNA_name = self.pegRNA_names_by_side_of_read[side]
            target_PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
            extended_al = self.extend_alignment_from_shared_feature(target_al, target_PBS_name, pegRNA_name, 'PBS')

        return extended_al

    def is_pegRNA_protospacer_alignment(self, al):
        ''' Returns True if al aligns almost entirely to a protospacer region of a pegRNA,
        typically for the purpose of deciding whether to plot it.
        '''
        strat = self.editing_strategy
        
        if strat.pegRNA_names is None:
            return False
        
        if al.reference_name not in strat.pegRNA_names:
            return False
        
        PS_feature = strat.features[al.reference_name, 'protospacer']
        outside_protospacer = sam.crop_al_to_ref_int(al, PS_feature.end + 1, np.inf)

        return (outside_protospacer is None or outside_protospacer.query_alignment_length <= 3)

    @memoized_property
    def uncategorized_relevant_alignments(self):
        als = self.target_alignments + self.pegRNA_alignments + self.extra_alignments + interval.make_parsimonious(self.nonredundant_supplemental_alignments)

        als = [al for al in als if not self.is_pegRNA_protospacer_alignment(al)]

        if self.editing_strategy.pegRNA_names is None:
            pegRNA_names = []
        else:
            pegRNA_names = self.editing_strategy.pegRNA_names

        pegRNA_als = [al for al in als if al.reference_name in pegRNA_names]
        target_als = [al for al in als if al.reference_name == self.editing_strategy.target]
        other_als = [al for al in als if al.reference_name not in pegRNA_names + [self.editing_strategy.target]]

        pegRNA_als = sam.make_noncontained(pegRNA_als)

        for al in pegRNA_als:
            # If it is already an extension al, the corresponding target al must already exist.
            if al not in self.pegRNA_extension_als_list:
                extended_al = self.generate_extended_target_PBS_alignment(al)
                if extended_al is not None:
                    target_als.append(extended_al)

        target_als = sam.make_noncontained(target_als)

        als = pegRNA_als + target_als + other_als

        als = sam.make_noncontained(als, max_length=10)

        return als

    @property
    def inferred_amplicon_length(self):
        ''' Infer the length of the amplicon including the portion
        of primers that is present in the genome. To prevent synthesis
        errors in primers from shifting this slightly, identify the
        distance in the query between the end of the left primer and
        the start of the right primer, then add the expected length of
        both primers to this. If the sequencing read is single-end
        and doesn't reach the right primer but ends in an alignment
        to the target, parsimoniously assume that this alignment continues
        on through the primer to infer length.
        ''' 

        if self.seq  == '':
            return 0
        elif (self.whole_read - self.covered_by_primers).total_length == 0:
            return len(self.seq)
        elif len(self.seq) <= 50:
            return len(self.seq)

        left_al = self.target_edge_alignments['left']
        right_al = self.target_edge_alignments['right']

        left_primer = self.primers_by_side_of_read['left']
        right_primer = self.primers_by_side_of_read['right']

        left_offset_to_q = self.feature_offset_to_q(left_al, left_primer.ID)
        right_offset_to_q = self.feature_offset_to_q(right_al, right_primer.ID)

        # Only trust the inferred length if there are non-spurious target alignments
        # to both edges.
        def is_nonspurious(al):
            min_nonspurious_length = 15
            return al is not None and al.query_alignment_length >= min_nonspurious_length

        if is_nonspurious(left_al) and is_nonspurious(right_al):
            # Calculate query distance between inner primer edges.
            if len(left_offset_to_q) > 0:
                left_inner_edge_offset = max(left_offset_to_q)
                left_inner_edge_q = left_offset_to_q[left_inner_edge_offset]
            else:
                left_inner_edge_q = 0

            if len(right_offset_to_q) > 0:
                right_inner_edge_offset = max(right_offset_to_q)
                right_inner_edge_q = right_offset_to_q[right_inner_edge_offset]
            else:
                right_inner_edge_q = sam.query_interval(right_al)[1]

            # *_inner_edge_q is last position in the primer, so shift each by one to 
            # have boundaries of region between them.
            length_seen_between_primers = (right_inner_edge_q - 1) - (left_inner_edge_q + 1) + 1

            right_al_edge_in_target = sam.reference_edges(right_al)[3]

            # Calculated inferred unseen length.
            if self.sequencing_direction == '+':
                distance_to_right_primer = right_primer.start - right_al_edge_in_target
            else:
                distance_to_right_primer = right_al_edge_in_target - right_primer.end

            # right_al might extend past the primer start, so only care about positive values.
            inferred_extra_at_end = max(distance_to_right_primer, 0)

            # Combine seen with inferred unseen and expected primer legnths.
            inferred_length = length_seen_between_primers + inferred_extra_at_end + len(left_primer) + len(right_primer)

        else:
            inferred_length = -1

        return inferred_length

    @memoized_property
    def manual_anchors(self):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        strat = self.editing_strategy

        manual_anchors = {}

        extension_als = self.pegRNA_extension_als_list

        if len(extension_als) > 0:
            pegRNA_name = strat.pegRNA_names[0]
            extension_al = extension_als[0]

            PBS_offset_to_qs = self.feature_offset_to_q(extension_al, 'PBS')
                
            if PBS_offset_to_qs:
                anchor_offset = sorted(PBS_offset_to_qs)[0]
                q = PBS_offset_to_qs[anchor_offset]

                ref_p = strat.feature_offset_to_ref_p(pegRNA_name, 'PBS')[anchor_offset]

                manual_anchors[pegRNA_name] = (q, ref_p)
                
        return manual_anchors

    def plot_parameters(self):
        strat = self.editing_strategy

        features_to_show = {*strat.features_to_show}
        label_overrides = {}
        label_offsets = {}
        feature_heights = {}
        color_overrides = {}

        flip_target = (self.sequencing_direction == '-')
        flip_pegRNA = False

        if len(strat.pegRNA_names) == 1:
            pegRNA_name = strat.pegRNA_names[0]

            PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
            PBS_strand = strat.features[strat.target, PBS_name].strand

            flip_pegRNA = (flip_target and PBS_strand == '-') or (not flip_target and PBS_strand == '+')

            label_overrides[f'HA_RT_{pegRNA_name}'] = 'HA_RT'

            for HA_side in ['PBS', 'RT']:
                name = f'HA_{HA_side}_{pegRNA_name}'
                label_overrides[strat.target, name] = None
                feature_heights[strat.target, name] = 0.5

                features_to_show.add((strat.target, name))

            for name in strat.protospacer_names:
                if name == strat.primary_protospacer:
                    new_name = 'pegRNA\nprotospacer'
                else:
                    new_name = 'ngRNA\nprotospacer'

                label_overrides[name] = new_name

        else:
            for pegRNA_name in strat.pegRNA_names:
                color = strat.pegRNA_name_to_color[pegRNA_name]
                light_color = hits.visualize.apply_alpha(color, 0.5)
                color_overrides[pegRNA_name] = color
                color_overrides[pegRNA_name, 'protospacer'] = light_color
                ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
                color_overrides[ps_name] = light_color

                PAM_name = f'{ps_name}_PAM'
                color_overrides[PAM_name] = color

            for name in strat.protospacer_names:
                label_overrides[name] = 'protospacer'

            for name in strat.PAM_features:
                label_overrides[name] = 'PAM'
                features_to_show.add((strat.target, name))

        for primer_name in strat.primer_names:
            color_overrides[primer_name] = 'lightgrey'

        for pegRNA_name in strat.pegRNA_names:
            PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
            features_to_show.add((strat.target, PBS_name))
            label_overrides[PBS_name] = None
            feature_heights[PBS_name] = 0.5

            # Draw PBS feature on the same side as corresponding nick.
            feature = strat.features[strat.target, PBS_name]
            if (feature.strand == '+' and not flip_target) or (feature.strand == '-' and flip_target):
                feature_heights[PBS_name] *= -1

        for deletion in strat.pegRNA_programmed_deletions:
            label_overrides[deletion.ID] = f'programmed deletion ({len(deletion)} nts)'
            feature_heights[deletion.ID] = -0.5

        for insertion in strat.pegRNA_programmed_insertion_features:
            label_overrides[insertion.ID] = 'insertion'
            label_offsets[insertion.ID] = 1

        for feature_name, feature in strat.PAM_features.items():
            if (feature.strand == '+' and not flip_target) or (feature.strand == '-' and flip_target):
                feature_heights[feature_name] = -1

            if len(strat.pegRNA_names) == 1:
                offset = 2
            else:
                offset = 1

            label_offsets[feature_name] = offset

        for feature_name, feature in strat.protospacer_features.items():
            if (feature.strand == '+' and not flip_target) or (feature.strand == '-' and flip_target):
                feature_heights[feature_name] = -1

        features_to_show.update({(strat.target, name) for name in strat.protospacer_names})
        features_to_show.update({(strat.target, name) for name in strat.PAM_features})

        plot_parameters = {
            'features_to_show': features_to_show,
            'label_overrides': label_overrides,
            'label_offsets': label_offsets,
            'feature_heights': feature_heights,
            'color_overrides': color_overrides,
            'flip_target': flip_target,
            'flip_pegRNA': flip_pegRNA,
        }

        return plot_parameters

    def plot(self,
             relevant=True,
             manual_alignments=None,
             extra_features_to_show=None,
             **manual_diagram_kwargs,
            ):

        plot_parameters = self.plot_parameters()

        features_to_show = manual_diagram_kwargs.pop('features_to_show', plot_parameters['features_to_show'])
        label_overrides = manual_diagram_kwargs.pop('label_overrides', plot_parameters['label_overrides'])
        label_offsets = manual_diagram_kwargs.pop('label_offsets', plot_parameters['label_offsets'])
        feature_heights = manual_diagram_kwargs.pop('feature_heights', plot_parameters['feature_heights'])

        if extra_features_to_show is not None:
            features_to_show.update(extra_features_to_show)

        flip_target = plot_parameters['flip_target']
        flip_pegRNA = plot_parameters['flip_pegRNA']

        if relevant and not self.categorized:
            self.categorize()

        strat = self.editing_strategy

        if manual_diagram_kwargs.get('reverse_complement', False):
            flip_pegRNA = not flip_pegRNA

        if 'refs_to_draw' in manual_diagram_kwargs:
            refs_to_draw = manual_diagram_kwargs.pop('refs_to_draw')
        else:
            refs_to_draw = set()

            if strat.amplicon_length < 10000:
                refs_to_draw.add(strat.target)

            if strat.pegRNA_names is not None:
                refs_to_draw.update(strat.pegRNA_names)

        if 'phiX' in strat.supplemental_indices:
            supplementary_reference_sequences = strat.supplemental_reference_sequences('phiX')
        else:
            supplementary_reference_sequences = {}

        if relevant:
            manual_anchors = manual_diagram_kwargs.get('manual_anchors', self.manual_anchors)
            inferred_amplicon_length = self.inferred_amplicon_length
        else:
            manual_anchors = {}
            inferred_amplicon_length = None

        diagram_kwargs = dict(
            draw_sequence=True,
            flip_target=flip_target,
            split_at_indels=False,
            features_to_show=features_to_show,
            manual_anchors=manual_anchors,
            refs_to_draw=refs_to_draw,
            label_offsets=label_offsets,
            label_overrides=label_overrides,
            inferred_amplicon_length=inferred_amplicon_length,
            highlight_programmed_substitutions=True,
            feature_heights=feature_heights,
            supplementary_reference_sequences=supplementary_reference_sequences,
        )

        for k, v in diagram_kwargs.items():
            manual_diagram_kwargs.setdefault(k, v)

        if manual_alignments is not None:
            als_to_plot = manual_alignments
        elif relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.alignments

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 strat,
                                                                 **manual_diagram_kwargs,
                                                                )

        # Note that diagram.alignments may be different than als_to_plot
        # due to application of parsimony.

        # Draw the pegRNA.
        if strat.pegRNA_names is not None and any(al.reference_name in strat.pegRNA_names for al in diagram.alignments):

            pegRNA_name = strat.pegRNA_names[0]

            ref_y = diagram.max_y + diagram.target_and_donor_y_gap

            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = diagram.draw_reference(pegRNA_name, ref_y, flip_pegRNA)

            pegRNA_seq = strat.reference_sequences[pegRNA_name]
            pegRNA_min_x, pegRNA_max_x = sorted([ref_p_to_xs(0), ref_p_to_xs(len(pegRNA_seq) - 1)])

            diagram.max_x = max(old_max_x, pegRNA_max_x)

            diagram.min_x = min(old_min_x, pegRNA_min_x)

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            diagram.update_size()

        return diagram