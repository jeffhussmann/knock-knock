import copy
import re

from collections import defaultdict
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import pysam

from hits import sam, interval, utilities, sw

import knock_knock.outcome

memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args
memoized_with_kwargs = utilities.memoized_with_kwargs
idx = pd.IndexSlice

def opposite_side(side):
    if side == 'left':
        opposite = 'right'
    elif side == 'right':
        opposite = 'left'
    else:
        raise ValueError(side)
    
    return opposite

class Categorizer:
    non_relevant_categories = []

    def __init__(self, alignments, editing_strategy, platform='illumina', flipped=False, **kwargs):
        self.alignments = [al for al in alignments if not al.is_unmapped]

        self.editing_strategy = editing_strategy
        self.platform = platform

        alignment = alignments[0]

        self.name = alignment.query_name
        self.query_name = self.name

        self.seq = sam.get_original_seq(alignment)
        if self.seq is None:
            self.seq = ''
        self.seq_bytes = self.seq.encode()

        # Note: don't try to make qual anything but a python array.
        # pysam will internally try to evaluate it's truth status
        # and fail.
        self.qual = sam.get_original_qual(alignment)

        self.read = sam.mapping_to_Read(alignment)

        self.read_length = len(self.read)

        self.relevant_alignments = self.alignments

        self.categorized = False

        self.details = knock_knock.outcome.Details()

        self.flipped = flipped

    @memoized_property
    def Q30_fractions(self):
        at_least_30 = np.array(self.qual) >= 30

        fracs = {
            'all': np.mean(at_least_30),
            'second_half': np.mean(at_least_30[len(at_least_30) // 2:]),
        }

        return fracs

    @property
    def primary_ref_names(self):
        return set(self.editing_strategy.reference_sequences)

    @memoized_property
    def primary_alignments(self):
        p_als = [
            al for al in self.alignments
            if al.reference_name in self.primary_ref_names
        ]
        
        return p_als

    @memoized_property
    def initial_target_alignments(self):
        t_als = [
            al for al in self.alignments
            if al.reference_name == self.editing_strategy.target
        ]
        
        return t_als

    def seed_and_extend(self, on, query_start, query_end):
        extender = self.editing_strategy.seed_and_extender[on]
        return extender(self.seq_bytes, query_start, query_end, self.query_name)
    
    @memoized_property
    def perfect_right_edge_alignment(self):
        ''' Single end reads may end in the middle of a rearrangement. '''
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
    def refined_target_alignments(self):
        ''' Initial refinement.
        '''

        refined_als = self.split_and_extend_alignments(self.initial_target_alignments)

        # TODO: figure out way to control false positives here.
        # Require to explain enough unexplained length?

        # Non-specific amplification can produce short primer alignments that may not
        # be picked up by initial alignment. Realign each edge to primers to check for this.

        for side in ['left', 'right']:
            primer_al = self.realign_edge_to_primer(side)
            if primer_al is not None:
                refined_als.append(primer_al)

        if self.perfect_right_edge_alignment is not None:
            refined_als.append(self.perfect_right_edge_alignment)

        refined_als = sam.merge_any_adjacent_pairs(refined_als,
                                                   self.editing_strategy.reference_sequences,
                                                   max_deletion_length=2,
                                                   max_insertion_length=2,
                                                  )

        refined_als = interval.make_parsimonious(refined_als)

        # Crop to a window +/- relevant_window_size on both sides

        cuts_interval = self.editing_strategy.around_or_between_cuts(0)

        relevant_window_size = 5000 # ideally a parameter

        relevant_interval = interval.Interval(cuts_interval.start - relevant_window_size, cuts_interval.end + relevant_window_size)

        refined_als = [
            cropped_al
            for al in refined_als
            if (cropped_al := sam.crop_al_to_ref_int(al, relevant_interval.start, relevant_interval.end)) is not None
        ] 

        return refined_als

    @memoized_property
    def whole_read(self):
        return interval.Interval(0, len(self.seq) - 1)

    def whole_read_minus_edges(self, edge_length):
        return interval.Interval(edge_length, len(self.seq) - 1 - edge_length)

    @classmethod
    def from_read(cls, read, editing_strategy):
        al = pysam.AlignedSegment(editing_strategy.header)
        al.query_sequence = read.seq
        al.query_qualities = read.qual
        al.query_name = read.name
        return cls([al], editing_strategy)
    
    @classmethod
    def from_seq(cls, seq, editing_strategy):
        al = pysam.AlignedSegment(editing_strategy.header)
        al.query_sequence = seq
        al.query_qualities = [41]*len(seq)
        return cls([al], editing_strategy)

    @classmethod
    def full_index(cls):
        full_index = []
        for cat, subcats in cls.category_order:
            for subcat in subcats:
                full_index.append((cat, subcat))
                
        full_index = pd.MultiIndex.from_tuples(full_index) 

        return full_index
    
    @classmethod
    def categories(cls):
        return [c for c, scs in cls.category_order]
    
    @classmethod
    def subcategories(cls):
        return {c: scs for c, scs in cls.category_order}

    @classmethod
    def order(cls, outcome):
        if isinstance(outcome, tuple):
            category, subcategory = outcome

            try:
                return (cls.categories().index(category),
                        cls.subcategories()[category].index(subcategory),
                       )
            except:
                raise ValueError(category, subcategory)
        else:
            category = outcome

            try:
                return cls.categories().index(category)
            except:
                raise ValueError(category)

    @classmethod
    def outcome_to_sanitized_string(cls, outcome):
        if isinstance(outcome, tuple):
            c, s = cls.order(outcome)
            return f'category{c:03d}_subcategory{s:03d}'
        else:
            c = cls.order(outcome)
            return f'category{c:03d}'

    @classmethod
    def sanitized_string_to_outcome(cls, sanitized_string):
        match = re.match(r'category(\d+)_subcategory(\d+)', sanitized_string)
        if match:
            c, s = map(int, match.groups())
            category, subcats = cls.category_order[c]
            subcategory = subcats[s]
            return category, subcategory
        else:
            match = re.match(r'category(\d+)', sanitized_string)
            if not match:
                raise ValueError(sanitized_string)
            c = int(match.group(1))
            category, subcats = cls.category_order[c]
            return category

    @property
    def programmed_substitutions(self):
        return {}

    @memoized_property
    def sequencing_direction(self):
        ''' If editing strategy provides a sequencing direction, use that.
        Otherwise, return the strand with more aligned length
        across parsimonious target alignments.
        '''

        if self.editing_strategy.sequencing_direction is not None:
            sequencing_direction = self.editing_strategy.sequencing_direction

            if self.flipped:
                sequencing_direction = sam.opposite_strand[sequencing_direction]

        else:
            target_als = interval.make_parsimonious(self.initial_target_alignments)
            grouped = utilities.group_by(target_als, key=lambda al: sam.get_strand(al), sort=True)

            total_length_by_strand = {strand: sum(al.query_alignment_length for al in als) for strand, als in grouped}

            sequencing_direction = max(total_length_by_strand, key=total_length_by_strand.get)

        return sequencing_direction

    @memoized_property
    def target_side_to_read_side(self):
        if self.sequencing_direction == '+' or self.sequencing_direction is None:
            target_to_read = {
                5: 'left',
                3: 'right',
            }

        elif self.sequencing_direction == '-':
            target_to_read = {
                5: 'right',
                3: 'left',
            }

        else:
            raise ValueError(self.sequencing_direction)
        
        return target_to_read

    @memoized_property
    def read_side_to_target_side(self):
        return utilities.reverse_dictionary(self.target_side_to_read_side)

    @memoized_property
    def primers_by_side_of_read(self):
        by_side = {
            self.target_side_to_read_side[target_side]: primer
            for target_side, primer in self.editing_strategy.primers_by_side_of_target.items()
        }

        return by_side

    def overlaps_primer(self, alignment, side, by='read', require_correct_orientation=False):
        ''' Note that sequencing_direction is a dependency if require_correct_orientation == True
        or if by == 'read'.
        '''

        if by == 'read':
            primers = self.primers_by_side_of_read
        elif by == 'target':
            primers = self.editing_strategy.primers_by_side_of_target
        else:
            raise ValueError

        primer = primers.get(side)

        if primer is None:
            overlaps = False
        else:
            # Primers are annotated on the strand they anneal to, so don't require strand match here.
            overlaps = sam.overlaps_feature(alignment, primer, require_same_strand=False)

        if require_correct_orientation:
            satisifes_orientation_requirement = sam.get_strand(alignment) == self.sequencing_direction 
        else:
            satisifes_orientation_requirement = True

        return overlaps and satisifes_orientation_requirement

    @memoized_property
    def all_primer_alignments(self):
        ''' Dictionary of 
        {
            side_of_target: all target alignments that overlap the primer on that side
            for side_of_target in [5, 3]
        }

        If multiple alignments share the feature, take the longest.
        '''

        als = {}

        for side_of_target in [5, 3]:
            side_als = [al for al in self.target_alignments if self.overlaps_primer(al, side_of_target, by='target')]

            if len(side_als) > 0:
                # Partition into equivalence classes of shared feature.
                primer_name = self.editing_strategy.primers_by_side_of_target[side_of_target].ID

                def relation(first_al, second_al):
                    return self.share_feature(first_al, primer_name, second_al, primer_name)

                eq_classes = utilities.equivalence_classes(side_als, relation)

                side_als = [max(als, key=lambda al: al.query_alignment_length) for als in eq_classes]

            als[side_of_target] = side_als

        return als

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
    
    def extend_primer_alignment(self, al):
        ''' If an edge of a target alignment falls in a primer and isn't at the edge of the read,
        try to extend repeatedly towards the edge of the amplicon.
        '''

        strat = self.editing_strategy

        if al.reference_name == strat.target:
            target_seq_bytes = strat.reference_sequence_bytes[strat.target]

            ref_edges = sam.reference_edges(al)
            query_interval = interval.get_covered(al)
            query_edges = {
                'left': query_interval.start,
                'right': query_interval.end,
            }

            interior = self.whole_read_minus_edges(1)

            for side, ref_edge in ref_edges.items():
                for primer_name, primer_interval in strat.primer_intervals.items():
                    if query_edges[side] in interior and ref_edge in primer_interval:

                        if primer_name == strat.primers_by_side_of_target[5].ID:
                            kwargs = dict(extend_after=False)
                        elif primer_name == strat.primers_by_side_of_target[3].ID: 
                            kwargs = dict(extend_before=False)
                        else:
                            raise ValueError

                        al = sw.extend_repeatedly(al, target_seq_bytes, minimum_gain=2, **kwargs)

        return al

    @memoized_property
    def target_flanking_alignments(self):
        ''' Target alignments that flank the region of the read containing
        any potential editing.

        Dependencies:
            refined_target_alignments: split at indels, then extended 
            sequencing_direction: predefined for Illumina, inferred for long read.
        '''

        flanking_alignments = {
            'left': [],
            'right': [],
        }

        relevant_als = [al for al in self.refined_target_alignments if sam.get_strand(al) == self.sequencing_direction]

        if len(self.editing_strategy.primers) == 2:
            for al in relevant_als:
                covered = interval.get_covered(al)

                if covered.total_length >= 10:
                    if covered.start <= 5 or (self.overlaps_primer(al, 'left') and covered.start < 25):
                        flanking_alignments['left'].append((al.query_alignment_length, al))
                    
                    if covered.end >= len(self.seq) - 1 - 5 or self.overlaps_primer(al, 'right'):
                        flanking_alignments['right'].append((al.query_alignment_length, al))

        else:
            if self.platform in {'nanopore', 'pacbio'}:
                min_relevant_length = 500
            else:
                min_relevant_length = 0

            # Note: refined_target_alignments is already cropped to a relevant_window on both sides
            # For each side, annotate each al with the length of its overlap with that side. 

            cuts_interval = self.editing_strategy.around_or_between_cuts(0)

            flanking_intervals = {
                self.target_side_to_read_side[5]: interval.Interval(0, cuts_interval.start),
                self.target_side_to_read_side[3]: interval.Interval(cuts_interval.end + 1, len(self.editing_strategy.target_sequence))
            }

            for side, flanking_interval in flanking_intervals.items():
                for al in relevant_als:
                    cropped_al = sam.crop_al_to_ref_int(al, flanking_interval.start, flanking_interval.end)
                    if cropped_al is not None and cropped_al.query_alignment_length >= min_relevant_length:
                        flanking_alignments[side].append((cropped_al.query_alignment_length, al))

        # TODO: require these to be concordantly placed on the read. If they aren't, only keep the longest one.
        for side in flanking_alignments:
            if len(flanking_alignments[side]) == 0:
                flanking_alignments[side] = None

            else:
                _, al = max(flanking_alignments[side], key=lambda t: t[0])
                flanking_alignments[side] = al

        return flanking_alignments

    @memoized_property
    def target_flanking_intervals(self):
        edges = {
            'read': {
                side: interval.get_covered(self.target_flanking_alignments[side])
                for side in ['left', 'right']
            },
            'ref': {
                side: interval.get_covered_on_ref(self.target_flanking_alignments[side])
                for side in ['left', 'right']
            },
        }

        return edges

    @memoized_property
    def length_change(self):
        ''' How far apart are the outside edges of target flanking alignments on the read
        compared to the target? 
        Note: doesn't account for primer indels.
        '''

        if self.has_target_flanking_alignments_on_both_sides:
            read_distance = self.target_flanking_intervals['read']['right'].start - self.target_flanking_intervals['read']['left'].end + 1
            if self.sequencing_direction == '+':
                ref_distance = self.target_flanking_intervals['ref']['right'].start - self.target_flanking_intervals['ref']['left'].end + 1
            else:
                ref_distance = self.target_flanking_intervals['ref']['left'].start - self.target_flanking_intervals['ref']['right'].end + 1

            length_change = read_distance - ref_distance

        else:
            length_change = None

        return length_change

    @memoized_property
    def target_flanking_alignments_by_target_side(self):
        by_target_side = {
            target_side: self.target_flanking_alignments[self.target_side_to_read_side[target_side]]
            for target_side in [5, 3]
        }

        return by_target_side

    @memoized_property
    def target_flanking_alignments_list(self):
        return [al for al in self.target_flanking_alignments.values() if al is not None]

    @memoized_property
    def has_any_target_flanking_alignment(self):
        return len(self.target_flanking_alignments_list) > 0

    @memoized_property
    def has_target_flanking_alignments_on_both_sides(self):
        return len(self.target_flanking_alignments_list) == 2

    @memoized_property
    def cropped_flanking_primer_alignments(self):
        primer_alignments = {}

        for side, primer in self.primers_by_side_of_read.items():
            al = self.target_flanking_alignments[side]
            primer_alignments[side] = sam.crop_al_to_feature(al, primer)

        return primer_alignments

    @memoized_property
    def covered_by_primers(self):
        return interval.get_disjoint_covered(self.cropped_flanking_primer_alignments.values())

    @memoized_property
    def between_primers(self):
        ''' More complicated than function name suggests. If there are primer
        alignments, returns the query interval between but not covered by them to enable ignoring
        the region outside of them, which is likely to be the result of incorrect trimming.

        If there are no primers, uses target flanking alignments.
        '''
        if len(self.editing_strategy.primers) == 2:
            primer_als = self.cropped_flanking_primer_alignments

            if self.covered_by_primers.is_empty:
                between_primers = self.whole_read

            elif primer_als['left'] and not primer_als['right']:
                between_primers = interval.Interval(self.covered_by_primers.end + 1, self.whole_read.end)

            elif primer_als['right'] and not primer_als['left']:
                between_primers = interval.Interval(self.whole_read.start, self.covered_by_primers.start - 1)

            else:
                between_primers = interval.Interval(self.covered_by_primers.start, self.covered_by_primers.end) - self.covered_by_primers 

            if between_primers.is_empty:
                between_primers = interval.Interval.empty()

        else:
            if self.has_target_flanking_alignments_on_both_sides:
                between_primers = interval.Interval(self.target_flanking_intervals['read']['left'].start, self.target_flanking_intervals['read']['right'].end)
            else:
                between_primers = interval.Interval.empty()

        return between_primers

    @memoized_property
    def between_primers_inclusive(self):
        return self.covered_by_primers | self.between_primers

    @memoized_property
    def not_covered_by_target_flanking_alignments(self):
        als = self.target_flanking_alignments_list
        not_covered = self.between_primers - interval.get_disjoint_covered(als)
        return not_covered

    @memoized_property
    def non_primer_nts(self):
        return self.between_primers.total_length

    @memoized_property
    def target_nts_past_primer(self):
        target_nts_past_primer = {}

        for side in ['left', 'right']:
            target_past_primer = interval.get_covered(self.target_flanking_alignments[side]) - interval.get_covered(self.cropped_flanking_primer_alignments[side])
            target_nts_past_primer[side] = target_past_primer.total_length 

        return target_nts_past_primer
    
    @memoized_property
    def min_relevant_length(self):
        min_relevant_length = self.editing_strategy.min_relevant_length

        if min_relevant_length is None:
            if self.editing_strategy.combined_primer_length is not None:
                min_relevant_length = self.editing_strategy.combined_primer_length + 10
            else:
                min_relevant_length = 0

        return min_relevant_length

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
            of the read not covered, and a single alignment to some other
            source covers the entire read with minimal edit distance.
         
         - read starts and ends with alignments to the expected primers, these
           alignments are spanned by a single alignment to some other source, and
           the inferred amplicon length is more than 20 nts different from the expected
           WT amplicon. This covers the case where an amplififcation product has enough
           homology around the primer for additional sequence to align to the target. 
        
        '''
        results = {}

        valid = False

        need_to_cover = self.whole_read_minus_edges(2) & self.between_primers

        results['covering_als'] = []

        if len(self.seq) <= self.min_relevant_length or self.non_primer_nts <= 10:
            valid = True

        if need_to_cover.total_length > 0 and self.cropped_flanking_primer_alignments.get('left') is not None:

            if self.target_nts_past_primer['left'] <= 10 and self.target_nts_past_primer['right'] <= 10:
                # Exclude phiX reads, which can rarely have spurious alignments to the forward primer
                # close to the start of the read that overlap the forward primer.
                relevant_alignments = self.supplemental_alignments + self.extra_alignments
                relevant_alignments = [al for al in relevant_alignments if 'phiX' not in al.reference_name]

                for al in relevant_alignments:
                    covered_by_al = interval.get_covered(al)
                    if (need_to_cover - covered_by_al).total_length == 0:
                        results['covering_als'].append(al)

            else:
                target_als = [al for al in self.primary_alignments if al.reference_name == self.editing_strategy.target]
                not_covered_by_any_target_als = need_to_cover - interval.get_disjoint_covered(target_als)

                has_substantial_not_covered = not_covered_by_any_target_als.total_length >= 100
                has_substantial_length_discrepancy = (
                    self.inferred_amplicon_length is not None and 
                    abs(self.inferred_amplicon_length - self.editing_strategy.amplicon_length) >= 20 and
                    self.cropped_flanking_primer_alignments.get('right') is not None
                )

                if has_substantial_not_covered or has_substantial_length_discrepancy:
                    ref_seqs = {**self.editing_strategy.reference_sequences}

                    # Exclude phiX reads, which can rarely have spurious alignments to the forward primer
                    # close to the start of the read that overlap the forward primer.
                    relevant_alignments = self.supplemental_alignments
                    relevant_alignments = [al for al in relevant_alignments if 'phiX' not in al.reference_name]

                    for al in relevant_alignments:
                        covered_by_al = interval.get_covered(al)
                        if (need_to_cover - covered_by_al).total_length == 0:
                            cropped_al = sam.crop_al_to_query_int(al, self.between_primers.start, self.between_primers.end)
                            total_edits = sum(knock_knock.architecture.edit_positions(cropped_al, ref_seqs, use_deletion_length=True))
                            if total_edits <= 5:
                                results['covering_als'].append(al)

            if len(results['covering_als']) > 0:
                if not any('phiX' in al.reference_name for al in results['covering_als']):
                    valid = True

        if not valid:
            results = None

        return results

    def register_nonspecific_amplification(self):
        results = self.nonspecific_amplification

        self.category = 'nonspecific amplification'

        if self.non_primer_nts <= 2:
            self.subcategory = 'primer dimer'
            self.relevant_alignments = self.target_flanking_alignments_list

        elif len(results['covering_als']) == 0:
            self.subcategory = 'short unknown'
            self.relevant_alignments = sam.make_noncontained(self.uncategorized_relevant_alignments)

        else:
            if any(al.reference_name in self.editing_strategy.pegRNA_names for al in results['covering_als']):
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

    @memoized_property
    def single_read_covering_target_alignment(self):
        need_to_cover = self.whole_read_minus_edges(2) & self.between_primers

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

    def extract_indels_from_alignments(self, als):
        strat = self.editing_strategy

        around_cut_interval = strat.around_cuts(5)

        primer_intervals = interval.make_disjoint(strat.primer_intervals.values())

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

    def interesting_and_uninteresting_indels(self, als):
        ''' For illumina data, uninteresting indels are entirely contained in a primer,
        or a single base deletion more than 5 nts from an expected cut.
        All other indels are interesting.

        For long read data, insertions and deletions less than 5 nts long and more than
        5 nts from an expected cut site are also uninteresting.
        '''

        indels = self.extract_indels_from_alignments(als)

        interesting = []
        uninteresting = []

        for indel, near_cut in indels:
            if near_cut:
                append_to = interesting

            else:
                if self.platform == 'illumina':
                    if indel.kind == 'D' and indel.length == 1:
                        append_to = uninteresting
                    else:
                        append_to = interesting

                elif self.platform in ['pacbio', 'ont', 'nanopore']:
                    if indel.length < 5:
                        append_to = uninteresting
                    else:
                        append_to = interesting

            append_to.append(indel)

        return interesting, uninteresting

    def summarize_mismatches_in_alignments(self, relevant_alignments):
        ''' TODO: this docstring is out of date.
        Record bases seen at programmed substitution positions relative to target +.
        '''
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

        elif len(self.editing_strategy.primers) == 0:
            return self.length_change

        elif (self.whole_read - self.covered_by_primers).total_length == 0:
            return len(self.seq)

        elif len(self.seq) <= 50:
            return len(self.seq)

        left_al = self.target_flanking_alignments['left']
        right_al = self.target_flanking_alignments['right']

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

            right_al_edge_in_target = sam.reference_edges(right_al)['right']

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
            inferred_length = None

        return inferred_length

    def q_to_feature_offset(self, al, feature_name, editing_strategy=None):
        ''' Returns dictionary of 
                {true query position: offset into feature relative to its strandedness
                 (i.e. from the start of + stranded and from the right of - stranded)
                }
        '''
        if al is None:
            return {}

        if editing_strategy is None:
            editing_strategy = self.editing_strategy

        feature = editing_strategy.features.get((al.reference_name, feature_name))
        if feature is None:
            return {}

        seq = editing_strategy.reference_sequences[al.reference_name]
        
        q_to_feature_offset = {}
        
        for q, read_b, ref_p, ref_b, qual in sam.aligned_tuples(al, seq):
            if q is not None and ref_p in feature.ref_p_to_offset:
                q_to_feature_offset[q] = feature.ref_p_to_offset[ref_p]
                
        return q_to_feature_offset

    def feature_offset_to_q(self, al, feature_name, editing_strategy=None):
        return utilities.reverse_dictionary(self.q_to_feature_offset(al, feature_name, editing_strategy=editing_strategy))

    def feature_query_interval(self, al, feature_name, editing_strategy=None):
        ''' Returns the query interval aligned to feature_name by al. '''
        qs = self.q_to_feature_offset(al, feature_name, editing_strategy=editing_strategy)
        if len(qs) == 0:
            return interval.Interval.empty()
        else:
            return interval.Interval(min(qs), max(qs))

    def share_feature(self, first_al, first_feature_name, second_al, second_feature_name):
        '''
        Returns True if any query position is aligned to equivalent positions in first_feature and second_feature
        by first_al and second_al.
        '''
        if first_al is None or second_al is None:
            return False
        
        first_q_to_offsets = self.q_to_feature_offset(first_al, first_feature_name)
        second_q_to_offsets = self.q_to_feature_offset(second_al, second_feature_name)
        
        share_any = any(second_q_to_offsets.get(q) == offset for q, offset in first_q_to_offsets.items())

        return share_any

    def are_mutually_extending_from_shared_feature(self,
                                                   left_al,
                                                   right_al,
                                                   feature_name,
                                                   contribution_test=lambda al: False,
                                                  ):
        strat = self.editing_strategy
        
        results = None

        if not self.share_feature(left_al, feature_name, right_al, feature_name):
            results = {
                'status': 'don\'t share feature',
                'alignments': {
                    'left': None,
                    'right': None,
                },
                'cropped_alignments': {
                    'left': None,
                    'right': None,
                },
            }

        else: 
            switch_results = sam.find_best_query_switch_after(left_al,
                                                              right_al,
                                                              reference_sequences=strat.reference_sequences,
                                                              tie_break=min,
                                                             )
                                                        
            # Does an optimal switch point occur somewhere in the shared feature?

            switch_interval = interval.Interval(min(switch_results['best_switch_points']), max(switch_results['best_switch_points']))
            
            left_feature_interval = self.feature_query_interval(left_al, feature_name)

            switch_in_shared = interval.are_overlapping(switch_interval, left_feature_interval)

            left_of_feature = interval.Interval(0, left_feature_interval.start - 1)

            # If as much query as possible is attributed to the right al, does the remaining left al
            # still explain part of the read to the left of the overlapping feature?

            cropped_left_al = sam.crop_al_to_query_int(left_al, 0, switch_interval.start)
            left_definite_contribution_past_overlap = interval.get_covered(cropped_left_al) & left_of_feature

            # If as much query as possible is attributed to the left al, does it extend outside of the
            # the overlapping feature?

            cropped_left_al = sam.crop_al_to_query_int(left_al, 0, switch_interval.end)

            right_feature_interval = self.feature_query_interval(right_al, feature_name)

            right_of_feature = interval.Interval(right_feature_interval.end + 1, self.whole_read.end)

            # Similarly, if as much query as possible is attributed to the left al, does the remaining right al
            # still explain part of the read to the right of the overlapping feature?

            cropped_right_al = sam.crop_al_to_query_int(right_al, switch_interval.end + 1, self.whole_read.end)
            right_definite_contribution_past_overlap = interval.get_covered(cropped_right_al) & right_of_feature

            # If as much query as possible is attributed to the right al, does it extend outside of the
            # overlapping feature?

            cropped_right_al = sam.crop_al_to_query_int(right_al, switch_interval.start + 1, self.whole_read.end)

            overlap_reaches_read_end = right_of_feature.is_empty

            # Heuristic: if an optimal switch point occurs in the shared feature,
            # any amount of sequence past the shared feature is enough.
            # If the optimal switch point is outside, require a longer amount. 
            # Motivation: for replacement edits in which pegRNA sequence is similar
            # to genomic sequence, we want to avoid identifying false positive
            # extension alignments that are actually just genomic sequence, while
            # still allowing the possibility of partial replacements that retain 
            # some genomic sequence after the transition.

            left_contributes = (switch_in_shared and left_definite_contribution_past_overlap.total_length > 0) or \
                               (left_definite_contribution_past_overlap.total_length >= 10) or \
                               contribution_test(left_al)

            right_contributes = right_definite_contribution_past_overlap or \
                                overlap_reaches_read_end or \
                                contribution_test(right_al)

            if left_contributes and right_contributes:
                status = 'definite'
            else:
                status = 'possible'

            results = {
                'status': status,
                'alignments': {
                    'left': left_al,
                    'right': right_al,
                },
                'cropped_alignments': {
                    'left': cropped_left_al,
                    'right': cropped_right_al,
                },
            }

        return results

    def extend_alignment_from_shared_feature(self,
                                             alignment_to_extend,
                                             feature_name_in_alignment,
                                             ref_to_search,
                                             feature_name_in_ref,
                                            ):
        ''' Generates the longest perfectly extended alignment to ref_to_search
        that pairs feature_name_in_alignment in alignment_to_extend with feature_name_in_ref.
        Motivation: if a potential transition occurs close to the end of a read or otherwise
        only involves a small amount of sequence past the transition, initial alignment
        generation may fail to identify a potentially relevant alignment.
        ''' 

        strat = self.editing_strategy

        feature_in_alignment = strat.features[alignment_to_extend.reference_name, feature_name_in_alignment]
        feature_al = sam.crop_al_to_feature(alignment_to_extend, feature_in_alignment)

        feature_in_ref = strat.features[ref_to_search, feature_name_in_ref]

        if feature_al is not None and not sam.contains_indel(feature_al):
            # Create a new alignment covering the feature on ref_to_search,
            # which will then be used as input to sw.extend_alignment.
            # This needs three things:
            #   - the query interval to be covered, which will be converted
            #     into lengths to soft clip before and after the alignment.
            #   - the reference interval to be covered, which will have its
            #     left-most value put into reference_start.
            #   - whether or not the alignment is reversed, which will be
            #     reflected in the relevant flag and by flipped seq, qual,
            #     and cigar.

            # Get the query interval covered.
            query_interval = self.feature_query_interval(feature_al, feature_name_in_alignment)

            # Get the ref interval covered.

            feature_offsets = self.q_to_feature_offset(feature_al, feature_name_in_alignment).values()
            ref_ps = [feature_in_ref.offset_to_ref_p[fo] for fo in feature_offsets]

            if len(ref_ps) == 0:
                raise ValueError
            else:
                ref_interval = interval.Interval(min(ref_ps), max(ref_ps))

            # Figure out the strand.
            if feature_in_ref.strand == feature_in_alignment.strand:
                is_reverse = feature_al.is_reverse
            else:
                is_reverse = not feature_al.is_reverse

            al = pysam.AlignedSegment(strat.header)

            al.query_sequence = self.seq
            al.query_qualities = self.qual

            soft_clip_before = query_interval.start
            soft_clip_after = len(self.seq) - 1 - query_interval.end

            al.cigar = [
                (sam.BAM_CSOFT_CLIP, soft_clip_before),
                (sam.BAM_CMATCH, feature_al.query_alignment_length),
                (sam.BAM_CSOFT_CLIP, soft_clip_after),
            ]

            if is_reverse:
                # Need to extract query_qualities before overwriting query_sequence.
                flipped_query_qualities = al.query_qualities[::-1]
                al.query_sequence = utilities.reverse_complement(al.query_sequence)
                al.query_qualities = flipped_query_qualities
                al.is_reverse = True
                al.cigar = al.cigar[::-1]

            al.reference_name = ref_to_search
            al.query_name = self.read.name
            al.next_reference_id = -1

            al.reference_start = ref_interval.start

            extended_al = sw.extend_alignment(al, strat.reference_sequence_bytes[ref_to_search])
        else:
            extended_al = None

        return extended_al

    def find_extending_alignment(self, alignment_to_extend, from_side, candidate_alignments, shared_feature, require_definite=True):

        extension_side = opposite_side(from_side)
        
        by_status = defaultdict(list)
            
        for candidate_al in candidate_alignments:
            if from_side == 'left':
                left_al = alignment_to_extend
                right_al = candidate_al
            elif from_side == 'right':
                left_al = candidate_al
                right_al = alignment_to_extend
            else:
                raise ValueError(from_side)
                
            extension_results = self.are_mutually_extending_from_shared_feature(left_al,
                                                                                right_al,
                                                                                shared_feature,
                                                                               )
            by_status[extension_results['status']].append(extension_results)
            
        eligible = by_status['definite']

        if not require_definite:
            eligible.extend(by_status['possible'])

        if len(eligible) > 0:
            best_results = max(eligible, key=lambda d: self.length_minus_edits(d['alignments'][extension_side]))
            
            extension_al = best_results['alignments'][extension_side]
            cropped_extension_al = best_results['cropped_alignments'][extension_side]
            
            croppped_original_al = best_results['cropped_alignments'][from_side]
        
        else:
            extension_al, cropped_extension_al, croppped_original_al = None, None, None

        return extension_al, cropped_extension_al, croppped_original_al

    @memoized_with_kwargs
    def extension_chains(self, *, require_definite=True):
        extension_chains = {}

        for kind, (link_specifications, last_al_to_description) in self.extension_chain_link_specifications.items():
            extension_chains[kind] = {}

            for side in ['left', 'right']:
                if (side == 'left' and self.sequencing_direction == '+') or (side == 'right' and self.sequencing_direction == '-'):
                    order = 'forward'
                else:
                    order = 'reverse'

                chain = ExtensionChain(link_specifications,
                                       last_al_to_description,
                                       order,
                                       side,
                                       require_definite=require_definite,
                                      ) 

                chain.build(self.target_flanking_alignments[side], 'first target', self)
                    
                extension_chains[kind][side] = chain

        return extension_chains

    @memoized_with_kwargs
    def extension_chain_possible_covers(self, *, require_definite=True):
        all_possible_covers = {}

        for kind, chains in self.extension_chains(require_definite=require_definite).items():
            possible_covers = set()

            # Check whether any members of an extension chain on one side are not
            # necessary to make it to the other chain. (Warning: could imagine a
            # scenario in which it would be possible to remove from either the
            # left or right chain.)

            name_orders = {side: ['none'] + chains[side].name_order for side in ['left', 'right']}

            for left_key in name_orders['left']:
                if left_key in chains['left'].alignments:
                    left_al = chains['left'].alignments[left_key]

                    for right_key in name_orders['right']:
                        if right_key in chains['right'].alignments:
                            right_al = chains['right'].alignments[right_key]

                            covered_left = chains['left'].query_covered_incremental[left_key]
                            covered_right = chains['right'].query_covered_incremental[right_key]

                            # Check if left and right overlap or abut each other.
                            if covered_left.end >= covered_right.start - 1:

                                # include_overlap=False to avoid double counting mismatch-containing region
                                cropped_als = sam.crop_to_best_switch_point(left_al,
                                                                            right_al,
                                                                            self.editing_strategy.reference_sequences,
                                                                            include_overlap=False,
                                                                           )
                            else:
                                cropped_als = {'left': left_al, 'right': right_al}

                            cropped_mismatches = {
                                side: chains[side].mismatches.copy()
                                for side in ['left', 'right']
                            }

                            total_mismatches = {}

                            for side, last_key in [('left', left_key), ('right', right_key)]:
                                cropped_mismatches[side][last_key] = self.edits(cropped_als[side])

                                last_key_i = name_orders[side].index(last_key)
                                total_mismatches[side] = sum(cropped_mismatches[side][key] for key in name_orders[side][:last_key_i + 1])

                            gap = max(0, covered_right.start - covered_left.end - 1)
                            mismatches_and_gap = total_mismatches['left'] + total_mismatches['right'] + gap
                            
                            possible_covers.add(('two chains', left_key, right_key, mismatches_and_gap))
            
            if chains['left'].query_covered == chains['right'].query_covered:
                last_parsimonious_key = {}
                total_mismatches = {}

                for side, chain in chains.items():
                    maximal_covers = [k for k, covered in chain.query_covered_incremental.items() if covered == chain.query_covered]
                    last_key = min(maximal_covers, key=name_orders[side].index, default='none')
                    last_key_i = name_orders[side].index(last_key)
                    last_parsimonious_key[side] = last_key
                    total_mismatches[side] = sum(chain.mismatches[key] for key in name_orders[side][:last_key_i + 1])

                possible_covers.add(('single chain', last_parsimonious_key['left'], last_parsimonious_key['right'], total_mismatches['left']))

            if possible_covers:
                single_exists = any(chain_number == 'single chain' for chain_number, *rest in possible_covers)

                if single_exists:
                    best_chain_number = 'single chain'
                else:
                    best_chain_number = 'two chains'

                min_mismatches = min(mismatches for chain_number, _, _, mismatches in possible_covers if chain_number == best_chain_number)

                min_mismatch_covers = {
                    (chain_number, left_key, right_key, mismatches)
                    for chain_number, left_key, right_key, mismatches in possible_covers
                    if chain_number == best_chain_number and mismatches == min_mismatches
                }

                best_cover = self.reconcile_function(min_mismatch_covers, key=lambda tuple_: (name_orders['left'].index(tuple_[1]), name_orders['right'].index(tuple_[2])))
                rejected_covers = {cover for cover in possible_covers if cover != best_cover}

                all_possible_covers[kind] = (best_cover, rejected_covers)

        return all_possible_covers

    @memoized_with_kwargs
    def reconciled_extension_chains(self, *, require_definite=True):
        ''' Note: reconciling modifies chains in place. '''

        possible_covers = self.extension_chain_possible_covers(require_definite=require_definite)

        for kind in self.extension_chain_link_specifications:
            chains = self.extension_chains(require_definite=require_definite)[kind]

            name_orders = {side: ['none'] + chains[side].name_order for side in ['left', 'right']}

            last_parsimonious_key = {}

            if kind in possible_covers:
                (_, last_parsimonious_key['left'], last_parsimonious_key['right'], _), _ = possible_covers[kind]
            else:
                for side, chain in chains.items():
                    maximal_covers = [k for k, covered in chain.query_covered_incremental.items() if covered == chain.query_covered]
                    last_parsimonious_key[side] = min(maximal_covers, key=name_orders[side].index, default='none')

            for side in ['left', 'right']:
                key = last_parsimonious_key[side]

                chains[side].description = chains[side].last_al_to_description[key]

                last_index = name_orders[side].index(key)
                chains[side].parsimonious_alignments = [al for key, al in chains[side].alignments.items() if name_orders[side].index(key) <= last_index]

                chains[side].query_covered = chains[side].query_covered_incremental[key]

            last_als = {
                side: chains[side].alignments[last_parsimonious_key[side]]
                if last_parsimonious_key[side] != 'none' else None
                for side in ['left', 'right']
            }

            cropped_als = sam.crop_to_best_switch_point(last_als['left'], last_als['right'], self.editing_strategy.reference_sequences)

            for side in ['left', 'right']:
                chains[side].cropped_last_al = cropped_als[side]

            # If one chain is absent and the other chain covers the whole read
            # (except possibly 2 nts at either edge), classify the missing side
            # as 'not seen'.

            between_primers_minus_edges = self.between_primers & self.whole_read_minus_edges(2)

            if between_primers_minus_edges in chains['left'].query_covered:
                if chains['right'].description == 'no target':
                    chains['right'].description = 'not seen'

            if between_primers_minus_edges in chains['right'].query_covered:
                if chains['left'].description == 'no target':
                    chains['left'].description = 'not seen'

        return self.extension_chains(require_definite=require_definite)

    @property
    def reconcile_function(self):
        return min

    @memoized_property
    def extension_chain_alignments(self):
        return {
            kind: self.reconciled_extension_chains()[kind]['left'].alignments_list + self.reconciled_extension_chains()[kind]['right'].alignments_list
            for kind in self.extension_chain_link_specifications
        }

    @memoized_property
    def not_covered_by_extension_chains(self):
        not_covered_by_extension_chains = {}

        for kind in self.extension_chain_link_specifications:
            chains = self.reconciled_extension_chains()[kind]

            left_covered = chains['left'].query_covered
            right_covered = chains['right'].query_covered

            combined_covered = left_covered | right_covered
            not_covered = self.between_primers - combined_covered

            # Allow failure to explain the last few nts of the read.
            not_covered = not_covered & self.whole_read_minus_edges(2)

            not_covered_by_extension_chains[kind] = not_covered

        return not_covered_by_extension_chains

    def edits(self, al):
        if al is None:
            edits = 0
        else:
            edits = edit_positions(al, self.editing_strategy.reference_sequences, programmed_substitutions=self.programmed_substitutions).sum()

        return edits

    def length_minus_edits(self, al):
        return al.query_alignment_length - self.edits(al)

    def als_with_min_edits(self, als):
        decorated_als = sorted([(self.edits(al), al) for al in als], key=lambda t: t[0])
        
        if len(decorated_als) > 0:
            min_edits, _ = decorated_als[0]
        else:
            min_edits = None

        return [al for edits, al in decorated_als if edits == min_edits]

    def als_with_max_length(self, als):
        decorated_als = sorted([(al.query_alignment_length, al) for al in als], key=lambda t: t[0], reverse=True)
        
        if len(decorated_als) > 0:
            max_length, _ = decorated_als[0]
        else:
            max_length = None

        return [al for length, al in decorated_als if length == max_length]

    def als_with_max_length_minus_edits(self, als):
        decorated_als = sorted([(self.length_minus_edits(al), al) for al in als], key=lambda t: t[0], reverse=True)
        
        if len(decorated_als) > 0:
            max_length_minus_edits, _ = decorated_als[0]
        else:
            max_length_minus_edits = None

        return [al for length_minus_edits, al in decorated_als if length_minus_edits == max_length_minus_edits]

    def split_and_extend_alignments(self, als):
        all_split_als = []

        for al in als:
            split_als = self.comprehensively_split_alignment(al)

            seq_bytes = self.editing_strategy.reference_sequence_bytes[al.reference_name]

            extended_als = []
            
            for split_al in split_als:
                extended_al = sw.extend_alignment(split_al, seq_bytes)
                extended_al = self.extend_primer_alignment(extended_al)
                extended_als.append(extended_al)

            all_split_als.extend(extended_als)
        
        return sam.make_nonredundant(all_split_als)

    def realign_edge_to_primer(self, read_side):
        strat = self.editing_strategy

        target_side = self.read_side_to_target_side[read_side]

        primer = self.editing_strategy.primers_by_side_of_target.get(target_side)

        if self.seq is None or primer is None:
            return None

        buffer_length = 5

        if target_side == 5:
            target_interval = interval.Interval(primer.start, primer.end + buffer_length)
        elif target_side == 3:
            target_interval = interval.Interval(primer.start - buffer_length, primer.end)
        else:
            raise ValueError

        if read_side == 'left':
            read_interval = interval.Interval(0, len(primer) + buffer_length)
        elif read_side == 'right':
            read_interval = interval.Interval(len(self.seq) - len(primer) - buffer_length, len(self.seq))
        else:
            raise ValueError

        ref_intervals = {strat.target: target_interval}

        als = sw.align_read(self.read,
                            [(strat.target, strat.target_sequence)],
                            5,
                            strat.header,
                            max_alignments_per_target=1,
                            min_score_ratio=0,
                            read_interval=read_interval,
                            ref_intervals=ref_intervals,
                            )

        edge_al = None

        if len(als) > 0:
            al = als[0]

            if read_side == 'left':
                primer_query_interval = interval.Interval(0, len(primer) - 1)
            elif read_side == 'right':
                primer_query_interval = interval.Interval(len(self.seq) - len(primer), np.inf)
            else:
                raise ValueError

            edits_in_primer = sam.edit_distance_in_query_interval(al, primer_query_interval, ref_seq=strat.target_sequence)

            if edits_in_primer <= 5:
                edge_al = al

        if edge_al is not None:
            seq_bytes = strat.reference_sequence_bytes[edge_al.reference_name]
            edge_al = sw.extend_alignment(edge_al, seq_bytes)
            
        return edge_al

    def comprehensively_split_alignment(self, al):
        ''' It is easier to reason about alignments if any that contain long
        insertions, long deletions, or clusters of many edits are split into
        multiple alignments.
        '''

        strat = self.editing_strategy

        split_als = []

        if self.platform == 'illumina':
            for split_1 in split_at_edit_clusters(al, strat.reference_sequences, programmed_substitutions=self.programmed_substitutions):
                for split_2 in sam.split_at_deletions(split_1, 1):
                    for split_3 in sam.split_at_large_insertions(split_2, 1):
                        cropped_al = crop_terminal_mismatches(split_3, strat.reference_sequences)
                        if cropped_al is not None and cropped_al.query_alignment_length >= 5:
                            split_als.append(cropped_al)

        elif self.platform in ['pacbio', 'ont', 'nanopore']:
            # Empirically, for long read data, it is hard to find a threshold
            # for number of edits within a window that doesn't produce a lot
            # of false positive splits, so don't try to split at edit clusters.

            indel_length_near_cuts = 3
            indel_length_far_from_cuts = 12

            if al.reference_name == strat.target:
                # First split at short indels close to expected cuts.
                exempt = strat.not_around_cuts(20)
                for split_1 in sam.split_at_deletions(al, indel_length_near_cuts, exempt_if_overlaps=exempt):
                    for split_2 in sam.split_at_large_insertions(split_1, indel_length_near_cuts, exempt_if_overlaps=exempt):
                        # Then at longer indels anywhere.
                        for split_3 in sam.split_at_deletions(split_2, indel_length_far_from_cuts):
                            for split_4 in sam.split_at_large_insertions(split_3, indel_length_far_from_cuts):
                                split_als.append(split_4)

            else:
                for split_1 in sam.split_at_deletions(al, indel_length_far_from_cuts):
                    for split_2 in sam.split_at_large_insertions(split_1, indel_length_far_from_cuts):
                        split_als.append(split_2)

        else:
            raise ValueError(self.platform)

        return split_als

    @memoized_property
    def supplemental_alignments(self):
        supp_als = [
            al for al in self.alignments
            if (not al.is_unmapped) and al.reference_name not in self.primary_ref_names
        ]

        if self.platform in ['pacbio', 'ont', 'nanopore']:
            ins_size_to_split_at = 10
        else:
            ins_size_to_split_at = 2

        split_als = []
        for supp_al in supp_als:
            split_als.extend(sam.split_at_large_insertions(supp_al, ins_size_to_split_at))
        
        few_mismatches = []
        # Alignments generated with STAR will have MD tags, but alignments
        # generated with blastn or minimap2 will not. 
        for al in split_als:
            ref_seq = self.editing_strategy.reference_sequences.get(al.reference_name)
            if al.has_tag('MD') or ref_seq is not None:
                if sam.total_edit_distance(al, ref_seq=ref_seq) / al.query_alignment_length < 0.2:
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

class NoOverlapPairCategorizer(Categorizer):
    def __init__(self, alignments, editing_strategy):
        self.alignments = alignments
        self.editing_strategy = editing_strategy
        self._flipped = False

        self.architecture = {
            'R1': type(self).individual_architecture_class(alignments['R1'], editing_strategy),
            'R2': type(self).individual_architecture_class(alignments['R2'], editing_strategy, flipped=True),
        }

        self._inferred_amplicon_length = -1

        self.details = knock_knock.outcome.Details()

def max_del_nearby(alignment, ref_pos, window):
    ref_pos_to_block = sam.get_ref_pos_to_block(alignment)
    nearby = range(ref_pos - window, ref_pos + window)
    blocks = [ref_pos_to_block.get(r, (-1, -1, -1)) for r in nearby]
    dels = [l for k, l, s in blocks if k == sam.BAM_CDEL]
    if dels:
        max_del = max(dels)
    else:
        max_del = 0

    return max_del

def max_ins_nearby(alignment, ref_pos, window):
    nearby = sam.crop_al_to_ref_int(alignment, ref_pos - window, ref_pos + window)
    max_ins = sam.max_block_length(nearby, {sam.BAM_CINS})
    return max_ins

def max_indel_nearby(alignment, ref_pos, window):
    max_del = max_del_nearby(alignment, ref_pos, window)
    max_ins = max_ins_nearby(alignment, ref_pos, window)
    return max(max_del, max_ins)

def get_mismatch_info(alignment, reference_sequences, programmed_substitutions=None):
    mismatches = []

    if alignment is None:
        return mismatches

    if programmed_substitutions is None:
        programmed_substitutions = {}

    if alignment.reference_name in programmed_substitutions:
        programmed_substitutions = programmed_substitutions[alignment.reference_name]

    tuples = []
    if reference_sequences.get(alignment.reference_name) is None:
        for read_p, ref_p, ref_b in alignment.get_aligned_pairs(with_seq=True):
            if read_p != None and ref_p != None:
                read_b = alignment.query_sequence[read_p]
                tuples.append((read_p, read_b, ref_p, ref_b))

    else:
        reference = reference_sequences[alignment.reference_name]
        for read_p, ref_p in alignment.get_aligned_pairs():
            if read_p != None and ref_p != None:
                read_b = alignment.query_sequence[read_p]
                ref_b = reference[ref_p]
                
                tuples.append((read_p, read_b, ref_p, ref_b))

    for read_p, read_b, ref_p, ref_b in tuples:
        read_b = read_b.upper()
        ref_b = ref_b.upper()

        if read_b != ref_b:
            if ref_p in programmed_substitutions and programmed_substitutions[ref_p] == read_b:
                continue

            true_read_p = sam.true_query_position(read_p, alignment)
            q = alignment.query_qualities[read_p]

            if alignment.is_reverse:
                read_b = utilities.reverse_complement(read_b)
                ref_b = utilities.reverse_complement(ref_b)

            mismatches.append((true_read_p, read_b, ref_p, ref_b, q))

    return mismatches

def get_indel_info(alignment):
    indels = []

    for i, (kind, length) in enumerate(alignment.cigar):
        if kind == sam.BAM_CDEL or kind == sam.BAM_CREF_SKIP:
            if kind == sam.BAM_CDEL:
                name = 'deletion'
            else:
                name = 'splicing'

            nucs_before = sam.total_read_nucs(alignment.cigar[:i])
            centered_at = np.mean([sam.true_query_position(p, alignment) for p in [nucs_before - 1, nucs_before]])

            indels.append((name, (centered_at, length)))

        elif kind == sam.BAM_CINS:
            # Note: edges are both inclusive.
            first_edge = sam.total_read_nucs(alignment.cigar[:i])
            second_edge = first_edge + length - 1
            starts_at, ends_at = sorted(sam.true_query_position(p, alignment) for p in [first_edge, second_edge])
            indels.append(('insertion', (starts_at, ends_at)))

    return indels

def edit_positions(al, reference_sequences, use_deletion_length=False, programmed_substitutions=None):
    ''' Returns an array of length equal to the query that marks all positions
    at which the query doesn't match the reference.
    ''' 

    bad_read_ps = np.zeros(al.query_length)
    
    for read_p, *rest in get_mismatch_info(al, reference_sequences, programmed_substitutions=programmed_substitutions):
        bad_read_ps[read_p] += 1
        
    for indel_type, indel_info in get_indel_info(al):
        if indel_type == 'deletion':
            centered_at, length = indel_info
            for offset in [-0.5, 0.5]:
                if use_deletion_length:
                    to_add = length / 2
                else:
                    to_add = 1
                bad_read_ps[int(centered_at + offset)] += to_add

        elif indel_type == 'insertion':
            starts_at, ends_at = indel_info
            for read_p in range(starts_at, ends_at + 1):
                bad_read_ps[read_p] += 1
               
    return bad_read_ps

def split_at_edit_clusters(al, reference_sequences, num_edits=5, window_size=11, programmed_substitutions=None):
    ''' Identify read locations at which there are at least num_edits edits in a windows_size nt window. 
    Excise outwards from any such location until reaching a stretch of 5 exact matches.
    Remove the excised region, producing new cropped alignments.
    '''
    split_als = []
    
    bad_read_ps = edit_positions(al, reference_sequences, programmed_substitutions=programmed_substitutions)
    rolling_sums = pd.Series(bad_read_ps).rolling(window=window_size, center=True, min_periods=1).sum()

    argmax = rolling_sums.idxmax()

    if rolling_sums[argmax] < num_edits:
        split_als.append(al)
    else:
        last_read_p_in_before = None
    
        window_edge = argmax
        for window_edge in range(argmax, -1, -1):
            errors_in_window_before = sum(bad_read_ps[window_edge + 1 - 5:window_edge + 1])
            if errors_in_window_before == 0:
                last_read_p_in_before = window_edge
                break
            
        if last_read_p_in_before is not None:
            cropped_before = sam.crop_al_to_query_int(al, 0, last_read_p_in_before)
            if cropped_before is not None:
                split_als.extend(split_at_edit_clusters(cropped_before, reference_sequences, programmed_substitutions=programmed_substitutions))

        first_read_p_in_after = None
            
        window_edge = argmax
        for window_edge in range(argmax, al.query_length):
            errors_in_window_after = sum(bad_read_ps[window_edge:window_edge + 5])
            if errors_in_window_after == 0:
                first_read_p_in_after = window_edge
                break

        if first_read_p_in_after is not None:
            cropped_after = sam.crop_al_to_query_int(al, first_read_p_in_after, np.inf)
            if cropped_after is not None:
                split_als.extend(split_at_edit_clusters(cropped_after, reference_sequences, programmed_substitutions=programmed_substitutions))

    split_als = [al for al in split_als if not al.is_unmapped]

    return split_als

def crop_terminal_mismatches(al, reference_sequences):
    ''' Remove all consecutive mismatches from the start and end of an alignment. '''
    covered = interval.get_covered(al)

    mismatch_ps = {p for p, *rest in get_mismatch_info(al, reference_sequences)}

    first = covered.start
    last = covered.end

    while first in mismatch_ps:
        first += 1
        
    while last in mismatch_ps:
        last -= 1

    cropped_al = sam.crop_al_to_query_int(al, first, last)

    return cropped_al

def junction_microhomology(reference_sequences, first_al, second_al):
    if first_al is None or second_al is None:
        return -1

    als_by_order = {
        'first': first_al,
        'second': second_al,
    }
    
    covered_by_order = {order: interval.get_covered(al) for order, al in als_by_order.items()}
    
    side_to_order = {
        'left': min(covered_by_order, key=covered_by_order.get),
        'right': max(covered_by_order, key=covered_by_order.get),
    }

    covered_by_side = {side: covered_by_order[order] for side, order in side_to_order.items()}
    als_by_side = {side: als_by_order[order] for side, order in side_to_order.items()}
    
    initial_overlap = covered_by_side['left'] & covered_by_side['right']

    if initial_overlap:
        # Trim back mismatches or indels in or near the overlap.
        mismatch_buffer_length = 5

        bad_read_ps = {
            'left': set(),
            'right': set(),
        }

        for side in ['left', 'right']:
            al = als_by_side[side]

            for read_p, *rest in get_mismatch_info(al, reference_sequences):
                bad_read_ps[side].add(read_p)

            for kind, info in get_indel_info(al):
                if kind == 'deletion':
                    read_p, length = info
                    bad_read_ps[side].add(read_p)
                elif kind == 'insertion':
                    starts_at, ends_at = info
                    bad_read_ps[side].update([starts_at, ends_at])

        covered_by_side_trimmed = {}

        left_buffer_start = initial_overlap.start - mismatch_buffer_length

        left_illegal_ps = [p for p in bad_read_ps['left'] if p >= left_buffer_start]

        if left_illegal_ps:
            old_start = covered_by_side['left'].start
            new_end = int(np.floor(min(left_illegal_ps))) - 1
            covered_by_side_trimmed['left'] = interval.Interval(old_start, new_end)
        else:
            covered_by_side_trimmed['left'] = covered_by_side['left']

        right_buffer_end = initial_overlap.end + mismatch_buffer_length

        right_illegal_ps = [p for p in bad_read_ps['right'] if p <= right_buffer_end]

        if right_illegal_ps:
            new_start = int(np.ceil(max(right_illegal_ps))) + 1
            old_end = covered_by_side['right'].end
            covered_by_side_trimmed['right'] = interval.Interval(new_start, old_end)
        else:
            covered_by_side_trimmed['right'] = covered_by_side['right']
    else:
        covered_by_side_trimmed = {side: covered_by_side[side] for side in ['left', 'right']}

    if interval.are_disjoint(covered_by_side_trimmed['left'], covered_by_side_trimmed['right']):
        gap = covered_by_side_trimmed['right'].start - covered_by_side_trimmed['left'].end - 1
        MH_nts = -gap
    else:   
        overlap = covered_by_side_trimmed['left'] & covered_by_side_trimmed['right']

        MH_nts = overlap.total_length

    return MH_nts

def extend_alignment_allowing_programmed_subs(query_seq, target_seq, query_start, query_end, target_start, target_end, programmed_subs):
    while query_start > 0 and target_start > 0 and (query_seq[query_start - 1] == target_seq[target_start - 1] or ((target_start - 1) in programmed_subs and query_seq[query_start - 1] == programmed_subs[target_start - 1])):
        query_start -= 1
        target_start -= 1

    while query_end < len(query_seq) and target_end < len(target_seq) and (query_seq[query_end] == target_seq[target_end] or (query_end in programmed_subs and query_seq[query_end] == programmed_subs[query_end])):
        query_end += 1
        target_end += 1

    return query_start, query_end, target_start

def extend_alignment(initial_al, target_seq_bytes, programmed_subs):
    query_seq_bytes = initial_al.query_sequence.encode()
    
    if not isinstance(target_seq_bytes, bytes):
        target_seq_bytes = target_seq_bytes.encode()

    new_query_start, new_query_end, new_target_start = extend_alignment_allowing_programmed_subs(query_seq_bytes,
                                                                                                 target_seq_bytes,
                                                                                                 initial_al.query_alignment_start,
                                                                                                 initial_al.query_alignment_end,
                                                                                                 initial_al.reference_start,
                                                                                                 initial_al.reference_end,
                                                                                                 programmed_subs[initial_al.reference_name],
                                                                                                )
    added_to_start = initial_al.query_alignment_start - new_query_start
    added_to_end = new_query_end - initial_al.query_alignment_end
    cigar = initial_al.cigar
    
    if added_to_start > 0:
        # Remove from starting soft clip...
        kind, length = cigar[0]
        if kind != sam.BAM_CSOFT_CLIP:
            raise ValueError(f'expected soft-clip, got {kind}')
        
        cigar[0] = (kind, length - added_to_start)

        # ... and add to subsequent match.
        kind, length = cigar[1]
        if kind != sam.BAM_CMATCH:
            raise ValueError(f'expected match, got {kind}')
            
        cigar[1] = (kind, length + added_to_start)

        if cigar[0][1] == 0:
            cigar = cigar[1:]
        
    if added_to_end > 0:
        # Remove from ending soft clip...
        kind, length = cigar[-1]
        if kind != sam.BAM_CSOFT_CLIP:
            raise ValueError(f'expected soft-clip, got {kind}')
        
        cigar[-1] = (kind, length - added_to_end)
        
        # ... and add to subsequent match.
        kind, length = cigar[-2]
        if kind != sam.BAM_CMATCH:
            raise ValueError(f'expected match, got {kind}')
            
        cigar[-2] = (kind, length + added_to_end)

        if cigar[-1][1] == 0:
            cigar = cigar[:-1]

    if added_to_start > 0 or added_to_end > 0:
        new_al = copy.deepcopy(initial_al)
        new_al.cigar = cigar
        new_al.reference_start = new_target_start
    else:
        new_al = initial_al
    
    return new_al

def experiment_type_to_categorizer(experiment_type):
    from . import prime_editing
    from . import twin_prime
    from . import integrase
    from . import ENDseq
    from . import uditas
    from . import HDR

    experiment_type_to_categorizer = {
        'prime_editing': prime_editing.Architecture,
        'twin_prime': twin_prime.Architecture,
        'Bxb1_twin_prime': integrase.Architecture,
        'ENDseq': ENDseq.Architecture,
        'ENDseq_dual_flap': ENDseq.TwinPrimeArchitecture,
        'uditas': uditas.Architecture,
        'uditas_dual_flap': uditas.DualFlapArchitecture,
        'HDR': HDR.Architecture,
    }

    aliases = {
        'single_flap': 'prime_editing',
        'dual_flap': 'twin_prime',
        'Bxb1_dual_flap': 'Bxb1_twin_prime',
    }

    for alias, original_name in aliases.items():
        experiment_type_to_categorizer[alias] = experiment_type_to_categorizer[original_name]

    return experiment_type_to_categorizer[experiment_type]

def experiment_type_to_no_overlap_pair_categorizer(experiment_type):
    from . import ENDseq
    from . import uditas

    experiment_type_to_categorizer = {
        'ENDseq': ENDseq.NoOverlapPairArchitecture,
        'ENDseq_dual_flap': ENDseq.NoOverlapPairTwinPrimeArchitecture,
        'uditas': uditas.NoOverlapPairArchitecture,
    }

    return experiment_type_to_categorizer(experiment_type)

class ExtensionChain:
    def __init__(self,
                 link_specifications,
                 last_al_to_description,
                 specification_order,
                 direction_in_read,
                 require_definite=True,
                ):

        self.specification_order = specification_order
        self.direction_in_read = direction_in_read

        self.require_definite = require_definite

        if specification_order == 'reverse':
            link_specifications = link_specifications[::-1]

        feature_names = link_specifications[1::2] + [None]
        candidate_alignments_lists = link_specifications[::2]

        self.name_order = []

        self.links_by_name = {}

        previous_link = None

        for (candidate_alignments, names_tuple), next_feature_name in zip(candidate_alignments_lists, feature_names):
            if self.specification_order == 'forward':
                link_name = names_tuple[0]
            elif self.specification_order == 'reverse':
                link_name = names_tuple[1]
            else:
                raise ValueError(self.specification_order)

            link = ExtensionChainLink(
                link_name,
                candidate_alignments,
                next_feature_name,
            )

            if previous_link is not None:
                previous_link.next_link = link

            self.name_order.append(link_name)

            self.links_by_name[link_name] = link

            previous_link = link

        self.last_al_to_description = last_al_to_description

    def build(
        self,
        starting_alignment,
        entry_point,
        architecture,
    ):

        current_link = self.links_by_name[entry_point]
        current_link.alignment = starting_alignment

        while current_link.alignment is not None and current_link.next_link is not None:
            extension_al, cropped_extension_al, cropped_original_al = architecture.find_extending_alignment(
                current_link.alignment,
                self.direction_in_read,
                current_link.next_link.candidate_alignments,
                current_link.next_feature,
                require_definite=self.require_definite,
            )

            if extension_al is not None and cropped_original_al is not None:
                current_link.next_link.alignment = cropped_extension_al
                current_link.alignment = cropped_original_al

            current_link = current_link.next_link

        self.query_covered = interval.DisjointIntervals([])
        self.query_covered_incremental = {'none': self.query_covered}

        self.mismatches = {
            'none': 0,
        }

        entry_point_i = self.name_order.index(entry_point)

        self.alignments = {name: self.links_by_name[name].alignment for name in self.name_order if self.links_by_name[name].alignment is not None}

        self.alignments_list = list(self.alignments.values())

        for name_order_i in range(entry_point_i, len(self.name_order)):
            name = self.name_order[name_order_i]

            if name in self.alignments:
                self.mismatches[name] = architecture.edits(self.alignments[name])
                als_up_to = [self.alignments[other_name] for other_name in self.name_order[entry_point_i:name_order_i + 1]]
                self.query_covered = interval.get_disjoint_covered(als_up_to)
                self.query_covered_incremental[name] = self.query_covered

@dataclass
class ExtensionChainLink:
    name: str
    candidate_alignments: list[pysam.AlignedSegment]
    next_feature: str | None = None
    next_link: Self | None = None
    alignment: pysam.AlignedSegment| None = None