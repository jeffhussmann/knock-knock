from collections import Counter, defaultdict

import numpy as np
import pysam

from sequencing import interval, sam, utilities, sw, fastq
from sequencing.utilities import memoized_property

from knockin.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs

class Layout(object):
    def __init__(self, alignments, target_info, supplemental_headers=None):
        if supplemental_headers is None:
            supplemental_headers = {}
        self.supplemental_headers = supplemental_headers

        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.seq_bytes = self.seq.encode()
        self.qual = np.array(sam.get_original_qual(alignment))
        
        self.primary_ref_names = set(self.target_info.reference_sequences)

        self.required_sw = False

        self.special_alignment = None
        
        #self.relevant_alignments = self.parsimonious_target_alignments + self.donor_alignments

    @classmethod
    def from_read(cls, read, target_info):
        al = pysam.AlignedSegment(header)
        al.query_sequence = read.seq
        al.query_qualities = read.qual
        al.query_name = read.name
        return cls([al], target_info)
    
    @classmethod
    def from_seq(cls, seq, target_info):
        al = pysam.AlignedSegment(target_info.header)
        al.query_sequence = seq
        al.query_qualities = [41]*len(seq)
        return cls([al], target_info)
        
    @memoized_property
    def target_alignments(self):
        t_als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.target
        ]
        
        return t_als

    @memoized_property
    def donor_alignments(self):
        d_als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.donor
        ]
        
        return d_als
    
    @memoized_property
    def primary_alignments(self):
        p_als = [
            al for al in self.alignments
            if al.reference_name in self.primary_ref_names
        ]
        
        return p_als
    
    @memoized_property
    def supplemental_alignments(self):
        s_als = [
            al for al in self.alignments
            if al.reference_name not in self.primary_ref_names
            and not sam.contains_splicing(al)
        ]
        
        return s_als
    
    @memoized_property
    def phiX_alignments(self):
        als = [
            al for al in self.alignments
            if al.reference_name == 'phiX'
        ]
        
        return als

    def covers_whole_read(self, al):
        if al is None:
            return False

        covered = interval.get_covered(al)

        return len(self.whole_read - covered) == 0

    @memoized_property
    def merged_perfect_edge_alignments(self):
        ''' If the read can be explain as perfect alignments from each end that cover the whole query,
        return a single (possibly deletion containing) alignment that does this.
        '''
        edge_als, gap = self.perfect_edge_alignments

        if self.covers_whole_read(edge_als['left']):
            merged_al = edge_als['left']
        elif self.covers_whole_read(edge_als['right']):
            merged_al = edge_als['right']
        else:
            merged_al = sam.merge_adjacent_alignments(edge_als['left'], edge_als['right'], self.target_info.reference_sequences)

        return merged_al
    
    @memoized_property
    def parsimonious_target_alignments(self):
        if self.merged_perfect_edge_alignments is not None:
            # If it is possible to explain the read as a single deletion, do it.
            als = [self.merged_perfect_edge_alignments]
        else:
            als = interval.make_parsimonious(self.target_alignments)

        if len(als) == 0:
            return als

        if len(als) == 2 and sam.get_strand(als[0]) == sam.get_strand(als[1]):
            upstream, downstream = sorted(als, key=lambda al: al.reference_start)

            merged = sam.merge_adjacent_alignments(upstream, downstream, self.target_info.reference_sequences)
            if merged is not None:
                als = [merged]

        # If the right edge of the read isn't covered, try to merge a perfect edge alignment to the right-most alignment.
        covered = interval.get_disjoint_covered(als)
        if len(self.seq) - 1 not in covered:
            right_most = max(als, key=lambda al: interval.get_covered(al).end)
            other = [al for al in als if al != right_most]

            perfect_edge_als, gap = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(right_most, perfect_edge_als['right'], self.target_info.reference_sequences)
            if merged is None:
                merged = right_most

            als = other + [merged]
        
        if 0 not in covered:
            left_most = min(als, key=lambda al: interval.get_covered(al).start)
            other = [al for al in als if al != left_most]

            perfect_edge_als, gap = self.perfect_edge_alignments
            merged = sam.merge_adjacent_alignments(perfect_edge_als['left'], left_most, self.target_info.reference_sequences)
            if merged is None:
                merged = left_most

            als = other + [merged]
                
        #exempt_if_overlaps = self.target_info.around_cuts(5)
        #split_als = []
        #for al in als:
        #    split_als.extend(sam.split_at_deletions(al, 3, exempt_if_overlaps))
        split_als = als
        
        return split_als
    
    @memoized_property
    def split_target_and_donor_alignments(self):
        split_als = []
        for al in self.target_alignments + self.donor_alignments:
            split_at_dels = sam.split_at_deletions(al, 3)

            split_at_ins = []
            for split_al in split_at_dels:
                split_at_ins.extend(sam.split_at_large_insertions(split_al, 2))

            target_seq_bytes = self.target_info.reference_sequences[al.reference_name].encode()
            extended = [sw.extend_alignment(split_al, target_seq_bytes) for split_al in split_at_ins]

            split_als.extend(extended)

        return split_als

    @memoized_property
    def split_donor_alignments(self):
        return [al for al in self.split_target_and_donor_alignments if al.reference_name == self.target_info.donor]
    
    @memoized_property
    def original_target_edge_alignments(self):
        return self.get_target_edge_alignments(self.parsimonious_target_alignments)
    
    @memoized_property
    def realigned_target_edge_alignments(self):
        return self.get_target_edge_alignments(self.realigned_target_alignments + self.short_edge_alignments)
    
    def get_target_edge_alignments(self, alignments):
        ''' Get target alignments that make it to the read edges. '''
        edge_alignments = {'left': [], 'right':[]}

        split_als = []
        for al in alignments:
            split_at_dels = sam.split_at_deletions(al, 3)

            split_at_ins = []
            for split_al in split_at_dels:
                split_at_ins.extend(sam.split_at_large_insertions(split_al, 2))
            
            target_seq_bytes = self.target_info.reference_sequences[al.reference_name].encode()
            extended = [sw.extend_alignment(split_al, target_seq_bytes) for split_al in split_at_ins]

            split_als.extend(extended)

        for al in split_als:
            if sam.get_strand(al) != self.target_info.primers_by_side_of_target[3].strand:
                continue

            covered = interval.get_covered(al)
            
            if covered.start <= 2:
                edge_alignments['left'].append(al)
            
            if covered.end == len(self.seq) - 1:
                edge_alignments['right'].append(al)

        for edge in ['left', 'right']:
            if len(edge_alignments[edge]) == 0:
                edge_alignments[edge] = None
            else:
                edge_alignments[edge] = max(edge_alignments[edge], key=lambda al: al.query_alignment_length)

        return edge_alignments

    def extends_past_PAS(self, al):
        if (self.target_info.target, 'PAS') in self.target_info.features:
            PAS = self.target_info.features[self.target_info.target, 'PAS']
            return al.reference_start <= PAS.start <= al.reference_end
        else:
            return False

    @memoized_property
    def whole_read(self):
        return interval.Interval(0, len(self.seq) - 1)
    
    @memoized_property
    def single_target_alignment(self):
        t_als = self.parsimonious_target_alignments
        
        if len(t_als) != 1:
            return None
        else:
            return t_als[0]

    @memoized_property
    def query_missing_from_single_target_alignment(self):
        t_al = self.single_target_alignment
        if t_al is None:
            return None
        else:
            split_als = sam.split_at_large_insertions(t_al, 5)
            covered = interval.get_disjoint_covered(split_als)
            ignoring_edges = interval.Interval(covered.start, covered.end)

            missing_from = {
                'start': covered.start,
                'end': len(self.seq) - covered.end - 1,
                'middle': (ignoring_edges - covered).total_length,
            }

            return missing_from

    @memoized_property
    def single_alignment_covers_read(self):
        missing_from = self.query_missing_from_single_target_alignment

        if missing_from is None:
            return False
        else:
            not_too_much = {
                'start': missing_from['start'] <= 2,
                'end': missing_from['end'] <= 2,
                'middle': missing_from['middle'] <= 5,
            }

            return all(not_too_much.values())

    @memoized_property
    def target_reference_edges(self):
        ''' reference positions on target of alignments that make it to the read edges. '''
        edges = {5: None, 3: None}
        # confusing: 'edge' means 5 or 3, 'side' means left or right here.
        for edge, side in [(5, 'left'), (3, 'right')]:
            edge_al = self.original_target_edge_alignments[side]
            if edge_al is not None:
                edges[side] = sam.reference_edges(edge_al)[edge]

        return edges

    @memoized_property
    def starts_at_primer(self):
        edge_al = self.original_target_edge_alignments['left']
        if edge_al is None:
            return False
        else:
            primer = self.target_info.primers_by_side_of_target[3]
            return sam.overlaps_feature(edge_al, primer)

    @memoized_property
    def Q30_fractions(self):
        at_least_30 = self.qual >= 30
        fracs = {
            'all': np.mean(at_least_30),
            'second_half': np.mean(at_least_30[len(at_least_30) // 2:]),
        }
        return fracs

    @memoized_property
    def SNVs_summary(self):
        SNPs = self.target_info.donor_SNVs
        position_to_name = {SNPs['target'][name]['position']: name for name in SNPs['target']}
        donor_SNV_locii = {name: [] for name in SNPs['target']}

        other_locii = []

        for al in self.parsimonious_target_alignments:
            for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if ref_i in position_to_name:
                    name = position_to_name[ref_i]

                    if SNPs['target'][name]['strand'] == '-':
                        read_b = utilities.reverse_complement(read_b)

                    donor_SNV_locii[name].append((read_b, qual))

                else:
                    if read_b != '-' and ref_b != '-' and read_b != ref_b:
                        snv = SNV(ref_i, read_b, qual)
                        other_locii.append(snv)

        other_locii = SNVs(other_locii)
        return donor_SNV_locii, other_locii

    @memoized_property
    def non_donor_SNVs(self):
        _, other_locii = self.SNVs_summary
        return other_locii

    def specific_to_donor(self, donor_al):
        ''' Does donor_al contain a donor SNV? '''
        ti = self.target_info

        contains_SNV = False

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(donor_al, ti.donor_sequence):
            if ref_i in ti.simple_donor_SNVs:
                if read_b == ti.simple_donor_SNVs[ref_i]:
                    contains_SNV = True

        return contains_SNV

    @memoized_property
    def donor_SNV_locii_summary(self):
        SNPs = self.target_info.donor_SNVs
        donor_SNV_locii, _ = self.SNVs_summary
        
        genotype = {}

        has_donor_SNV = False

        for name in sorted(SNPs['target']):
            bs = defaultdict(list)

            for b, q in donor_SNV_locii[name]:
                bs[b].append(q)

            if len(bs) == 0:
                genotype[name] = '-'
            elif len(bs) != 1:
                genotype[name] = 'N'
            else:
                b, qs = list(bs.items())[0]

                if b == SNPs['target'][name]['base']:
                    genotype[name] = '_'
                else:
                    genotype[name] = b
                
                    if b == SNPs['donor'][name]['base']:
                        has_donor_SNV = True

        string_summary = ''.join(genotype[name] for name in sorted(SNPs['target']))

        return has_donor_SNV, string_summary

    @memoized_property
    def has_donor_SNV(self):
        has_donor_SNV, _ = self.donor_SNV_locii_summary
        return has_donor_SNV

    @memoized_property
    def has_any_SNV(self):
        return self.has_donor_SNV or (len(self.non_donor_SNVs) > 0)

    @memoized_property
    def donor_SNV_string(self):
        _, string_summary = self.donor_SNV_locii_summary
        return string_summary

    @memoized_property
    def indels(self):
        ti = self.target_info

        around_cut_interval = ti.around_cuts(10)

        indels = []
        for al in self.parsimonious_target_alignments:
            for i, (cigar_op, length) in enumerate(al.cigar):
                if cigar_op == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before
                    ends_at = starts_at + length - 1

                    indel_interval = interval.Interval(starts_at, ends_at)

                    indel = DegenerateDeletion([starts_at], length)

                elif cigar_op == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    indel_interval = interval.Interval(starts_after, starts_after)

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    indel = DegenerateInsertion([starts_after], [insertion])
                    
                else:
                    continue

                near_cut = len(indel_interval & around_cut_interval) > 0

                indel = self.target_info.expand_degenerate_indel(indel)
                indels.append((indel, near_cut))

        # Ignore any 1 nt deletions in the polyT track.
        polyT = self.target_info.features.get((self.target_info.target, 'polyT-track'))
        if polyT is not None:
            polyT_interval = interval.Interval(polyT.start, polyT.end)
            def should_ignore(indel):
                if indel.kind != 'D' or indel.length != 1:
                    return False
                deletion_interval = interval.Interval(min(indel.starts_ats), max(indel.ends_ats))
                return len(deletion_interval & polyT_interval) > 1

            indels = [(indel, near_cut) for indel, near_cut in indels if not should_ignore(indel)]

        return indels

    @memoized_property
    def indels_near_cut(self):
        return [indel for indel, near_cut in self.indels if near_cut]

    @memoized_property
    def indels_string(self):
        reps = [str(indel) for indel in self.indels_near_cut]
        string = ' '.join(reps)
        return string

    @memoized_property
    def covered_from_target(self):
        als = list(self.original_target_edge_alignments.values())
        return interval.get_disjoint_covered(als)
    
    @memoized_property
    def not_covered_by_target_or_donor(self):
        target_and_donor = self.parsimonious_target_alignments + self.donor_alignments
        covered = interval.get_disjoint_covered(target_and_donor)
        return self.whole_read - covered

    @memoized_property
    def perfect_gap_als(self):
        all_gap_als = []

        for query_interval in self.not_covered_by_target_or_donor:
            for on in ['target', 'donor']:
                gap_als = self.seed_and_extend(on, query_interval.start, query_interval.end)
                all_gap_als.extend(gap_als)

        return all_gap_als
    
    @memoized_property
    def nonredundant_supplemental_alignments(self):
        nonredundant = []
        
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            novel_length = (covered - self.covered_from_target).total_length
            if novel_length >= 10:
                nonredundant.append(al)

        return nonredundant

    @memoized_property
    def original_header(self):
        return self.alignments[0].header

    @memoized_property
    def read(self):
        return fastq.Read(self.name, self.seq, fastq.encode_sanger(self.qual))

    @memoized_property
    def short_edge_alignments(self):
        ''' Look for short alignments at the end of the read to sequence around the cut site. '''
        ti = self.target_info
        
        cut_after = ti.cut_after
        before_cut = utilities.reverse_complement(ti.target_sequence[:cut_after + 1])[:25]
        after_cut = ti.target_sequence[cut_after + 1:][:25]

        new_targets = [
            ('edge_before_cut', before_cut),
            ('edge_after_cut', after_cut),
        ]

        # Note: this used to use self.original_header instead of ti.header
        full_refs = list(ti.header.references) + [name for name, seq in new_targets]
        full_lengths = list(ti.header.lengths) + [len(seq) for name, seq in new_targets]

        expanded_header = pysam.AlignmentHeader.from_references(full_refs, full_lengths)

        edge_alignments = sw.align_read(self.read, new_targets, 3, expanded_header, both_directions=False, alignment_type='query_end', N_matches=False)

        def comparison_key(al):
            return (al.reference_name, al.reference_start, al.is_reverse, al.cigarstring)

        already_seen = {comparison_key(al) for al in self.target_alignments}

        new_alignments = []

        for alignment in edge_alignments:
            if alignment.reference_name == 'edge_before_cut':
                alignment.is_reverse = True
                alignment.cigar = alignment.cigar[::-1]
                qual = alignment.query_qualities[::-1]
                alignment.query_sequence = utilities.reverse_complement(alignment.query_sequence)
                alignment.query_qualities = qual
                alignment.reference_start = cut_after + 1 - sam.total_reference_nucs(alignment.cigar)
            else:
                alignment.reference_start = cut_after + 1
                
            alignment.reference_name = ti.target

            if comparison_key(alignment) not in already_seen:
                already_seen.add(comparison_key(alignment))
                new_alignments.append(alignment)
        
        return new_alignments

    @memoized_property
    def sw_alignments(self):
        self.required_sw = True

        ti = self.target_info

        targets = [
            (ti.target, ti.target_sequence),
        ]

        if ti.donor is not None:
            targets.append((ti.donor, ti.donor_sequence))

        stringent_als = sw.align_read(self.read, targets, 5, ti.header,
                                      max_alignments_per_target=10,
                                      mismatch_penalty=-8,
                                      indel_penalty=-60,
                                      min_score_ratio=0,
                                      both_directions=True,
                                      N_matches=False,
                                     )

        no_Ns = [al for al in stringent_als if 'N' not in al.get_tag('MD')]

        return no_Ns

    @memoized_property
    def perfect_local_target_alignments(self):
        ti = self.target_info

        targets = [
            (ti.target, ti.target_sequence),
        ]

        perfect_als = sw.align_read(self.read, targets, 5, self.original_header,
                                    max_alignments_per_target=30,
                                    mismatch_penalty=-10000,
                                    indel_penalty=-10000,
                                    min_score_ratio=0,
                                    both_directions=True,
                                    N_matches=False,
                                   )

        return perfect_als

    def seed_and_extend(self, on, query_start, query_end):
        extender = self.target_info.seed_and_extender[on]
        return extender(self.seq_bytes, query_start, query_end, self.name)
    
    @memoized_property
    def valid_intervals_for_edge_alignments(self):
        # this primer should be annotated 'R2' or 'expected start'
        ti = self.target_info
        forward_primer = ti.primers_by_side_of_target[5]
        reverse_primer = ti.primers_by_side_of_target[3]
        valids = {
            'left': interval.Interval(ti.cut_after + 1, reverse_primer.end),
            'right': interval.Interval(forward_primer.start, ti.cut_after + 1),
        }
        return valids

    @memoized_property
    def valid_strand_for_edge_alignments(self):
        return '-'

    @memoized_property
    def perfect_edge_alignments(self):
        # Set up keys to prioritize alignments.
        # Prioritize by (correct strand and side of cut, length (longer better), distance from cut (closer better)) 
        
        def sort_key(al, side):
            length = al.query_alignment_length

            valid_int = self.valid_intervals_for_edge_alignments[side]
            valid_strand = self.valid_strand_for_edge_alignments

            correct_strand = sam.get_strand(al) == valid_strand

            cropped = sam.crop_al_to_ref_int(al, valid_int.start, valid_int.end)
            if cropped is None or cropped.is_unmapped:
                correct_side = 0
                inverse_distance = 0
            else:
                correct_side = 1

                if side == 'left':
                    if valid_strand == '+':
                        edge = cropped.reference_end - 1
                    else:
                        edge = cropped.reference_start
                else:
                    if valid_strand == '+':
                        edge = cropped.reference_start
                    else:
                        edge = cropped.reference_end - 1

                inverse_distance = 1 / (abs(edge - self.target_info.cut_after) + 0.1)

            return correct_strand & correct_side, length, inverse_distance

        def is_valid(al, side):
            correct_strand_and_side, length, inverse_distance = sort_key(al, side)
            return correct_strand_and_side
        
        best_edge_als = {'left': None, 'right': None}

        # Insist that the alignments be to the right side and strand, even if longer ones
        # to the wrong side or strand exist.
        for side in ['left', 'right']:
            for length in range(20, 3, -1):
                if side == 'left':
                    start = 0
                    end = length
                else:
                    start = len(self.seq) - length
                    end = len(self.seq)

                als = self.seed_and_extend('target', start, end)

                valid = [al for al in als if is_valid(al, side)]
                if len(valid) > 0:
                    break
                
            if len(valid) > 0:
                key = lambda al: sort_key(al, side)
                best_edge_als[side] = max(valid, key=key)

        covered_from_edges = interval.get_disjoint_covered(best_edge_als.values())
        uncovered = self.whole_read - covered_from_edges
        
        if uncovered.total_length == 0:
            gap_interval = None
        elif len(uncovered.intervals) > 1:
            # This shouldn't be possible since seeds start at each edge
            raise ValueError('disjoint gap', uncovered)
        else:
            gap_interval = uncovered.intervals[0]
        
        return best_edge_als, gap_interval

    @memoized_property
    def perfect_edge_alignment_reference_edges(self):
        best_edge_als, gap_interval = self.perfect_edge_alignments

        # TODO: this isn't general for a primer upstream of cut
        left_edge = sam.reference_edges(best_edge_als['left'])[3]
        right_edge = sam.reference_edges(best_edge_als['right'])[5]

        return left_edge, right_edge

    def reference_distances_from_perfect_edge_alignments(self, al):
        al_edges = sam.reference_edges(al)
        left_edge, right_edge = self.perfect_edge_alignment_reference_edges 
        return abs(left_edge - al_edges[5]), abs(right_edge - al_edges[3])

    def gap_covering_alignments(self, required_MH_start, required_MH_end):
        longest_edge_als, gap_interval = self.perfect_edge_alignments
        gap_query_start = gap_interval.start - required_MH_start
        # Note: interval end is the last base, but seed_and_extend wants one past
        gap_query_end = gap_interval.end + 1 + required_MH_end
        gap_covering_als = self.seed_and_extend('target', gap_query_start, gap_query_end)
        return gap_covering_als
    
    def partial_gap_perfect_alignments(self, required_MH_start, required_MH_end, on='target', only_close=True):
        def close_enough(al):
            if not only_close:
                return True
            else:
                return min(*self.reference_distances_from_perfect_edge_alignments(al)) < 100

        edge_als, gap_interval = self.perfect_edge_alignments
        if gap_interval is None:
            return [], []

        # Note: interval end is the last base, but seed_and_extend wants one past
        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end

        from_start_gap_als = []
        while (end > start) and not from_start_gap_als:
            end -= 1
            from_start_gap_als = self.seed_and_extend(on, start, end)
            from_start_gap_als = [al for al in from_start_gap_als if close_enough(al)]
            
        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end
        from_end_gap_als = []
        while (end > start) and not from_end_gap_als:
            start += 1
            from_end_gap_als = self.seed_and_extend(on, start, end)
            from_end_gap_als = [al for al in from_end_gap_als if close_enough(al)]

        return from_start_gap_als, from_end_gap_als

    @memoized_property
    def multi_step_SD_MMEJ_gap_cover(self):
        partial_als = {}
        partial_als['start'], partial_als['end'] = self.partial_gap_perfect_alignments(2, 2)
        def is_valid(al):
            close_enough = min(self.reference_distances_from_perfect_edge_alignments(al)) < 50
            return close_enough and not self.target_info.overlaps_cut(al)

        valid_als = {side: [al for al in partial_als[side] if is_valid(al)] for side in ('start', 'end')}
        intervals = {side: [interval.get_covered(al) for al in valid_als[side]] for side in ('start', 'end')}

        part_of_cover = {'start': set(), 'end': set()}

        valid_cover_found = False
        for s, start_interval in enumerate(intervals['start']):
            for e, end_interval in enumerate(intervals['end']):
                if len((start_interval & end_interval)) >= 2:
                    valid_cover_found = True
                    part_of_cover['start'].add(s)
                    part_of_cover['end'].add(e)

        if valid_cover_found:
            final_als = {side: [valid_als[side][i] for i in part_of_cover[side]] for side in ('start', 'end')}
            return final_als
        else:
            return None

    @memoized_property
    def SD_MMEJ(self):
        details = {}

        best_edge_als, gap_interval = self.perfect_edge_alignments
        overlaps_cut = self.target_info.overlaps_cut

        if best_edge_als['left'] is None or best_edge_als['right'] is None:
            details['failed'] = 'missing edge alignment'
            return details

        details['edge alignments'] = best_edge_als
        details['alignments'] = list(best_edge_als.values())
        details['all alignments'] = list(best_edge_als.values())
        
        if best_edge_als['left'] == best_edge_als['right']:
            details['failed'] = 'perfect wild type'
            return details
        elif sam.get_strand(best_edge_als['left']) != sam.get_strand(best_edge_als['right']):
            details['failed'] = 'edges align to different strands'
            return details
        else:
            edge_als_strand = sam.get_strand(best_edge_als['left'])
        
        details['left edge'], details['right edge'] = self.perfect_edge_alignment_reference_edges

        # Require resection on both sides of the cut.
        for edge in ['left', 'right']:
            if overlaps_cut(best_edge_als[edge]):
                details['failed'] = f'{edge}\' edge alignment extends over cut'
                return details

        if gap_interval is None:
            details['failed'] = 'no gap' 
            return details

        details['gap length'] = len(gap_interval)

        # Insist on at least 2 nt of MH on each side.
        gap_covering_als = self.gap_covering_alignments(2, 2)

        min_distance = np.inf
        closest_gap_covering = None

        for al in gap_covering_als:
            left_distance, right_distance = self.reference_distances_from_perfect_edge_alignments(al)
            distance = min(left_distance, right_distance)
            if distance < 100:
                details['all alignments'].append(al)

            # A valid gap covering alignment must lie entirely on one side of the cut site in the target.
            if distance < min_distance and not overlaps_cut(al):
                min_distance = distance
                closest_gap_covering = al

        # Empirically, the existence of any gap alignments that cover cut appears to be from overhang duplication, not SD-MMEJ.
        # but not comfortable excluding these yet
        #if any(overlaps_cut(al) for al in gap_covering_als):
        #    details['failed'] = 'gap alignment overlaps cut'
        #    return details
        
        if min_distance <= 50:
            details['gap alignment'] = closest_gap_covering
            gap_edges = sam.reference_edges(closest_gap_covering)
            details['gap edges'] = {'left': gap_edges[5], 'right': gap_edges[3]}

            if closest_gap_covering is not None:
                gap_covering_strand = sam.get_strand(closest_gap_covering)
                if gap_covering_strand == edge_als_strand:
                    details['kind'] = 'loop-out'
                else:
                    details['kind'] = 'snap-back'

            details['alignments'].append(closest_gap_covering)

            gap_covered = interval.get_covered(closest_gap_covering)
            edge_covered = {side: interval.get_covered(best_edge_als[side]) for side in ['left', 'right']}
            homology_lengths = {side: len(gap_covered & edge_covered[side]) for side in ['left', 'right']}

            details['homology lengths'] = homology_lengths
            
        else:
            # Try to cover with multi-step.
            multi_step_als = self.multi_step_SD_MMEJ_gap_cover
            if multi_step_als is not None:
                for side in ['start', 'end']:
                    details['alignments'].extend(multi_step_als[side])
                details['kind'] = 'multi-step'
                details['gap edges'] = {'left': 'PH', 'right': 'PH'}
                details['homology lengths'] = {'left': 'PH', 'right': 'PH'}
            else:
                details['failed'] = 'no valid alignments cover gap' 
                return details

        return details

    @memoized_property
    def is_valid_SD_MMEJ(self):
        return 'failed' not in self.SD_MMEJ

    @memoized_property
    def SD_MMEJ_string(self):
        fields = [
            self.SD_MMEJ['left edge'],
            self.SD_MMEJ['gap edges']['left'],
            self.SD_MMEJ['gap edges']['right'],
            self.SD_MMEJ['right edge'],
            self.SD_MMEJ['gap length'],
            self.SD_MMEJ['homology lengths']['left'],
            self.SD_MMEJ['homology lengths']['right'],
        ]
        return ','.join(str(f) for f in fields)

    @memoized_property
    def realigned_target_alignments(self):
        return [al for al in self.sw_alignments if al.reference_name == self.target_info.target]
    
    @memoized_property
    def realigned_donor_alignments(self):
        return [al for al in self.sw_alignments if al.reference_name == self.target_info.donor]
    
    @memoized_property
    def genomic_insertion(self):
        possible = self.identify_genomic_insertions()
        valid = [d for d in possible if 'failed' not in d]

        def priority(details):
            key_order = [
                'gap_before',
                'gap_after',
                'edit_distance',
            ]
            return [details[k] for k in key_order]

        valid = sorted(valid, key=priority)

        if len(valid) == 0:
            valid = None

        return valid

    @memoized_property
    def genomic_insertion_string(self):
        details = self.genomic_insertion[0]
        fields = [
            details['target_bounds'][5],
            details['target_bounds'][3],
            details['gap_before'],
            details['gap_after'],
            details['edit_distance'],
            details['chr'],
            details['genomic_bounds'][5],
            details['genomic_bounds'][3],
        ]
        return ','.join(map(str, fields))

    def identify_genomic_insertions(self):
        target_seq = self.target_info.target_sequence

        possible_insertions = []

        edge_covered = {}
        target_bounds = {}

        perfect_edge_alignments, _ = self.perfect_edge_alignments

        edge_als = {}
        for side in ['left', 'right']:
            original_al = self.original_target_edge_alignments[side]
            if original_al is not None:
                edge_als[side] = original_al
            else:
                edge_als[side] = perfect_edge_alignments[side]

        if edge_als['left'] is None:
            edge_covered['left'] = interval.Interval(-np.inf, -1)
            target_bounds['left'] = None
        else:
            edge_covered['left'] = interval.get_covered(edge_als['left'])
            target_bounds['left'] = sam.reference_edges(edge_als['left'])[3]

            # If there is a target alignment at the left edge, insist that
            # it include the reverse primer.
            if not sam.overlaps_feature(edge_als['left'], self.target_info.primers_by_side_of_target[3]):
                return [{'failed': 'no left edge alignment'}]

        if edge_als['right'] is None:
            edge_covered['right'] = interval.Interval(len(self.seq), np.inf)
            target_bounds['right'] = None
        else:
            edge_covered['right'] = interval.get_covered(edge_als['right'])
            target_bounds['right'] = sam.reference_edges(edge_als['right'])[5]

        for genomic_al in self.nonredundant_supplemental_alignments:
            genomic_covered = interval.get_covered(genomic_al)

            non_overlapping = genomic_covered - edge_covered['left'] - edge_covered['right']
            if len(non_overlapping) == 0:
                details = {'failed': 'doesn\'t cover any of gap'}
                possible_insertions.append(details)
                continue

            left_overlap = edge_covered['left'] & genomic_covered
            right_overlap = edge_covered['right'] & genomic_covered
            
            target_bounds = {}
            genomic_bounds = {}
            
            left_switch_after, left_min_edits, gap_before = sam.find_best_query_switch_after(edge_als['left'], genomic_al, target_seq, None, max)
            right_switch_after, right_min_edits, gap_after = sam.find_best_query_switch_after(genomic_al, edge_als['right'], None, target_seq, min)

            if edge_als['left'] is None:
                cropped_left_al = None
                target_bounds[5] = None
            else:
                cropped_left_al = sam.crop_al_to_query_int(edge_als['left'], -np.inf, left_switch_after)
                target_bounds[5] = sam.reference_edges(cropped_left_al)[3]

            if edge_als['right'] is None:
                cropped_right_al = None
                target_bounds[3] = None
            else:
                cropped_right_al = sam.crop_al_to_query_int(edge_als['right'], right_switch_after + 1, np.inf)
                if cropped_right_al is None:
                    target_bounds[3] = None
                else:
                    target_bounds[3] = sam.reference_edges(cropped_right_al)[5]

            cropped_genomic_al = sam.crop_al_to_query_int(genomic_al, left_switch_after + 1, right_switch_after)
            genomic_bounds = sam.reference_edges(cropped_genomic_al)   
                
            middle_edits = sam.edit_distance_in_query_interval(genomic_al, non_overlapping)
            total_edit_distance = left_min_edits + middle_edits + right_min_edits

            organism, original_name = cropped_genomic_al.reference_name.split('_', 1)
            if organism not in self.supplemental_headers:
                raise ValueError(organism, self.supplemental_headers)
            else:
                header = self.supplemental_headers[organism]
                al_dict = cropped_genomic_al.to_dict()
                al_dict['ref_name'] = original_name
                original_al = pysam.AlignedSegment.from_dict(al_dict, header)
            
            details = {
                'gap_before': gap_before,
                'gap_after': gap_after,
                'edit_distance': total_edit_distance,
                'chr': genomic_al.reference_name,
                'organism': organism,
                'original_alignment': original_al,
                'cropped_genomic_alignment': cropped_genomic_al,
                'genomic_bounds': genomic_bounds,
                'target_bounds': target_bounds,
                'cropped_alignments': [al for al in [cropped_left_al, cropped_genomic_al, cropped_right_al] if al is not None],
                'full_alignments': [al for al in [edge_als['left'], genomic_al, edge_als['right']] if al is not None],
            }

            failures = []

            if gap_before > 5:
                failures.append(f'gap_before = {gap_before}')

            if gap_after > 5:
                failures.append(f'gap_after = {gap_after}')

            if total_edit_distance > 5:
                failures.append(f'edit_distance = {total_edit_distance}')

            possible_insertions.append(details)

        return possible_insertions

    @memoized_property
    def donor_insertion(self):
        if self.target_info.donor is None:
            valid = []
        else:
            possible = self.identify_donor_insertions(self.original_target_edge_alignments, self.split_donor_alignments)
            valid = [d for d in possible if 'failed' not in d]

            perfect_edge_als, _ = self.perfect_edge_alignments
            if len(valid) == 0:
                possible = self.identify_donor_insertions(perfect_edge_als, self.split_donor_alignments)
                valid = [d for d in possible if 'failed' not in d]
            
            #if len(valid) == 0:
            #    possible = self.identify_donor_insertions(self.realigned_target_edge_alignments, self.realigned_donor_alignments)
            #    valid = [d for d in possible if 'failed' not in d]

        def priority(details):
            key_order = [
                'gap_before',
                'gap_after',
                'edit_distance',
            ]
            return [details[k] for k in key_order]

        valid = sorted(valid, key=priority)

        if len(valid) == 0:
            results = None
        else:
            best_explanation = valid[0]

            shared_HAs = best_explanation['shared_HAs']
            if len(shared_HAs) > 1:
                subcategory = 'uses both homology arms'
            elif len(shared_HAs) == 1:
                read_side = shared_HAs.pop()
                PAM_side = self.target_info.read_side_to_PAM_side[read_side]
                subcategory = f'uses {PAM_side} homology'
            else:
                subcategory = 'neither homology arm used'

            if not best_explanation['contains_donor_SNV']:
                subcategory += ' (no SNVs)'
            
            fields = [
                best_explanation['strand'],
                best_explanation['target_bounds'][5],
                best_explanation['target_bounds'][3],
                best_explanation['gap_before'],
                best_explanation['gap_after'],
                best_explanation['edit_distance'],
                best_explanation['length'],
                best_explanation['donor_bounds'][5],
                best_explanation['donor_bounds'][3],
            ]
            details_string = ','.join(map(str, fields))

            relevant_alignments = best_explanation['full_alignments'] + best_explanation['concatamer_alignments']

            results = subcategory, details_string, relevant_alignments, best_explanation

        return results
    
    @memoized_property
    def donor_deletions_seen(self):
        seen = [d for d, _ in self.indels if d.kind == 'D' and d in self.target_info.donor_deletions]
        return seen

    def shared_HAs(self, donor_al, target_edge_als):
        q_to_HA_offsets = defaultdict(lambda: defaultdict(set))

        for side in ['left', 'right']:
            # Only register the left-most occurence of the left HA in the left target edge alignment
            # and right-most occurece of the right HA in the right target edge alignment.
            all_q_to_HA_offset = {}
            
            al = target_edge_als[side]
            
            for q, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if q is None:
                    continue
                offset = self.target_info.HA_ref_p_to_offset['target', side].get(ref_i)
                if offset is not None:
                    all_q_to_HA_offset[q] = offset
            
            if side == 'left':
                get_most_extreme = min
                direction = 1
            else:
                get_most_extreme = max
                direction = -1
                
            if len(all_q_to_HA_offset) > 0:
                q = get_most_extreme(all_q_to_HA_offset)
                while q in all_q_to_HA_offset:
                    q_to_HA_offsets[q][side, all_q_to_HA_offset[q]].add('target')
                    q += direction
                
        for q, read_b, ref_i, ref_b, qual in sam.aligned_tuples(donor_al, self.target_info.donor_sequence):
            if q is None:
                continue
            for side in ['left', 'right']:
                offset = self.target_info.HA_ref_p_to_offset['donor', side].get(ref_i)
                if offset is not None:
                    q_to_HA_offsets[q][side, offset].add('donor')
            
        shared = set()
        for q in q_to_HA_offsets:
            for side, offset in q_to_HA_offsets[q]:
                if len(q_to_HA_offsets[q][side, offset]) == 2:
                    shared.add(side)

        return shared

    def long_insertion_framework(self, edge_als):
        edge_covered = {}

        if edge_als['left'] is None:
            edge_covered['left'] = interval.Interval(-np.inf, -1)
        else:
            edge_covered['left'] = interval.get_covered(edge_als['left'])
        
        if edge_als['right'] is None:
            edge_covered['right'] = interval.Interval(len(self.seq), np.inf)
        else:
            edge_covered['right'] = interval.get_covered(edge_als['right'])

        return edge_covered

    def identify_donor_insertions(self, edge_als, donor_als):
        ''' edge_als: dictionary of alignments that include the 5' and 3' edges of the read
        '''
        ti = self.target_info

        cut_after = ti.cut_after

        # (5' edge of 3' target alignment  - cut_after) will be multiplied by
        # resection_sign to give how many bases were resected on the sequencing
        # primer distal side of the cut.
        if ti.cut_after >= ti.primers_by_side_of_target[3].start:
            resection_sign = 1
        else:
            resection_sign = -1

        target_seq = ti.target_sequence
        donor_seq = ti.donor_sequence

        edge_covered = {}
        if edge_als['left'] is not None:
            cropped_to_cut = sam.crop_al_to_ref_int(edge_als['left'], cut_after + 1, np.inf) 
            edge_covered['left'] = interval.get_covered(cropped_to_cut)
            if not sam.overlaps_feature(cropped_to_cut, self.target_info.primers_by_side_of_target[3]):
                return [{'failed': 'left edge alignment isn\'t to primer'}]
        else:
            return [{'failed': 'no left edge alignment'}]

        if edge_als['right'] is not None:
            cropped_to_cut = sam.crop_al_to_ref_int(edge_als['right'], -np.inf, cut_after) 
            edge_covered['right'] = interval.get_covered(cropped_to_cut)
        else:
            return [{'failed': 'no right edge alignment'}]

        #covered_from_edges = interval.DisjointIntervals(edge_covered.values())

        possible_inserts = []

        for donor_al in donor_als:
            #insert_covered = interval.get_covered(donor_al)
            #novel_length = (insert_covered - covered_from_edges).total_length

            #if novel_length == 0:
            #    details = {'failed': 'donor alignment covered by edge alignments'}
            #    possible_inserts.append(details)
            #    continue
                
            target_bounds = {}
            insert_bounds = {}

            left_switch_after, left_min_edits, gap_before = sam.find_best_query_switch_after(edge_als['left'], donor_al, target_seq, donor_seq, max)
            right_switch_after, right_min_edits, gap_after = sam.find_best_query_switch_after(donor_al, edge_als['right'], donor_seq, target_seq, min)

            cropped_left_al = sam.crop_al_to_query_int(edge_als['left'], -np.inf, left_switch_after)
            cropped_right_al = sam.crop_al_to_query_int(edge_als['right'], right_switch_after + 1, np.inf)

            cropped_donor_al = sam.crop_al_to_query_int(donor_al, left_switch_after + 1, right_switch_after)

            if cropped_donor_al is None or cropped_donor_al.is_unmapped:
                details = {'failed': 'cropping removes entire donor alignment'}
                possible_inserts.append(details)
                continue

            if cropped_left_al is None:
                target_bounds[5] = None
            else:
                target_bounds[5] = sam.reference_edges(cropped_left_al)[3]

            if cropped_right_al is None:
                target_bounds[3] = None
            else:
                target_bounds[3] = sam.reference_edges(cropped_right_al)[5]

            resection = {3: None}
            if target_bounds[3] is not None:
                resection[3] = (target_bounds[3] - ti.cut_after) * resection_sign

            concatamer_als = []
            if resection[3] is not None and resection[3] < -10:
                # Non-physical resection might indicate concatamer. Check for
                # a relevant donor alignment.
                for al in donor_als:
                    possible_concatamer = sam.crop_al_to_query_int(al, right_switch_after + 1, np.inf)
                    if possible_concatamer is not None:
                        concatamer_als.append(possible_concatamer)

            insert_bounds =  sam.reference_edges(cropped_donor_al)

            left_edits = sam.edit_distance_in_query_interval(cropped_left_al, interval.get_covered(cropped_left_al), target_seq)
            right_edits = sam.edit_distance_in_query_interval(cropped_right_al, interval.get_covered(cropped_right_al), target_seq)
            middle_edits = sam.edit_distance_in_query_interval(cropped_donor_al, interval.get_covered(cropped_donor_al), donor_seq)
            total_edit_distance = left_edits + middle_edits + right_edits

            details = {
                'shared_HAs': self.shared_HAs(donor_al, edge_als),
                'gap_before': gap_before,
                'gap_after': gap_after,
                'edit_distance': total_edit_distance,
                'strand': sam.get_strand(donor_al),
                'donor_bounds': insert_bounds,
                'target_bounds': target_bounds,
                'length': abs(insert_bounds[5] - insert_bounds[3]) + 1,
                'donor_al': donor_al,
                'cropped_alignments': [cropped_left_al, cropped_donor_al, cropped_right_al],
                'full_alignments': [edge_als['left'], donor_al, edge_als['right']],
                'resection': resection,
                'concatamer_alignments': concatamer_als,
                'contains_donor_SNV': self.specific_to_donor(donor_al),
            }

            failures = []

            if gap_before > 5:
                failures.append(f'gap_before = {gap_before}')

            if gap_after > 5:
                failures.append(f'gap_after = {gap_after}')

            if total_edit_distance > 5:
                failures.append(f'edit_distance = {total_edit_distance}')

            if details['length'] < 5:
                failures.append(f'length = {details["length"]}')

            edit_distance_over_length = details['edit_distance'] / details['length']
            if edit_distance_over_length >= 0.1:
                failures.append(f'edit distance / length = {edit_distance_over_length}')

            if details['target_bounds'][5] is None:
                failures.append('target_bounds[5] is None')

            if details['target_bounds'][3] is None:
                failures.append('target_bounds[3] is None')

            if len(failures) > 0:
                details['failed'] = ' '.join(failures)

            possible_inserts.append(details)

        #def is_valid(d):
        #    return (d['gap_before'] <= 5 and 
        #            d['gap_after'] <= 5 and
        #            d['edit_distance'] <= 5 and
        #            d['length'] >= 5 and
        #            d['edit_distance'] / d['length'] < 0.05 and
        #            (d['target_bounds'][5] is not None and d['target_bounds'][3] is not None)
        #           )
        #valid = [d for d in possible_inserts if is_valid(d)]

        return possible_inserts

    @memoized_property
    def no_alignments_detected(self):
        return all(al.is_unmapped for al in self.alignments)

    @memoized_property
    def one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if indel.kind == 'D' and indel.length == 1]

    @memoized_property
    def indels_besides_one_base_deletions(self):
        return [indel for indel, near_cut in self.indels if not (indel.kind == 'D' and indel.length == 1)]
    
    def categorize(self):
        if self.no_alignments_detected:
            category = 'bad sequence'
            subcategory = 'no alignments detected'
            details = 'n/a'

        elif self.single_alignment_covers_read:
            if len(self.indels) == 0:
                if len(self.non_donor_SNVs) == 0:
                    if self.has_donor_SNV:
                        category = 'donor'
                        subcategory = 'clean'
                        outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                        details = str(outcome)
                    else:
                        if self.starts_at_primer:
                            category = 'wild type'
                            subcategory = 'clean'
                            details = 'n/a'
                        else:
                            category = 'truncation'
                            subcategory = 'clean'
                            details = str(self.target_reference_edges[5])

                else: # no indels but mismatches
                    # If extra bases synthesized during SD-MMEJ are the same length
                    # as the total resections, there will not be an indel, so check if
                    # the mismatches can be explained by SD-MMEJ.
                    if self.is_valid_SD_MMEJ:
                        category = 'SD-MMEJ'
                        subcategory = self.SD_MMEJ['kind']
                        details = self.SD_MMEJ_string
                        self.relevant_alignments = self.SD_MMEJ['alignments']
                    else:
                        if self.starts_at_primer:
                            category = 'mismatches'
                            subcategory = 'mismatches'
                            outcome = MismatchOutcome(self.non_donor_SNVs)
                            details = str(outcome)
                        else:
                            category = 'truncation'
                            subcategory = 'mismatches'
                            details = str(self.target_reference_edges[5])

            elif len(self.indels) == 1:
                indel, near_cut = self.indels[0]
                if len(self.donor_deletions_seen) == 1:
                    if len(self.non_donor_SNVs) == 0:
                            category = 'donor'
                            subcategory = 'clean'
                            outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            details = str(outcome)

                    else: # has non-donor SNVs
                        if self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                            self.relevant_alignments = self.SD_MMEJ['alignments']
                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'donor deletion plus at least one non-donor mismatch'

                else: # one indel, not a donor deletion
                    if self.has_any_SNV:
                        if self.has_donor_SNV and len(self.non_donor_SNVs) == 0:
                            if len(self.indels_besides_one_base_deletions) == 0:
                                # Interpret single base deletions in sequences that otherwise look
                                # like donor incorporation as synthesis errors in oligo donors.
                                category = 'donor'
                                subcategory = 'synthesis errors'
                                outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                details = str(outcome)

                                if len(self.donor_alignments) != 1:
                                    raise ValueError
                                
                                self.special_alignment = self.donor_alignments[0]

                            else:
                                if indel.kind == 'D':
                                    category = 'donor + deletion'
                                    subcategory = 'donor + deletion'
                                    HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                    deletion_outcome = DeletionOutcome(indel)
                                    outcome = HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)
                                    details = str(outcome)
                                elif indel.kind == 'I' and indel.length == 1:
                                    category = 'donor + insertion'
                                    subcategory = 'donor + insertion'
                                    HDR_outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                                    insertion_outcome = InsertionOutcome(indel)
                                    outcome = HDRPlusInsertionOutcome(HDR_outcome, insertion_outcome)
                                    details = str(outcome)
                                else:
                                    # insertion can be a mis-identified duplication of part of HA
                                    if self.donor_insertion is not None:
                                        category = 'donor insertion'
                                        subcategory, details, self.relevant_alignments, best_explanation = self.donor_insertion
                                        self.special_alignment = best_explanation['donor_al']
                                    else:
                                        category = 'uncategorized'
                                        subcategory = 'uncategorized'
                                        details = 'donor SNV with non-donor insertion'

                        elif self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                            self.relevant_alignments = self.SD_MMEJ['alignments']
                        else:
                            if len(self.non_donor_SNVs) == 1:
                                SNV_position = self.non_donor_SNVs.positions[0]

                                if indel.kind == 'D':
                                    deletion = indel
                                    if near_cut:
                                        mismatch_before = any(SNV_position == s - 1 for s in deletion.starts_ats)
                                        mismatch_after = any(SNV_position == e + 1 for e in deletion.ends_ats)
                                        if mismatch_before or mismatch_after:
                                            category = 'deletion + adjacent mismatch'
                                            subcategory = 'deletion + adjacent mismatch'
                                            deletion_outcome = DeletionOutcome(indel)
                                            mismatch_outcome = MismatchOutcome(self.non_donor_SNVs)
                                            outcome = DeletionPlusMismatchOutcome(deletion_outcome, mismatch_outcome)
                                            details = str(outcome)
                                        else:
                                            category = 'uncategorized'
                                            subcategory = 'uncategorized'
                                            details = 'deletion plus non-adjacent mismatch'
                                    else:
                                        category = 'uncategorized'
                                        subcategory = 'uncategorized'
                                        details = 'deletion far from cut plus mismatch'
                                else:
                                    category = 'uncategorized'
                                    subcategory = 'uncategorized'
                                    details = 'insertion plus one mismatch'
                            else:
                                category = 'uncategorized'
                                subcategory = 'uncategorized'
                                details = 'one indel plus more than one mismatch'

                    else: # no SNVs
                        if len(self.indels_near_cut) == 1:
                            if indel.kind == 'D':
                                category = 'deletion'
                                subcategory = 'clean'

                            elif indel.kind == 'I':
                                category = 'insertion'
                                subcategory = 'insertion'

                            details = self.indels_string

                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'indel far from cut'

            else: # more than one indel
                if self.has_any_SNV:
                    if self.has_donor_SNV and len(self.non_donor_SNVs) == 0:
                        if len(self.indels_besides_one_base_deletions) == 0:
                            category = 'donor'
                            subcategory = 'synthesis errors'
                            outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            details = str(outcome)

                            if len(self.donor_alignments) != 1:
                                raise ValueError
                            
                            self.special_alignment = self.donor_alignments[0]

                        elif self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                            self.relevant_alignments = self.SD_MMEJ['alignments']

                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'multiple indels plus at least one donor SNV'

                    elif self.is_valid_SD_MMEJ:
                        category = 'SD-MMEJ'
                        subcategory = self.SD_MMEJ['kind']
                        details = self.SD_MMEJ_string
                        self.relevant_alignments = self.SD_MMEJ['alignments']

                    else:
                        category = 'uncategorized'
                        subcategory = 'uncategorized'
                        details = 'multiple indels plus at least one mismatch'

                else: # no SNVs
                    if len(self.donor_deletions_seen) > 0:
                        if len(self.indels_besides_one_base_deletions) == 0:
                            category = 'donor'
                            subcategory = 'synthesis errors'
                            outcome = HDROutcome(self.donor_SNV_string, self.donor_deletions_seen)
                            details = str(outcome)
                        else:
                            if self.is_valid_SD_MMEJ:
                                category = 'SD-MMEJ'
                                subcategory = self.SD_MMEJ['kind']
                                details = self.SD_MMEJ_string
                                self.relevant_alignments = self.SD_MMEJ['alignments']
                            else:
                                category = 'uncategorized'
                                subcategory = 'uncategorized'
                                details = 'multiple indels including donor deletion'

                    else:
                        if self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                            self.relevant_alignments = self.SD_MMEJ['alignments']
                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'multiple indels'

        elif self.is_valid_SD_MMEJ:
            #TODO: if gap length is zero, classify as deletion
            category = 'SD-MMEJ'
            subcategory = self.SD_MMEJ['kind']
            details = self.SD_MMEJ_string
            self.relevant_alignments = self.SD_MMEJ['alignments']
        
        elif self.donor_insertion is not None:
            category = 'donor insertion'
            subcategory, details, self.relevant_alignments, best_explanation = self.donor_insertion
            self.special_alignment = best_explanation['donor_al']
        
        elif self.genomic_insertion is not None:
            category = 'genomic insertion'

            best_explanation = self.genomic_insertion[0]
            # which supplemental index did it come from?
            subcategory = best_explanation['chr'].split('_')[0]

            details = self.genomic_insertion_string

            self.relevant_alignments = best_explanation['full_alignments']
            self.special_alignment = best_explanation['cropped_genomic_alignment']

        elif self.phiX_alignments:
            category = 'phiX'
            subcategory = 'phiX'
            details = 'n/a'

        else:
            num_Ns = Counter(self.seq)['N']

            if num_Ns > 10:
                category = 'bad sequence'
                subcategory = 'too many Ns'
                details = str(num_Ns)

            elif self.Q30_fractions['all'] < 0.5:
                category = 'bad sequence'
                subcategory = 'low quality'
                fraction = self.Q30_fractions['all']
                details = f'{fraction:0.2f}'

            elif self.Q30_fractions['second_half'] < 0.5:
                category = 'bad sequence'
                subcategory = 'low quality tail'
                fraction = self.Q30_fractions['second_half']
                details = f'{fraction:0.2f}'
                
            elif self.longest_polyG >= 20:
                category = 'bad sequence'
                subcategory = 'long polyG'
                details = str(self.longest_polyG)

            else:
                category = 'uncategorized'
                subcategory = 'uncategorized'
                details = 'n/a'
                
                self.relevant_alignments = self.alignments

        return category, subcategory, details

    @memoized_property
    def longest_polyG(self):
        locations = utilities.homopolymer_lengths(self.seq, 'G')

        if locations:
            max_length = max(length for p, length in locations)
        else:
            max_length = 0

        return max_length

category_order = [
    ('wild type',
        ('clean',
        ),
    ),
    ('mismatches',
        ('mismatches',
        ),
    ),
    ('donor',
        ('clean',
         'short deletions',
         'deletion',
         'other',
         'collapsed',
        ),
    ),
    ('donor + deletion',
        ('donor + deletion',
        ),
    ),
    ('deletion',
        ('clean',
         'mismatch nearby',
        ),
    ),
    ('deletion + adjacent mismatch',
        ('deletion + adjacent mismatch',
        ),
    ),
    ('insertion',
        ('insertion',
        ),
    ),
    ('SD-MMEJ',
        ('loop-out',
         'snap-back',
         'multi-step',
        ),
    ),
    ('genomic insertion',
        ('hg19',
         'bosTau7',
        ),
    ),
    ('donor insertion',
        ('uses PAM-distal homology',
         'uses PAM-proximal homology',
         'neither homology arm used',
         'uses PAM-distal homology (no SNVs)',
         'uses PAM-proximal homology (no SNVs)',
         'neither homology arm used (no SNVs)',
         'uses both homology arms',
        ),
    ),
    ('truncation',
        ('clean',
         'mismatches',
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
    ('bad sequence',
        ('too many Ns',
         'long polyG',
         'low quality',
         'low quality tail',
         'no alignments detected',
        ),
    ), 
]

categories = [c for c, scs in category_order]
subcategories = dict(category_order)

def order(outcome):
    category, subcategory = outcome
    try:
        return (categories.index(category),
                subcategories[category].index(subcategory),
               )
    except:
        print(category, subcategory)
        raise

class DeletionOutcome():
    def __init__(self, deletion):
        self.deletion = deletion

    @classmethod
    def from_string(cls, details_string):
        deletion = DegenerateDeletion.from_string(details_string)
        return DeletionOutcome(deletion)

    def __str__(self):
        return str(self.deletion)

class InsertionOutcome():
    def __init__(self, insertion):
        self.insertion = insertion

    @classmethod
    def from_string(cls, details_string):
        insertion = DegenerateInsertion.from_string(details_string)
        return InsertionOutcome(insertion)

    def __str__(self):
        return str(self.insertion)

class HDROutcome():
    def __init__(self, donor_SNV_read_bases, donor_deletions):
        self.donor_SNV_read_bases = donor_SNV_read_bases
        self.donor_deletions = donor_deletions

    @classmethod
    def from_string(cls, details_string):
        SNV_string, donor_deletions_string = details_string.split(';', 1)
        if donor_deletions_string == '':
            deletions = []
        else:
            deletions = [DegenerateDeletion.from_string(s) for s in donor_deletions_string.split(';')]
        return HDROutcome(SNV_string, deletions)

    def __str__(self):
        donor_deletions_string = ';'.join(str(d) for d in self.donor_deletions)
        return f'{self.donor_SNV_read_bases};{donor_deletions_string}'

class HDRPlusDeletionOutcome():
    def __init__(self, HDR_outcome, deletion_outcome):
        self.HDR_outcome = HDR_outcome
        self.deletion_outcome = deletion_outcome
    
    @classmethod
    def from_string(cls, details_string):
        deletion_string, HDR_string = details_string.split(';', 1)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)
        HDR_outcome = HDROutcome.from_string(HDR_string)

        return HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.HDR_outcome}'

class HDRPlusInsertionOutcome():
    def __init__(self, HDR_outcome, insertion_outcome):
        self.HDR_outcome = HDR_outcome
        self.insertion_outcome = insertion_outcome
    
    @classmethod
    def from_string(cls, details_string):
        insertion_string, HDR_string = details_string.split(';', 1)
        insertion_outcome = InsertionOutcome.from_string(insertion_string)
        HDR_outcome = HDROutcome.from_string(HDR_string)

        return HDRPlusDeletionOutcome(HDR_outcome, insertion_outcome)

    def __str__(self):
        return f'{self.insertion_outcome};{self.HDR_outcome}'

class MismatchOutcome():
    def __init__(self, snvs):
        self.snvs = snvs

    @classmethod
    def from_string(cls, details_string):
        snvs = SNVs.from_string(details_string)
        return MismatchOutcome(snvs)

    def __str__(self):
        return str(self.snvs)

class DeletionPlusMismatchOutcome():
    def __init__(self, deletion_outcome, mismatch_outcome):
        self.deletion_outcome = deletion_outcome
        self.mismatch_outcome = mismatch_outcome

    @classmethod
    def from_string(cls, details_string):
        deletion_string, mismatch_string = details_string.split(';', 1)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)
        mismatch_outcome = MismatchOutcome.from_string(mismatch_string)
        return DeletionPlusMismatchOutcome(deletion_outcome, mismatch_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.mismatch_outcome}'
