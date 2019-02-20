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
    def split_target_alignments(self):
        split_als = []
        for al in self.target_alignments:
            split_als.extend(sam.split_at_deletions(al, 2))
        return split_als
    
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
    
    @memoized_property
    def parsimonious_target_alignments(self):
        als = interval.make_parsimonious(self.target_alignments)

        if len(als) == 2 and sam.get_strand(als[0]) == sam.get_strand(als[1]):
            upstream, downstream = sorted(als, key=lambda al: al.reference_start)

            merged = sam.merge_adjacent_alignments(upstream, downstream, self.target_info.reference_sequences)
            if merged is not None:
                als = [merged]

        exempt_if_overlaps = self.target_info.around_cuts(5)
        #split_als = []
        #for al in als:
        #    split_als.extend(sam.split_at_deletions(al, 2, exempt_if_overlaps))
        split_als = als
        
        return split_als

    @memoized_property
    def split_donor_alignments(self):
        split_als = []
        for al in self.donor_alignments:
            split_als.extend(sam.split_at_deletions(al, 2))

        return split_als
    
    @memoized_property
    def original_target_edge_alignments(self):
        return self.get_target_edge_alignments(self.parsimonious_target_alignments)
    
    @memoized_property
    def realigned_target_edge_alignments(self):
        return self.get_target_edge_alignments(self.realigned_target_alignments + self.short_edge_alignments)
    
    def get_target_edge_alignments(self, alignments):
        ''' Get target alignments that make it to the read edges. '''
        edge_alignments = {5: [], 3:[]}

        split_als = []
        for al in alignments:
            split_als.extend(sam.split_at_deletions(al, 2))

        for al in split_als:
            if sam.get_strand(al) != self.target_info.primers[3].strand:
                continue

            covered = interval.get_covered(al)
            
            if covered.start <= 2:
                edge_alignments[5].append(al)
            
            if covered.end == len(self.seq) - 1:
                edge_alignments[3].append(al)

        for edge in [5, 3]:
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
    
    def extends_to_primers(self, al):
        primers = self.target_info.primers
        return all(sam.overlaps_feature(al, primers[edge]) for edge in [5, 3])

    @memoized_property
    def single_alignment_covers_read(self):
        t_als = self.parsimonious_target_alignments
        
        if len(t_als) != 1:
            return False
        else:
            t_al = t_als[0]
            start, end = sam.query_interval(t_al)

            missing_from = {
                'start': start,
                'end': len(self.seq) - end - 1,
            }

            reaches_edge = {
                'start': missing_from['start'] <= 20,
                'end': missing_from['end'] <= 20 or self.extends_to_primers(t_al),
            }

            return reaches_edge['start'] and reaches_edge['end']

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
        SNPs = self.target_info.SNPs
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

    @memoized_property
    def donor_SNV_locii_summary(self):
        SNPs = self.target_info.SNPs
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
        return interval.get_disjoint_covered(self.parsimonious_target_alignments)
    
    @memoized_property
    def nonredundant_supplemental_alignments(self):
        nonredundant = []
        
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            novel_length = (covered - self.covered_from_target).total_length
            if novel_length > 15:
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

        full_refs = list(self.original_header.references) + [name for name, seq in new_targets]
        full_lengths = list(self.original_header.lengths) + [len(seq) for name, seq in new_targets]

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
            (ti.donor, ti.donor_sequence),
        ]

        stringent_als = sw.align_read(self.read, targets, 5, self.original_header,
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

    def seed_and_extend_on_target(self, query_start, query_end):
        query = self.seq_bytes
        target = self.target_info.target_sequence_bytes
        seed_locations = self.target_info.target_sequence_seed_locations
        header = self.original_header
        target_name = self.target_info.target
        return sw.seed_and_extend(query, target, seed_locations, query_start, query_end, header, target_name, self.name)
    
    def seed_and_extend_on_donor(self, query_start, query_end):
        query = self.seq_bytes
        donor = self.target_info.donor_sequence_bytes
        seed_locations = self.target_info.donor_sequence_seed_locations
        header = self.original_header
        donor_name = self.target_info.donor
        return sw.seed_and_extend(query, donor, seed_locations, query_start, query_end, header, donor_name, self.name)

    @memoized_property
    def perfect_edge_alignments(self):
        edge_als = {5: [], 3: []}
        for edge, start, end in [(5, 0, 20), (3, len(self.seq) - 20, len(self.seq))]:
            edge_als[edge] = self.seed_and_extend_on_target(start, end)
        
        longest_edge_als = {}
        for edge, als in edge_als.items():
            if len(als) == 0:
                longest_edge_als[edge] = None
            else:
                longest_edge_als[edge] = max(als, key=lambda al: al.query_alignment_length)
        
        whole_read = interval.Interval(0, len(self.seq) - 1)
        covered_from_edges = interval.get_disjoint_covered(longest_edge_als.values())
        uncovered = whole_read - covered_from_edges
        
        if uncovered.total_length == 0:
            gap_interval = None
        elif len(uncovered.intervals) > 1:
            # This shouldn't be possible since seeds start at each edge
            raise ValueError('disjoint gap', uncovered)
        else:
            gap_interval = uncovered.intervals[0]
        
        return longest_edge_als, gap_interval

    @memoized_property
    def perfect_edge_alignment_reference_edges(self):
        longest_edge_als, gap_interval = self.perfect_edge_alignments
        left_edge = sam.reference_edges(longest_edge_als[5])[3]
        right_edge = sam.reference_edges(longest_edge_als[3])[5]

        return left_edge, right_edge

    def reference_distances_from_perfect_edge_alignments(self, al):
        al_edges = sam.reference_edges(al)
        left_edge, right_edge = self.perfect_edge_alignment_reference_edges 
        return abs(left_edge - al_edges[5]), abs(right_edge - al_edges[3])

    def gap_covering_alignments(self, required_MH_start, required_MH_end):
        longest_edge_als, gap_interval = self.perfect_edge_alignments
        gap_query_start = gap_interval.start - required_MH_start
        gap_query_end = gap_interval.end + 1 + required_MH_end
        gap_covering_als = self.seed_and_extend_on_target(gap_query_start, gap_query_end)
        return gap_covering_als
    
    def partial_gap_perfect_alignments(self, required_MH_start, required_MH_end):
        def close_enough(al):
            return min(*self.reference_distances_from_perfect_edge_alignments(al)) < 100

        edge_als, gap_interval = self.perfect_edge_alignments
        if gap_interval is None:
            return [], []

        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end

        from_start_gap_als = []
        while (end > start) and not from_start_gap_als:
            end -= 1
            from_start_gap_als = self.seed_and_extend_on_target(start, end)
            from_start_gap_als = [al for al in from_start_gap_als if close_enough(al)]
            
        start = gap_interval.start - required_MH_start
        end = gap_interval.end + 1 + required_MH_end
        from_end_gap_als = []
        while (end > start) and not from_end_gap_als:
            start += 1
            from_end_gap_als = self.seed_and_extend_on_target(start, end)
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

        longest_edge_als, gap_interval = self.perfect_edge_alignments
        overlaps_cut = self.target_info.overlaps_cut

        if longest_edge_als[5] is None or longest_edge_als[3] is None:
            details['failed'] = 'missing edge alignment'
            return details

        details['edge alignments'] = longest_edge_als
        details['alignments'] = list(longest_edge_als.values())
        details['all alignments'] = list(longest_edge_als.values())
        
        if longest_edge_als[5] == longest_edge_als[3]:
            details['failed'] = 'perfect wild type'
            return details
        elif sam.get_strand(longest_edge_als[5]) != sam.get_strand(longest_edge_als[3]):
            details['failed'] = 'edges align to different strands'
            return details
        else:
            edge_als_strand = sam.get_strand(longest_edge_als[5])
        
        details['left edge'], details['right edge'] = self.perfect_edge_alignment_reference_edges

        # Require resection on both sides of the cut.
        for edge in [5, 3]:
            if overlaps_cut(longest_edge_als[edge]):
                details['failed'] = '{}\' edge alignment extends over cut'.format(edge)
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
            details['gap edges'] = sam.reference_edges(closest_gap_covering)

            if closest_gap_covering is not None:
                gap_covering_strand = sam.get_strand(closest_gap_covering)
                if gap_covering_strand == edge_als_strand:
                    details['kind'] = 'loop-out'
                else:
                    details['kind'] = 'snap-back'

            details['alignments'].append(closest_gap_covering)

            gap_covered = interval.get_covered(closest_gap_covering)
            edge_covered = {side: interval.get_covered(longest_edge_als[side]) for side in [5, 3]}
            homology_lengths = {side: len(gap_covered & edge_covered[side]) for side in [5, 3]}

            details['homology lengths'] = homology_lengths
            
        else:
            # Try to cover with multi-step.
            multi_step_als = self.multi_step_SD_MMEJ_gap_cover
            if multi_step_als is not None:
                for side in ['start', 'end']:
                    details['alignments'].extend(multi_step_als[side])
                details['kind'] = 'multi-step'
                details['gap edges'] = {5: 'PH', 3: 'PH'}
                details['homology lengths'] = {5: 'PH', 3: 'PH'}
            else:
                details['failed'] = 'no valid alignments cover gap' 
                return details

        return details

    @memoized_property
    def SD_MMEJ_slow(self):
        perfect_als = self.perfect_local_target_alignments

        details = {}

        # Find longest perfect alignments to the target that start at each edge of the read.
        edge_als = {5: [], 3: []}

        for al in perfect_als:
            covered = interval.get_covered(al)
            if covered.start <= 2:
                edge_als[5].append(al)
            if len(self.seq) - 1 - covered.end <= 2:
                edge_als[3].append(al)

        if len(edge_als[5]) == 0 or len(edge_als[3]) == 0:
            details['failed'] = 'missing edge alignment'
            return details
                
        longest_edge_als = {edge: max(als, key=lambda al: al.query_alignment_length) for edge, als in edge_als.items()}
        if longest_edge_als[5] == longest_edge_als[3]:
            details['failed'] = 'perfect wild type'
            return details
        elif sam.get_strand(longest_edge_als[5]) != sam.get_strand(longest_edge_als[3]):
            details['failed'] = 'edges align to different strands'
            return details
        else:
            edge_als_strand = sam.get_strand(longest_edge_als[5])

        details['edge alignments'] = longest_edge_als

        # Find perfect alignments to the target that cover the whole gap between the edge-containing alignments.
        whole_read = interval.Interval(2, len(self.seq) - 3)
        covered_from_edges = interval.get_disjoint_covered(longest_edge_als.values())
        gap = whole_read - covered_from_edges
        details['gap length'] = gap.total_length

        gap_covering_als = [al for al in perfect_als if (gap - interval.get_covered(al)).total_length == 0]
        if len(gap_covering_als) == 0:
            details['failed'] = 'no alignments cover gap' 
            return details

        min_distance = np.inf
        closest_gap_covering = None

        left_edge = sam.reference_edges(longest_edge_als[5])[3]
        right_edge = sam.reference_edges(longest_edge_als[3])[5]

        details['left edge'] = left_edge
        details['right edge'] = right_edge

        for al in gap_covering_als:
            al_edges = sam.reference_edges(al)
            distance = min(abs(left_edge - al_edges[5]), abs(right_edge - al_edges[3]))
            if (distance < 50) and (distance < min_distance):
                min_distance = distance
                closest_gap_covering = al

        if closest_gap_covering is None:
            details['failed'] = 'no valid alignments cover gap' 
            return details

        details['gap alignment'] = closest_gap_covering
        details['gap edges'] = sam.reference_edges(closest_gap_covering)

        if closest_gap_covering is not None:
            gap_covering_strand = sam.get_strand(closest_gap_covering)
            if gap_covering_strand == edge_als_strand:
                details['kind'] = 'loop-out'
            else:
                details['kind'] = 'snap-back'

        gap_covered = interval.get_covered(closest_gap_covering)
        edge_covered = {side: interval.get_covered(longest_edge_als[side]) for side in [5, 3]}
        homology_lengths = {side: len(gap_covered & edge_covered[side]) for side in [5, 3]}

        details.update({
            'alignments': list(longest_edge_als.values()) + [closest_gap_covering],
            'homology lengths': homology_lengths,
        })

        return details

    @memoized_property
    def is_valid_SD_MMEJ(self):
        return 'failed' not in self.SD_MMEJ

    @memoized_property
    def SD_MMEJ_string(self):
        fields = [
            self.SD_MMEJ['left edge'],
            self.SD_MMEJ['gap edges'][5],
            self.SD_MMEJ['gap edges'][3],
            self.SD_MMEJ['right edge'],
            self.SD_MMEJ['gap length'],
            self.SD_MMEJ['homology lengths'][5],
            self.SD_MMEJ['homology lengths'][3],
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
        valid = self.identify_genomic_insertions(self.original_target_edge_alignments)
        if len(valid) == 0:
            valid = self.identify_genomic_insertions(self.realigned_target_edge_alignments)

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

    def identify_genomic_insertions(self, edge_als):
        target_seq = self.target_info.target_sequence

        possible_insertions = []

        edge_covered = {}
        target_bounds = {}

        if edge_als[5] is None:
            edge_covered[5] = interval.Interval(-np.inf, -1)
            target_bounds[5] = None
        else:
            edge_covered[5] = interval.get_covered(edge_als[5])
            target_bounds[5] = sam.reference_edges(edge_als[5])[3]

            # If there is a target alignment at the 5' edge, insist that
            # it include the reverse primer.
            if not sam.overlaps_feature(edge_als[5], self.target_info.primers[3]):
                return []

        if edge_als[3] is None:
            edge_covered[3] = interval.Interval(len(self.seq), np.inf)
            target_bounds[3] = None
        else:
            edge_covered[3] = interval.get_covered(edge_als[3])
            target_bounds[3] = sam.reference_edges(edge_als[3])[5]

        for genomic_al in self.nonredundant_supplemental_alignments:
            genomic_covered = interval.get_covered(genomic_al)

            non_overlapping = genomic_covered - edge_covered[5] - edge_covered[3]
            if len(non_overlapping) == 0:
                continue

            left_overlap = edge_covered[5] & genomic_covered
            right_overlap = edge_covered[3] & genomic_covered
            
            target_bounds = {}
            genomic_bounds = {}
            
            if left_overlap:
                left_ceds = sam.cumulative_edit_distances(edge_als[5], target_seq, left_overlap, False)
                right_ceds = sam.cumulative_edit_distances(genomic_al, None, left_overlap, True)

                switch_after_edits = {
                    left_overlap.start - 1 : right_ceds[left_overlap.start],
                    left_overlap.end: left_ceds[left_overlap.end],
                }

                for q in range(left_overlap.start, left_overlap.end):
                    switch_after_edits[q] = left_ceds[q] + right_ceds[q + 1]

                left_min_edits = min(switch_after_edits.values())
                best_switch_points = [s for s, d in switch_after_edits.items() if d == left_min_edits]
                left_switch_after = max(best_switch_points)
                
                gap_before = 0
            else:
                left_min_edits = 0
                left_switch_after = edge_covered[5].end
                
                gap_before = genomic_covered.start - (edge_covered[5].end + 1)
                
            if right_overlap:
                left_ceds = sam.cumulative_edit_distances(genomic_al, None, right_overlap, False)
                right_ceds = sam.cumulative_edit_distances(edge_als[3], target_seq, right_overlap, True)

                switch_after_edits = {
                    right_overlap.start - 1 : right_ceds[right_overlap.start],
                    right_overlap.end: left_ceds[right_overlap.end],
                }

                for q in range(right_overlap.start, right_overlap.end):
                    switch_after_edits[q] = left_ceds[q] + right_ceds[q + 1]

                right_min_edits = min(switch_after_edits.values())
                best_switch_points = [s for s, d in switch_after_edits.items() if d == right_min_edits]
                right_switch_after = min(best_switch_points)
                
                gap_after = 0
            else:
                right_min_edits = 0
                right_switch_after = genomic_covered.end
                
                gap_after = (edge_covered[3].start - 1) - genomic_covered.end
                
            if edge_als[5] is None:
                cropped_left_al = None
                target_bounds[5] = None
            else:
                cropped_left_al = sam.crop_al_to_query_int(edge_als[5], -np.inf, left_switch_after)
                target_bounds[5] = sam.reference_edges(cropped_left_al)[3]

            if edge_als[3] is None:
                cropped_right_al = None
                target_bounds[3] = None
            else:
                cropped_right_al = sam.crop_al_to_query_int(edge_als[3], right_switch_after + 1, np.inf)
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
                'genomic_bounds': genomic_bounds,
                'target_bounds': target_bounds,
                'cropped_alignments': [al for al in [cropped_left_al, cropped_genomic_al, cropped_right_al] if al is not None],
                'full_alignments': [al for al in [edge_als[5], genomic_al, edge_als[3]] if al is not None],
            }
            
            possible_insertions.append(details)

        is_valid = lambda d: d['gap_before'] <= 5 and d['gap_after'] <= 5 and d['edit_distance'] <= 5 
        valid = [d for d in possible_insertions if is_valid(d)]

        return valid

    @memoized_property
    def donor_insertion(self):
        valid = self.identify_donor_insertions(self.original_target_edge_alignments, self.split_donor_alignments)
        if len(valid) == 0:
            valid = self.identify_donor_insertions(self.realigned_target_edge_alignments, self.realigned_donor_alignments)

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
    def donor_deletions_seen(self):
        seen = [d for d, _ in self.indels if d.kind == 'D' and d in self.target_info.donor_deletions]
        return seen

    @memoized_property
    def donor_deletions_string(self):
        return ';'.join(str(d) for d in self.donor_deletions_seen)
    
    def shared_HAs(self, donor_al, target_al):
        q_to_HA_offsets = defaultdict(lambda: defaultdict(set))

        for (al, which) in [(donor_al, 'donor'), (target_al, 'target')]:
            for side in [5, 3]:
                for q, ref_p in al.aligned_pairs:
                    if q is not None:
                        offset = self.target_info.HA_ref_p_to_offset[which, side].get(ref_p)

                        if offset is not None:
                            q_to_HA_offsets[sam.true_query_position(q, al)][side, offset].add(which)
                        
        shared = set()
        for q in q_to_HA_offsets:
            for side, offset in q_to_HA_offsets[q]:
                if len(q_to_HA_offsets[q][side, offset]) == 2:
                    shared.add(side)
                    
        return shared

    def identify_donor_insertions(self, edge_als, donor_als):
        ''' edge_als: dictionary of alignments that include the 5' and 3' edges of the rad
        '''
        ti = self.target_info

        cut_after = ti.cut_after

        # (5' edge of 3' target alignment  - cut_after) will be multiplied by
        # resection_sign to give how many bases were resected on the sequencing
        # primer distal side of the cut.
        if ti.cut_after >= ti.primers[3].start:
            resection_sign = 1
        else:
            resection_sign = -1

        target_seq = ti.target_sequence
        donor_seq = ti.donor_sequence

        edge_covered = {}
        if edge_als[5] is not None:
            cropped_to_cut = sam.crop_al_to_ref_int(edge_als[5], cut_after + 1, np.inf) 
            edge_covered[5] = interval.get_covered(cropped_to_cut)
        else:
            return []

        if edge_als[3] is not None:
            cropped_to_cut = sam.crop_al_to_ref_int(edge_als[3], -np.inf, cut_after) 
            edge_covered[3] = interval.get_covered(cropped_to_cut)
        else:
            return []

        covered_from_edges = interval.DisjointIntervals(edge_covered.values())

        possible_inserts = []

        for donor_al in donor_als:
            insert_covered = interval.get_covered(donor_al)
            novel_length = (insert_covered - covered_from_edges).total_length

            if novel_length > 0:
                insert_covered = interval.get_covered(donor_al)
                left_overlap = edge_covered[5] & insert_covered
                right_overlap = edge_covered[3] & insert_covered

                target_bounds = {}
                insert_bounds = {}

                if left_overlap:
                    left_ceds = sam.cumulative_edit_distances(edge_als[5], target_seq, left_overlap, False)
                    right_ceds = sam.cumulative_edit_distances(donor_al, donor_seq, left_overlap, True)

                    switch_after_edits = {
                        left_overlap.start - 1 : right_ceds[left_overlap.start],
                        left_overlap.end: left_ceds[left_overlap.end],
                    }

                    for q in range(left_overlap.start, left_overlap.end):
                        switch_after_edits[q] = left_ceds[q] + right_ceds[q + 1]

                    left_min_edits = min(switch_after_edits.values())
                    best_switch_points = [s for s, d in switch_after_edits.items() if d == left_min_edits]
                    left_switch_after = max(best_switch_points)

                    gap_before = 0
                else:
                    left_min_edits = 0
                    left_switch_after = edge_covered[5].end

                    gap_before = insert_covered.start - (edge_covered[5].end + 1)

                if right_overlap:
                    left_ceds = sam.cumulative_edit_distances(donor_al, donor_seq, right_overlap, False)
                    right_ceds = sam.cumulative_edit_distances(edge_als[3], target_seq, right_overlap, True)

                    switch_after_edits = {
                        right_overlap.start - 1 : right_ceds[right_overlap.start],
                        right_overlap.end: left_ceds[right_overlap.end],
                    }

                    for q in range(right_overlap.start, right_overlap.end):
                        switch_after_edits[q] = left_ceds[q] + right_ceds[q + 1]

                    right_min_edits = min(switch_after_edits.values())
                    best_switch_points = [s for s, d in switch_after_edits.items() if d == right_min_edits]
                    right_switch_after = min(best_switch_points)

                    gap_after = 0
                else:
                    right_min_edits = 0
                    right_switch_after = insert_covered.end

                    gap_after = (edge_covered[3].start - 1) - insert_covered.end

                cropped_left_al = sam.crop_al_to_query_int(edge_als[5], -np.inf, left_switch_after)
                cropped_right_al = sam.crop_al_to_query_int(edge_als[3], right_switch_after + 1, np.inf)

                cropped_donor_al = sam.crop_al_to_query_int(donor_al, left_switch_after + 1, right_switch_after)

                if cropped_donor_al is None or cropped_donor_al.is_unmapped:
                    continue

                if cropped_left_al is None:
                    target_bounds[5] = None
                else:
                    target_bounds[5] = sam.reference_edges(cropped_left_al)[3]

                if cropped_right_al is None:
                    target_bounds[3] = None
                else:
                    target_bounds[3] = sam.reference_edges(cropped_right_al)[5]

                resection = {5: None, 3: None}
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

                shared_HAs = {side: self.shared_HAs(donor_al, edge_als[side]) for side in [5, 3]}
                
                non_overlapping = insert_covered - edge_covered[5] - edge_covered[3]
                middle_edits = sam.edit_distance_in_query_interval(donor_al, non_overlapping, donor_seq)
                total_edit_distance = left_min_edits + middle_edits + right_min_edits

                details = {
                    'shared_HAs': shared_HAs,
                    'gap_before': gap_before,
                    'gap_after': gap_after,
                    'edit_distance': total_edit_distance,
                    'strand': sam.get_strand(donor_al),
                    'donor_bounds': insert_bounds,
                    'target_bounds': target_bounds,
                    'length': abs(insert_bounds[5] - insert_bounds[3]) + 1,
                    'cropped_alignments': [cropped_left_al, cropped_donor_al, cropped_right_al],
                    'full_alignments': [edge_als[5], donor_al, edge_als[3]],
                    'resection': resection,
                    'concatamer_alignments': concatamer_als,
                }

                possible_inserts.append(details)

        def is_valid(d):
            return (d['gap_before'] <= 5 and 
                    d['gap_after'] <= 5 and
                    d['edit_distance'] <= 5 and
                    d['length'] >= 5 and
                    d['edit_distance'] / d['length'] < 0.05 and
                    (d['target_bounds'][5] is not None and d['target_bounds'][3] is not None)
                   )
        valid = [d for d in possible_inserts if is_valid(d)]

        return valid

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
        self.relevant_alignments = self.parsimonious_target_alignments + self.donor_alignments

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
                        details = '{};{}'.format(self.donor_SNV_string, self.donor_deletions_string)
                    else:
                        category = 'wild type'
                        subcategory = 'clean'
                        details = 'n/a'

                else: # no indels but mismatches
                    # If extra bases synthesized during SD-MMEJ are the same length
                    # as the total resections, there will not be an indel, so check if
                    # the mismatches can be explained by SD-MMEJ.
                    if self.is_valid_SD_MMEJ:
                        category = 'SD-MMEJ'
                        subcategory = self.SD_MMEJ['kind']
                        details = self.SD_MMEJ_string
                    else:
                        category = 'mismatches'
                        subcategory = 'mismatches'
                        details = str(self.non_donor_SNVs)

            elif len(self.indels) == 1:
                if len(self.donor_deletions_seen) == 1:
                    if len(self.non_donor_SNVs) == 0:
                            category = 'donor'
                            subcategory = 'clean'
                            details = '{};{}'.format(self.donor_SNV_string, self.donor_deletions_string)

                    else: # has non-donor SNVs
                        if self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'donor deletion plus at least one non-donor mismatches'

                else: # one indel, not a donor deletion
                    if self.has_any_SNV:
                        if self.has_donor_SNV and len(self.non_donor_SNVs) == 0:
                            if len(self.indels_besides_one_base_deletions) == 0:
                                # Interpret single base deletions in sequences that otherwise look
                                # like donor incorporation as synthesis errors in oligo donors.
                                category = 'donor'
                                subcategory = 'synthesis errors'
                                details = '{};{}'.format(self.donor_SNV_string, self.donor_deletions_string)
                            else:
                                category = 'uncategorized'
                                subcategory = 'uncategorized'
                                details = 'donor SNV with non-donor indel'

                        elif self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'one indel plus at least one mismatch'

                    else: # no SNVs
                        if len(self.indels_near_cut) == 1:
                            indel = self.indels_near_cut[0]

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
                            details = '{};{}'.format(self.donor_SNV_string, self.donor_deletions_string)
                        elif self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
                        else:
                            category = 'uncategorized'
                            subcategory = 'uncategorized'
                            details = 'multiple indels plus at least one donor SNV'
                    elif self.is_valid_SD_MMEJ:
                        category = 'SD-MMEJ'
                        subcategory = self.SD_MMEJ['kind']
                        details = self.SD_MMEJ_string
                    else:
                        category = 'uncategorized'
                        subcategory = 'uncategorized'
                        details = 'multiple indels plus at least one mismatch'

                else: # no SNVs
                    if len(self.donor_deletions_seen) > 0:
                        if len(self.indels_besides_one_base_deletions) == 0:
                            category = 'donor'
                            subcategory = 'synthesis errors'
                            details = '{};{}'.format(self.donor_SNV_string, self.donor_deletions_string)
                        else:
                            if self.is_valid_SD_MMEJ:
                                category = 'SD-MMEJ'
                                subcategory = self.SD_MMEJ['kind']
                                details = self.SD_MMEJ_string
                            else:
                                category = 'uncategorized'
                                subcategory = 'uncategorized'
                                details = 'multiple indels including donor deletion'

                    else:
                        if self.is_valid_SD_MMEJ:
                            category = 'SD-MMEJ'
                            subcategory = self.SD_MMEJ['kind']
                            details = self.SD_MMEJ_string
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
            def details_to_string(details):
                fields = [
                    details['strand'],
                    details['target_bounds'][5],
                    details['target_bounds'][3],
                    details['gap_before'],
                    details['gap_after'],
                    details['edit_distance'],
                    details['length'],
                    details['donor_bounds'][5],
                    details['donor_bounds'][3],
                ]
                return ','.join(map(str, fields))

            best_explanation = self.donor_insertion[0]

            category = 'donor insertion'

            if len(best_explanation['shared_HAs'][5]) > 0:
                subcategory = '5\' homology'
            elif len(best_explanation['shared_HAs'][3]) > 0:
                subcategory = '3\' homology'
            else:
                subcategory = 'no homology'

            details = details_to_string(best_explanation)

            self.relevant_alignments = best_explanation['full_alignments'] + best_explanation['concatamer_alignments']

        elif self.genomic_insertion is not None:
            category = 'genomic insertion'

            best_explanation = self.genomic_insertion[0]
            # which supplemental index did it come from?
            subcategory = best_explanation['chr'].split('_')[0]

            details = self.genomic_insertion_string

            self.relevant_alignments = best_explanation['full_alignments']

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
                details = '{:0.2f}'.format(self.Q30_fractions['all'])

            elif self.Q30_fractions['second_half'] < 0.5:
                category = 'bad sequence'
                subcategory = 'low quality tail'
                details = '{:0.2f}'.format(self.Q30_fractions['second_half'])
                
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
        ('wild type',
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
        ),
    ),
    ('deletion',
        ('clean',
         'mismatch nearby',
        ),
    ),
    ('insertion',
        ('insertion',
        ),
    ),
    ('uncategorized',
        ('uncategorized',
        ),
    ),
    ('genomic insertion',
        ('hg19',
         'bosTau7',
        ),
    ),
    ('donor insertion',
        ('donor insertion',
        ),
    ),
    ('phiX',
        ('phiX',
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
    return (categories.index(category),
            subcategories[category].index(subcategory),
           )
