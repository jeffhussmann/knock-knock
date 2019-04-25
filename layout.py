from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

from sequencing import sam, interval, utilities, fastq, sw
from knockin.target_info import DegenerateDeletion, DegenerateInsertion

memoized_property = utilities.memoized_property
idx = pd.IndexSlice

class Layout(object):
    def __init__(self, alignments, target_info):
        self.target_info = target_info

        self.original_alignments = [al for al in alignments if not al.is_unmapped]

        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.qual = np.array(sam.get_original_qual(alignment))

        self.relevant_alignments = self.original_alignments

    @memoized_property
    def target_alignments(self):
        if self.seq is None:
            return []

        primers = self.target_info.primers_by_side_of_target
        target_seq_bytes = self.target_info.target_sequence_bytes

        original_als = [al for al in self.original_alignments if al.reference_name == self.target_info.target]

        processed_als = []

        for al in original_als:
            # Ignore alignments to the target that fall entirely outside the amplicon interval.
            # These should typically be considered genomic insertions and caught by supplementary alignments;
            # counting on target alignments to get them makes behavior dependant on the amount of flanking
            # sequence included around the amplicon.
            if not (self.target_info.amplicon_interval & sam.reference_interval(al)):
                continue

            query_interval = interval.get_covered(al)

            # Primers frequently contain 1 nt deletions, sometimes causing truncated alignments at the edges
            # of reads. If alignment ends close to but not at a read end, try to refine.
            extend_before = (0 < query_interval.start <= len(primers[5]) + 5)
            extend_after = (len(self.seq) - 1 - len(primers[3]) - 5 <= query_interval.end < len(self.seq) - 1)
            if extend_before or extend_after:
                al = sw.extend_repeatedly(al, target_seq_bytes, extend_before=extend_before, extend_after=extend_after)

            # Easier to reason about alignments if any that contain long insertions or deletions are split into multiple
            # alignments.
            split_at_dels = sam.split_at_deletions(al, 3)
            split_at_both = []
            for split_al in split_at_dels:
                split_at_both.extend(sam.split_at_large_insertions(split_al, 2))

            extended = [sw.extend_alignment(split_al, target_seq_bytes) for split_al in split_at_both]

            processed_als.extend(extended)

        # If processed alignments don't cover either edge, this typically means non-specific amplification.
        # Try to realign each uncovered edge to the relevant primer.
        existing_covered = interval.get_disjoint_covered(processed_als)

        possible_edge_als = []

        if existing_covered.start != 0:
            possible_edge_als.append(self.realign_edges_to_primers(5))

        if existing_covered.end != len(self.seq) - 1:
            possible_edge_als.append(self.realign_edges_to_primers(3))

        for edge_al in possible_edge_als:
            if edge_al is not None:
                new_covered = interval.get_covered(edge_al) - existing_covered
                # Only add the new alignment if it explains a substantial new amount of the read.
                if new_covered.total_length > 10:
                    processed_als.append(edge_al)
        
        return processed_als

    @memoized_property
    def donor_alignments(self):
        original_als = [al for al in self.original_alignments if al.reference_name == self.target_info.donor]
        processed_als = []

        for al in original_als:
            processed_als.extend(sam.split_at_deletions(al, 3))

        return processed_als

    @memoized_property
    def alignments(self):
        return self.target_alignments + self.donor_alignments
    
    @memoized_property
    def supplemental_alignments(self):
        return [al for al in self.original_alignments if al.reference_name not in self.target_info.reference_sequences]

    def categorize(self):
        details = 'n/a'

        if all(al.is_unmapped for al in self.alignments):
            category = 'malformed layout'
            subcategory = 'no alignments detected'

        elif self.extra_copy_of_primer:
            category = 'malformed layout'
            subcategory = 'extra copy of primer'

        elif self.missing_a_primer:
            category = 'malformed layout'
            subcategory = 'missing a primer'

        elif self.primer_strands[5] != self.primer_strands[3]:
            category = 'malformed layout'
            subcategory = 'primers not in same orientation'
        
        elif not self.primer_alignments_reach_edges:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'

        elif not self.has_integration:
            if self.indel_near_cut is not None:
                if len(self.indel_near_cut) > 1:
                    category = 'uncategorized'
                    subcategory = 'multiple indels near cut'
                else:
                    category = 'indel'
                    indel = self.indel_near_cut[0]
                    if indel.kind == 'D':
                        if indel.length < 50:
                            subcategory = 'deletion <50 nt'
                        else:
                            subcategory = 'deletion >=50 nt'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                details = self.indel_string
                self.relevant_alignments = self.target_alignments

            elif len(self.mismatches_near_cut) > 0:
                category = 'uncategorized'
                subcategory = 'mismatch(es) near cut'
                details = 'n/a'

            elif self.any_donor_specific_present:
                category = 'uncategorized'
                subcategory = 'donor specific present'
                details = 'n/a'

            else:
                category = 'WT'
                subcategory = 'WT'

        elif self.integration_summary == 'donor':
            if self.junction_summary_per_side[5] == 'HDR' and self.junction_summary_per_side[3] == 'HDR':
                category = 'HDR'
                subcategory = 'HDR'
                self.relevant_alignments = self.parsimonious_alignments
            else:
                category = 'misintegration'
                subcategory = '5\' {}, 3\' {}'.format(self.junction_summary_per_side[5], self.junction_summary_per_side[3])
                self.relevant_alignments = self.alignments

        elif self.integration_summary == 'concatamer':
            category = 'concatamer'
            subcategory = self.junction_summary

        elif self.nonspecific_amplification is not None:
            category = 'nonspecific amplification'
            subcategory = 'nonspecific amplification'
            details = 'n/a'
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonspecific_amplification

        elif self.genomic_insertion is not None:
            category = 'genomic insertion'
            subcategory = 'genomic insertion'
            details = 'n/a'

            self.relevant_alignments = self.parsimonious_target_alignments + self.min_edit_distance_genomic_insertions

        elif self.integration_summary in ['donor with indel', 'other', 'unexpected length', 'unexpected source']:
            category = 'uncategorized'
            subcategory = self.integration_summary

            self.relevant_alignments = self.parsimonious_alignments + self.supplemental_alignments + self.gap_alignments

        else:
            print(self.integration_summary)

        if self.strand == '-':
            self.relevant_alignments = [sam.flip_alignment(al) for al in self.relevant_alignments]

        return category, subcategory, details
    
    def categorize_no_donor(self):
        details = 'n/a'

        if all(al.is_unmapped for al in self.alignments):
            category = 'malformed layout'
            subcategory = 'no alignments detected'

        elif self.extra_copy_of_primer:
            category = 'malformed layout'
            subcategory = 'extra copy of primer'

        elif self.missing_a_primer:
            category = 'malformed layout'
            subcategory = 'missing a primer'

        elif self.primer_strands[5] != self.primer_strands[3]:
            category = 'malformed layout'
            subcategory = 'primers not in same orientation'
        
        elif not self.primer_alignments_reach_edges:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'

        else:
            if self.indel_near_cut is not None:
                category = 'indel'
                if len(self.indel_near_cut) > 1:
                    subcategory = 'complex indel'
                else:
                    indel = self.indel_near_cut[0]
                    if indel.kind == 'D':
                        if indel.length < 50:
                            subcategory = 'deletion <50 nt'
                        else:
                            subcategory = 'deletion >=50 nt'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                details = self.indel_string
            else:
                category = 'WT'
                subcategory = 'WT'

        return category, subcategory, details
    
    @memoized_property
    def read(self):
        if self.seq is None:
            return None
        else:
            return fastq.Read(self.name, self.seq, fastq.encode_sanger(self.qual))
    
    @memoized_property
    def whole_read_interval(self):
        return interval.Interval(0, len(self.seq) - 1)
    
    def realign_edges_to_primers(self, side):
        if self.seq is None:
            return []

        buffer_length = 5

        primer = self.target_info.primers_by_side_of_target[side]

        if side == 5:
            amplicon_slice = idx[primer.start:primer.end + 1 + buffer_length]
            read_slice = idx[:len(primer) + buffer_length]
            alignment_type = 'fixed_start'
        else:
            amplicon_slice = idx[primer.start - buffer_length:primer.end + 1]
            read_slice = idx[-(len(primer) + buffer_length):]
            alignment_type = 'fixed_end'

        amplicon_side_seq = self.target_info.target_sequence[amplicon_slice]

        read = self.read[read_slice]
        soft_clip_length = len(self.seq) - len(read)

        temp_header = pysam.AlignmentHeader.from_references(['amplicon_side'], [len(amplicon_side_seq)])

        targets = [('amplicon_side', amplicon_side_seq)]
        als = sw.align_read(read, targets, 5, temp_header,
                            alignment_type=alignment_type,
                            max_alignments_per_target=1,
                            both_directions=False,
                            min_score_ratio=0,
                            )

        edge_al = None

        if len(als) > 0:
            al = als[0]
            if side == 5:
                offset = primer.start
                new_cigar = al.cigar + [(sam.BAM_CSOFT_CLIP, soft_clip_length)]
                primer_query_interval = interval.Interval(0, len(primer) - 1)
            else:
                offset = primer.start - buffer_length
                new_cigar = [(sam.BAM_CSOFT_CLIP, soft_clip_length)] + al.cigar
                # can't just use buffer_length as start in case read is shorter than primer + buffer_length
                primer_query_interval = interval.Interval(len(read) - len(primer), np.inf)
            
            edits_in_primer = sam.edit_distance_in_query_interval(al, primer_query_interval, ref_seq=amplicon_side_seq)
            if edits_in_primer <= 5:
                al.reference_start = al.reference_start + offset
                al.cigar = sam.collapse_soft_clip_blocks(new_cigar)
                al.query_sequence = self.seq
                al.query_qualities = self.qual
                al_dict = al.to_dict()
                al_dict['ref_name'] = self.target_info.target
                edge_al = pysam.AlignedSegment.from_dict(al_dict, self.target_info.header)
            
        return edge_al

    @memoized_property
    def all_primer_alignments(self):
        ''' Get all alignments that contain the amplicon primers. '''
        als = {}
        for side, primer in self.target_info.primers_by_side_of_target.items():
            # Prefer to have the primers annotated on the strand they anneal to,
            # so don't require strand match here.
            als[side] = [al for al in self.parsimonious_alignments if sam.overlaps_feature(al, primer, False)]

        return als

    @memoized_property
    def gap_alignments(self):
        seq_bytes = self.seq.encode()
        gap_als = []

        gap = self.gap_between_primer_alignments
        if len(gap) >= 4:
            for on in ['target', 'donor']:
                aligner = self.target_info.seed_and_extender[on]
                als = aligner(seq_bytes, gap.start, gap.end, self.name)
                als = sorted(als, key=lambda al: al.query_alignment_length, reverse=True)
                # For same reasoning as in target_alignments, only consider als that overlap the amplicon interval.
                if on == 'target':
                    als = [al for al in als if (self.target_info.amplicon_interval & sam.reference_interval(al))]

                gap_als.extend(als[:10])

        return gap_als

    @memoized_property
    def extra_copy_of_primer(self):
        ''' Check if too many alignments containing either primer were found. '''
        return any(len(als) > 1 for side, als in self.all_primer_alignments.items())
    
    @memoized_property
    def missing_a_primer(self):
        ''' Check if either primer was not found in an alignments. '''
        return any(len(als) == 0 for side, als in self.all_primer_alignments.items())
        
    @memoized_property
    def primer_alignments(self):
        ''' Get the single alignment containing each primer. '''
        if self.extra_copy_of_primer or self.missing_a_primer:
            return None
        else:
            return {side: als[0] for side, als in self.all_primer_alignments.items()}
        
    @memoized_property
    def primer_strands(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        if self.primer_alignments is None:
            return None
        else:
            return {side: sam.get_strand(al) for side, al in self.primer_alignments.items()}
    
    @memoized_property
    def strand(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        if self.primer_strands is None:
            return None
        else:
            strands = set(self.primer_strands.values())
            if len(strands) > 1:
                return None
            else:
                return strands.pop()

    @memoized_property
    def covered_by_primers_alignments(self):
        ''' How much of the read is covered by alignments containing the primers? '''
        if self.strand is None:
            # primer-containing alignments mapped to opposite strands
            return None
        elif self.primer_alignments is None:
            return None
        else:
            return interval.get_disjoint_covered([self.primer_alignments[5], self.primer_alignments[3]])

    @memoized_property
    def primer_alignments_reach_edges(self):
        if self.covered_by_primers_alignments is None:
            return False
        else:
            return (self.covered_by_primers_alignments.start <= 10 and
                    len(self.seq) - self.covered_by_primers_alignments.end <= 10
                   )

    @memoized_property
    def any_donor_specific_present(self):
        ti = self.target_info
        donor_specific = ti.features[ti.donor, ti.donor_specific]
        return any(sam.overlaps_feature(al, donor_specific, False) for al in self.donor_alignments)

    @memoized_property
    def single_merged_primer_alignment(self):
        ''' If the alignments from the primers are adjacent to each other on the query, merge them. '''

        primer_als = self.primer_alignments
        ref_seqs = self.target_info.reference_sequences

        if self.primer_alignments_reach_edges:
            merged = sam.merge_adjacent_alignments(primer_als[5], primer_als[3], ref_seqs)
        else:
            merged = None

        return merged
    
    @memoized_property
    def has_integration(self):
        return self.primer_alignments_reach_edges and (self.single_merged_primer_alignment is None)

    @memoized_property
    def mismatches_near_cut(self):
        merged_primer_al = self.single_merged_primer_alignment
        if merged_primer_al is None:
            return []
        else:
            mismatches = []
            tuples = sam.aligned_tuples(merged_primer_al, self.target_info.target_sequence)
            for true_read_i, read_b, ref_i, ref_b, qual in tuples:
                if ref_i is not None and true_read_i is not None and ref_i in self.near_cut_intervals:
                    if read_b != ref_b:
                        mismatches.append(ref_i)

            return mismatches

    @memoized_property
    def indel_near_cut(self):
        d = self.largest_deletion_near_cut
        i = self.largest_insertion_near_cut

        if d is None:
            d_length = 0
        else:
            d_length = d.length

        if i is None:
            i_length = 0
        else:
            i_length = i.length

        if d_length == 0 and i_length == 0:
            scar = None
        elif d_length > i_length:
            scar = [d]
        elif i_length > d_length:
            scar = [i]
        else:
            scar = [d, i]

        return scar

    @memoized_property
    def near_cut_intervals(self):
        return self.target_info.around_cuts(10)

    @memoized_property
    def largest_deletion_near_cut(self):
        dels = [indel for indel in self.indels if indel.kind == 'D']

        near_cut = []
        for deletion in dels:
            del_interval = interval.Interval(min(deletion.starts_ats), max(deletion.starts_ats) + deletion.length - 1)
            if del_interval & self.near_cut_intervals:
                near_cut.append(deletion)

        if near_cut:
            largest = max(near_cut, key=lambda d: d.length)
            largest = self.target_info.expand_degenerate_indel(largest)
        else:
            largest = None

        return largest

    @memoized_property
    def largest_insertion_near_cut(self):
        insertions = [indel for indel in self.indels if indel.kind == 'I']

        near_cut = [ins for ins in insertions if any(sa in self.near_cut_intervals for sa in ins.starts_afters)]

        if near_cut:
            largest = max(near_cut, key=lambda ins: len(ins.seqs[0]))
            largest = self.target_info.expand_degenerate_indel(largest)
        else:
            largest = None

        return largest
    
    @memoized_property
    def indel_string(self):
        if self.indel_near_cut is None:
            indel_string = None
        else:
            indel_string = ' '.join(map(str, self.indel_near_cut))

        return indel_string

    @memoized_property
    def parsimonious_alignments(self):
        return interval.make_parsimonious(self.alignments)

    @memoized_property
    def parsimonious_target_alignments(self):
        return [al for al in self.parsimonious_alignments if al.reference_name == self.target_info.target]

    @memoized_property
    def parsimonious_donor_alignments(self):
        return [al for al in self.parsimonious_alignments if al.reference_name == self.target_info.donor]

    @memoized_property
    def closest_donor_alignment_to_edge(self):
        ''' Identify the alignments to the donor closest to edge of the read
        that has the PAM-proximal and PAM-distal amplicon primer. '''
        donor_als = self.donor_alignments

        if self.strand is None or len(donor_als) == 0:
            closest = {5: None, 3: None}
        else:
            closest = {}

            left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
            right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)

            if self.strand == '+':
                closest[5] = left_most
                closest[3] = right_most
            else:
                closest[5] = right_most
                closest[3] = left_most

        return closest

    @memoized_property
    def clean_handoff(self):
        ''' Check if target sequence cleanly transitions to donor sequence at
        each junction between the two, with one full length copy of the relevant
        homology arm and no large indels (i.e. not from sequencing errors) near
        the internal edge.
        '''
        if len(self.donor_alignments) == 0 or self.primer_alignments is None:
            return {5: False, 3: False}

        from_primer = self.primer_alignments
        HAs = self.target_info.homology_arms
        closest_donor = self.closest_donor_alignment_to_edge

        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            # The donor doesn't share homology arms with the target.
            return {5: False, 3: False}

        target_contains_full_arm = {
            5: HAs[5]['target'].end - from_primer[5].reference_end <= 10,
            3: from_primer[3].reference_start - HAs[3]['target'].start <= 10,
        }

        donor_contains_arm_external = {
            5: closest_donor[5].reference_start - HAs[5]['donor'].start <= 10,
            3: HAs[3]['donor'].end - (closest_donor[3].reference_end - 1) <= 10,
        }

        donor_contains_arm_internal = {
            5: closest_donor[5].reference_end - 1 - HAs[5]['donor'].end >= 20,
            3: HAs[3]['donor'].start - closest_donor[3].reference_start >= 20,
        }

        donor_contains_full_arm = {
            side: donor_contains_arm_external[side] and donor_contains_arm_internal[side]
            for side in [5, 3]
        }
            
        target_external_edge_query = {
            5: sam.closest_query_position(HAs[5]['target'].start, from_primer[5]),
            3: sam.closest_query_position(HAs[3]['target'].end, from_primer[3]),
        }
        
        donor_external_edge_query = {
            5: sam.closest_query_position(HAs[5]['donor'].start, closest_donor[5]),
            3: sam.closest_query_position(HAs[3]['donor'].end, closest_donor[3]),
        }

        arm_overlaps = {
            side: abs(target_external_edge_query[side] - donor_external_edge_query[side]) <= 10
            for side in [5, 3]
        }

        junction = {
            5: HAs[5]['donor'].end,
            3: HAs[3]['donor'].start,
        }

        max_indel_near_junction = {
            side: max_indel_nearby(closest_donor[side], junction[side], 10)
            for side in [5, 3]
        }

        clean_handoff = {}
        for side in [5, 3]:
            clean_handoff[side] = (
                target_contains_full_arm[side] and
                donor_contains_full_arm[side] and
                arm_overlaps[side] and
                max_indel_near_junction[side] <= 2
            )

        return clean_handoff
    
    @memoized_property
    def edge_q(self):
        ''' Where in the query are the edges of the integration? '''
        if self.strand == '+':
            edge_q = {
                5: self.integration_interval.start,
                3: self.integration_interval.end,
            }
        else:
            edge_q = {
                5: self.integration_interval.end,
                3: self.integration_interval.start,
            }
        return edge_q

    @memoized_property
    def edge_r(self):
        ''' i don't understand this. certainly needs to be renamed '''
        edge_r = {
            5: [],
            3: [],
        }

        for al in self.parsimonious_donor_alignments:
            cropped = sam.crop_al_to_query_int(al, self.integration_interval.start, self.integration_interval.end)
            start = cropped.reference_start
            end = cropped.reference_end - 1
            # NOTE: testing if this needs to be strand-specific
            edge_r[5].append(start)
            edge_r[3].append(end)

        edge_r[5] = min(edge_r[5])
        edge_r[3] = max(edge_r[3])

        return edge_r

    @memoized_property
    def donor_relative_to_arm(self):
        ''' How much of the donor is integrated relative to the edges of the HAs? '''
        HAs = self.target_info.homology_arms

        # convention: positive if there is extra in the integration, negative if truncated
        relative_to_arm = {
            'internal': {
                5: (HAs[5]['donor'].end + 1) - self.edge_r[5],
                3: self.edge_r[3] - (HAs[3]['donor'].start - 1),
            },
            'external': {
                5: HAs[5]['donor'].start - self.edge_r[5],
                3: self.edge_r[3] - HAs[3]['donor'].end,
            },
        }

        return relative_to_arm
    
    @memoized_property
    def donor_relative_to_cut(self):
        ''' Distance on query between base aligned to donor before/after cut
        and start of target alignment.
        This doesn't appear to be used.
        '''
        to_cut = {
            5: None,
            3: None,
        }

        ti = self.target_info

        try:
            donor_edge = {
                5: ti.features[ti.donor, "5' edge"].start,
                3: ti.features[ti.donor, "3' edge"].start,
            }
        except KeyError:
            return to_cut

        for side in [5, 3]:
            if self.edge_r[side] is not None:
                to_cut[side] = self.edge_r[side] - donor_edge[side]

        return to_cut

    @memoized_property
    def donor_integration_contains_full_HA(self):
        HAs = self.target_info.homology_arms
        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            return {5: False, 3: False}

        full_HA = {}
        for side in [5, 3]:
            offset = self.donor_relative_to_arm['external'][side]
            
            full_HA[side] = offset > 0

        return full_HA

    @memoized_property
    def integration_interval(self):
        ''' because cut site might not exactly coincide with boundary between
        HAs, the relevant part of query to call integration depends on whether
        a clean HDR handoff is detected at each edge '''
        if not self.has_integration:
            return None

        HAs = self.target_info.homology_arms
        cut_after = self.target_info.cut_after

        flanking_al = {}
        mask_start = {5: -np.inf}
        mask_end = {3: np.inf}
        for side in [5, 3]:
            if self.clean_handoff[side]:
                flanking_al[side] = self.closest_donor_alignment_to_edge[side]
            else:
                flanking_al[side] = self.primer_alignments[side]

        if self.clean_handoff[5]:
            mask_end[5] = HAs[5]['donor'].end
        else:
            mask_end[5] = cut_after

        if self.clean_handoff[3]:
            mask_start[3] = HAs[3]['donor'].start
        else:
            mask_start[3] = cut_after + 1

        covered = {
            side: sam.crop_al_to_ref_int(flanking_al[side], mask_start[side], mask_end[side])
            for side in [5, 3]
        }

        disjoint_covered = interval.get_disjoint_covered([covered[5], covered[3]])

        return interval.Interval(disjoint_covered[0].end + 1, disjoint_covered[-1].start - 1)

    @memoized_property
    def gap_between_primer_alignments(self):
        if self.primer_alignments[5] is None or self.primer_alignments[3] is None:
            return interval.Interval.empty()

        left_covered = interval.get_covered(self.primer_alignments[5])
        right_covered = interval.get_covered(self.primer_alignments[3])
        gap = interval.Interval(left_covered.start, right_covered.end) - left_covered - right_covered
        
        return gap

    @memoized_property
    def target_to_at_least_cut(self):
        cut_after = self.target_info.cut_after
        primer_als = self.primer_alignments

        target_to_at_least_cut = {
            5: primer_als[5].reference_end - 1 >= cut_after,
            3: primer_als[3].reference_start <= (cut_after + 1),
        }

        return target_to_at_least_cut

    @memoized_property
    def junction_summary_per_side(self):
        per_side = {}

        for side in [5, 3]:
            if self.clean_handoff[side]:
                per_side[side] = 'HDR'
            elif self.donor_integration_contains_full_HA[side]:
                per_side[side] = 'NHEJ'
            else:
                per_side[side] = 'truncated'

        return per_side
                
    @memoized_property
    def junction_summary(self):
        per_side = self.junction_summary_per_side

        if (per_side[5] == 'HDR' and
            per_side[3] == 'HDR'):

            summary = 'HDR'

        elif (per_side[5] == 'NHEJ' and
              per_side[3] == 'HDR'):

            summary = "5' NHEJ"
        
        elif (per_side[5] == 'HDR' and
              per_side[3] == 'NHEJ'):

            summary = "3' NHEJ"
        
        elif (per_side[5] == 'NHEJ' and
              per_side[3] == 'NHEJ'):

            summary = "5' and 3' NHEJ"

        else:
            summary = 'uncategorized'

        return summary

    @memoized_property
    def e_coli_integration(self):
        assert self.has_integration

        e_coli_alignments = [al for al in self.parsimonious_alignments if al.reference_name == 'e_coli_K12']

        int_start = self.integration_interval.start
        int_end = self.integration_interval.end

        if len(self.parsimonious_donor_alignments) == 0:
            if len(e_coli_alignments) == 1:
                covered = interval.get_covered(e_coli_alignments[0])
                if covered.start - int_start <= 10 and int_end - covered.end <= 10:
                    return True

        return False

    @memoized_property
    def flipped_donor(self):
        return any(sam.get_strand(al) != self.strand for al in self.parsimonious_donor_alignments)
    
    @memoized_property
    def messy_junction_description(self):
        fields = []
        for side in [5, 3]:
            if self.junction_summary_per_side[side] == 'uncategorized':
                #if self.donor_relative_to_arm['internal'][side] < 0:
                #    fields += ["{0}' truncated".format(side)]
                #elif self.donor_relative_to_arm['internal'][side] > 0:
                #    fields += ["{0}' extended".format(side)]
                pass

        if len(fields) > 0:
            description = ', '.join(fields)
        else:
            description = 'uncategorized'

        return description

    @memoized_property
    def integration_summary(self):
        if len(self.parsimonious_donor_alignments) == 0:
            summary = 'other'

        elif len(self.parsimonious_donor_alignments) == 1:
            donor_al = self.parsimonious_donor_alignments[0]
            max_indel_length = sam.max_block_length(donor_al, {sam.BAM_CDEL, sam.BAM_CINS})
            if max_indel_length > 1:
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
        HAs = self.target_info.homology_arms
        p_donor_als = self.parsimonious_donor_alignments

        if len(p_donor_als) <= 1:
            return 0

        # TEMPORARY
        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            # The donor doesn't share homology arms with the target.
            return 0

        if self.strand == '+':
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

            overlap_slightly = len(before_int & after_int) <= 2
            adjacent = interval.are_adjacent(before_int, after_int)

            missing_before = HAs[3]['donor'].end - (before.reference_end - 1)
            missing_after = after.reference_start - HAs[5]['donor'].start

            clean = (adjacent or overlap_slightly) and (missing_before <= 0) and (missing_after <= 0)

            junctions_clean.append(clean)

        if all(junctions_clean):
            return len(junctions_clean) + 1
        else:
            return 0
    
    @memoized_property
    def indels(self):
        indels = []

        al = self.single_merged_primer_alignment

        if al is not None:
            for i, (kind, length) in enumerate(al.cigar):
                if kind == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before

                    indel = DegenerateDeletion([starts_at], length)

                elif kind == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    indel = DegenerateInsertion([starts_after], [insertion])
                    
                else:
                    continue

                indels.append(indel)

        return indels

    @memoized_property
    def genomic_insertion(self):
        min_gap_length = 10
        gap = self.gap_between_primer_alignments
        
        covered_by_normal = interval.get_disjoint_covered(self.alignments)
        unexplained_gap = gap - covered_by_normal

        if unexplained_gap.total_length < min_gap_length:
            return None
        elif self.gap_alignments:
            # gap aligns to the target in the amplicon region
            return None
        else:
            relevant_ref_names = []

            for genome_name, header in self.target_info.supplemental_headers.items():
                for ref_name in header.references:
                    full_ref_name = f'{genome_name}_{ref_name}'
                    relevant_ref_names.append(full_ref_name)

            covering_als = []
            for al in self.supplemental_alignments:
                covered = interval.get_covered(al)
                if (gap - covered).total_length <= 1:
                    edit_distance = sam.edit_distance_in_query_interval(al, gap)
                    error_rate = edit_distance / len(gap)
                    if error_rate < 0.1:
                        covering_als.append(al)
                    
            if len(covering_als) == 0:
                covering_als = None

            return covering_als

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
    def nonspecific_amplification(self):
        if not self.primer_alignments_reach_edges:
            return None

        whole_read = interval.Interval(0, len(self.seq) - 1)
        not_primer_length = {}
        primer_interval = {}

        # If alignments from the primers extend substantially into the read,
        # don't consider this nonspecific amplification. 
        for target_side in [5, 3]:
            al = self.primer_alignments[target_side]
            primer = self.target_info.primers_by_side_of_target[target_side]
            just_primer_al = sam.crop_al_to_ref_int(al, primer.start, primer.end)
            start, end = sam.query_interval(just_primer_al)
            if (target_side == 5 and self.strand == '+') or (target_side == 3 and self.strand == '-'):
                read_side = 'left'
                primer_interval[read_side] = interval.Interval(0, end)
            elif (target_side == 3 and self.strand == '+') or (target_side == 5 and self.strand == '-'):
                read_side = 'right'
                primer_interval[read_side] = interval.Interval(start, len(self.seq) - 1)

            not_primer_interval = whole_read - primer_interval[read_side]
            not_primer_al = sam.crop_al_to_query_int(al, not_primer_interval.start, not_primer_interval.end)
            if not_primer_al is None:
                not_primer_length[read_side] = 0
            else:
                not_primer_length[read_side] = not_primer_al.query_alignment_length

        if not_primer_length['left'] >= 20 or not_primer_length['right'] >= 20:
            return None

        need_to_cover = whole_read - primer_interval['left'] - primer_interval['right']

        covering_als = []
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            if len(need_to_cover - covered) == 0:
                covering_als.append(al)
                
        if len(covering_als) == 0:
            covering_als = None
            
        return covering_als
    
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

category_order = [
    ('WT',
        ('WT',
        ),
    ),
    ('indel',
        ('insertion',
         'deletion',
         'deletion <50 nt',
         'deletion >=50 nt',
         'complex indel',
        ),
    ),
    ('HDR',
        ('HDR',
        ),
    ),
    ('concatamer',
        ('HDR',
         '5\' NHEJ',
         '3\' NHEJ',
         '5\' and 3\' NHEJ',
         'uncategorized',
        ),
    ),
    ('misintegration',
        ("5' HDR, 3' NHEJ",
         "5' NHEJ, 3' HDR",
         "5' HDR, 3' truncated",
         "5' truncated, 3' HDR",
         "5' NHEJ, 3' truncated",
         "5' truncated, 3' NHEJ",
         "5' NHEJ, 3' NHEJ",
         "5' truncated, 3' truncated",
        ),
    ),
    ('nonspecific amplification',
        ('nonspecific amplification',
        ),
    ),
    ('genomic insertion',
        ('genomic insertion',
        ),
    ),
    ('uncategorized',
        ('uncategorized',
         'donor with indel',
         'mismatch(es) near cut',
         'multiple indels near cut',
         'donor specific present',
         'other',
        ),
    ),
    ('unexpected source',
        ('flipped',
         'e coli',
         'uncategorized',
        ),
    ),
    ('malformed layout',
        ('extra copy of primer',
         'missing a primer',
         'primer far from read edge',
         'primers not in same orientation',
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