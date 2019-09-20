import itertools
import re
import html
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

from hits import sam, interval, utilities, fastq, sw
from .target_info import DegenerateDeletion, DegenerateInsertion

memoized_property = utilities.memoized_property
idx = pd.IndexSlice

class Layout(object):
    def __init__(self, alignments, target_info, mode='illumina'):
        self.mode = mode
        if mode == 'illumina':
            self.indel_size_to_split_at = 3
            self.max_indel_allowed_in_donor = 1
        elif mode == 'pacbio':
            self.indel_size_to_split_at = 5
            self.max_indel_allowed_in_donor = 3

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
            split_at_dels = sam.split_at_deletions(al, self.indel_size_to_split_at)
            split_at_both = []
            for split_al in split_at_dels:
                split_at_both.extend(sam.split_at_large_insertions(split_al, self.indel_size_to_split_at))

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
        if self.target_info.donor is None:
            return []

        original_als = [al for al in self.original_alignments if al.reference_name == self.target_info.donor]
        split_at_both = []

        for al in original_als:
            split_at_dels = sam.split_at_deletions(al, self.indel_size_to_split_at)
            for split_al in split_at_dels:
                split_at_both.extend(sam.split_at_large_insertions(split_al, self.indel_size_to_split_at))

        return split_at_both
    
    @memoized_property
    def nonhomologous_donor_alignments(self):
        if self.target_info.nonhomologous_donor is None:
            return []

        original_als = [al for al in self.original_alignments if al.reference_name == self.target_info.nonhomologous_donor]
        processed_als = []

        for al in original_als:
            processed_als.extend(sam.split_at_deletions(al, self.indel_size_to_split_at))

        return processed_als

    @memoized_property
    def nonredundant_supplemental_alignments(self):
        primary_als = self.alignments + self.nonhomologous_donor_alignments + self.extra_alignments
        covered = interval.get_disjoint_covered(primary_als)

        supp_als_to_keep = []

        for al in self.supplemental_alignments:
            if interval.get_covered(al) - covered:
                supp_als_to_keep.append(al)

        supp_als_to_keep = sorted(supp_als_to_keep, key=lambda al: al.query_alignment_length, reverse=True)
        return supp_als_to_keep

    @memoized_property
    def nonredundant_halfbell_alignments(self):
        primary_als = self.alignments + self.nonhomologous_donor_alignments
        covered = interval.get_disjoint_covered(primary_als)

        halfbell_als = [al for al in self.original_alignments if al.reference_name.startswith('halfbell_v2')]
        
        als_to_keep = []

        for al in halfbell_als:
            if interval.get_covered(al) - covered:
                als_to_keep.append(al)

        return als_to_keep

    @memoized_property
    def alignments(self):
        return self.target_alignments + self.donor_alignments
    
    @memoized_property
    def supplemental_alignments(self):
        als = [al for al in self.original_alignments if al.reference_name not in self.target_info.reference_sequences]

        # For performance reasons, cap the number of alignments considered, prioritizing
        # alignments that explain more matched bases.
        def priority(al):
            return al.query_alignment_length - al.get_tag('NM')

        best_als = sorted(als, key=priority, reverse=True)[:100]

        split_als = []
        for al in best_als:
            split_als.extend(sam.split_at_large_insertions(al, 10))

        return split_als

    @memoized_property
    def extra_alignments(self):
        ti = self.target_info
        extra_ref_names = {n for n in ti.reference_sequences if n not in [ti.target, ti.donor]}
        als = [al for al in self.original_alignments if al.reference_name in extra_ref_names]
        return als

    @property
    def whole_read(self):
        return interval.Interval(0, len(self.seq) - 1)

    def categorize(self):
        if self.target_info.donor is None and self.target_info.nonhomologous_donor is None:
           c, s, d = self.categorize_no_donor()
        else:
            c, s, d = self.categorize_with_donor()
        
        if self.strand == '-':
            self.relevant_alignments = [sam.flip_alignment(al) for al in self.relevant_alignments]

        return c, s, d

    def categorize_with_donor(self):
        details = 'n/a'
        
        if len(self.seq) <= 50:
            category = 'malformed layout'
            subcategory = 'too short'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif all(al.is_unmapped for al in self.alignments):
            category = 'malformed layout'
            subcategory = 'no alignments detected'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.extra_copy_of_primer:
            category = 'malformed layout'
            subcategory = 'extra copy of primer'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.missing_a_primer:
            category = 'malformed layout'
            subcategory = 'missing a primer'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.primer_strands[5] != self.primer_strands[3]:
            category = 'malformed layout'
            subcategory = 'primers not in same orientation'
            self.relevant_alignments = self.uncategorized_relevant_alignments
        
        elif not self.primer_alignments_reach_edges:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif not self.has_integration:
            if self.indel_near_cut is not None:
                if len(self.indel_near_cut) > 1:
                    category = 'uncategorized'
                    subcategory = 'multiple indels near cut'
                else:
                    category = 'simple indel'
                    indel = self.indel_near_cut[0]
                    if indel.kind == 'D':
                        if indel.length < 50:
                            subcategory = 'deletion <50 nt'
                        else:
                            subcategory = 'deletion >=50 nt'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                details = self.indel_string
                self.relevant_alignments = self.parsimonious_target_alignments

            elif len(self.mismatches_near_cut) > 0:
                category = 'uncategorized'
                subcategory = 'mismatch(es) near cut'
                details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

            else:
                category = 'WT'
                subcategory = 'WT'
                self.relevant_alignments = self.parsimonious_target_alignments

        elif self.integration_summary == 'donor':
            junctions = set(self.junction_summary_per_side.values())
            if junctions == set(['HDR']):
                category = 'HDR'
                subcategory = 'HDR'
                self.relevant_alignments = self.parsimonious_and_gap_alignments
            else:
                subcategory = f'5\' {self.junction_summary_per_side[5]}, 3\' {self.junction_summary_per_side[3]}'
                if 'blunt' in junctions:
                    category = 'blunt misintegration'
                elif junctions == set(['imperfect', 'HDR']):
                    category = 'incomplete HDR'
                    details = str(self.donor_microhomology)
                else:
                    if self.gap_covered_by_target_alignment:
                        category = 'complex indel'
                        subcategory = 'templated insertion'
                        details = 'n/a'
                    else:
                        category = 'complex misintegration'

                self.relevant_alignments = self.uncategorized_relevant_alignments
        
        # TODO: check here for HA extensions into donor specific
        elif self.gap_covered_by_target_alignment:
            category = 'complex indel'
            subcategory = 'templated insertion'
            details = 'n/a'
            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.integration_interval.total_length <= 5:
            if self.target_to_at_least_cut[5] and self.target_to_at_least_cut[3]:
                category = 'simple indel'
                subcategory = 'insertion'
            else:
                category = 'complex indel'
                subcategory = 'deletion plus insertion'
            details = 'n/a'
            self.relevant_alignments = self.parsimonious_and_gap_alignments

        elif self.integration_summary == 'concatamer':
            category = 'concatenated misintegration'
            subcategory = self.junction_summary
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.nonhomologous_donor_integration is not None:
            category = 'non-homologous donor'
            subcategory = 'simple'
            details = 'n/a'
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonhomologous_donor_alignments

        elif self.nonspecific_amplification is not None:
            category = 'malformed layout'
            subcategory = 'nonspecific amplification'
            details = 'n/a'
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonspecific_amplification

        elif self.genomic_insertion is not None:
            category = 'genomic insertion'
            subcategory = 'genomic insertion'
            details = 'n/a'

            self.relevant_alignments = self.parsimonious_target_alignments + self.min_edit_distance_genomic_insertions

        elif self.partial_nonhomologous_donor_integration is not None:
            category = 'non-homologous donor'
            subcategory = 'complex'
            details = 'n/a'
            
            self.relevant_alignments = self.parsimonious_target_alignments + self.nonhomologous_donor_alignments + self.nonredundant_supplemental_alignments
        
        elif self.any_donor_specific_present:
            category = 'complex misintegration'
            subcategory = 'other'
            details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.integration_summary in ['donor with indel', 'other', 'unexpected length', 'unexpected source']:
            category = 'uncategorized'
            subcategory = self.integration_summary

            self.relevant_alignments = self.uncategorized_relevant_alignments

        else:
            print(self.integration_summary)

        return category, subcategory, details
    
    def categorize_no_donor(self):
        details = 'n/a'

        if len(self.seq) <= 50:
            category = 'malformed layout'
            subcategory = 'too short'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif all(al.is_unmapped for al in self.alignments):
            category = 'malformed layout'
            subcategory = 'no alignments detected'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.extra_copy_of_primer:
            category = 'malformed layout'
            subcategory = 'extra copy of primer'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.missing_a_primer:
            category = 'malformed layout'
            subcategory = 'missing a primer'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.primer_strands[5] != self.primer_strands[3]:
            category = 'malformed layout'
            subcategory = 'primers not in same orientation'
            self.relevant_alignments = self.uncategorized_relevant_alignments
        
        elif not self.primer_alignments_reach_edges:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.single_merged_primer_alignment is not None:
            num_indels = len(self.all_indels_near_cuts)
            if num_indels > 0:
                if num_indels > 1:
                    category = 'complex indel'
                    subcategory = 'multiple indels'
                else:
                    category = 'simple indel'
                    indel = self.indel_near_cut[0]
                    if indel.kind == 'D':
                        if indel.length < 50:
                            subcategory = 'deletion <50 nt'
                        else:
                            subcategory = 'deletion >=50 nt'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                # Split at every indel
                if self.mode == 'illumina':
                    split_at_both = []

                    for al in self.parsimonious_target_alignments:
                        split_at_dels = sam.split_at_deletions(al, 1)
                        for split_al in split_at_dels:
                            split_at_both.extend(sam.split_at_large_insertions(split_al, 1))
                    self.relevant_alignments = split_at_both
                else:
                    self.relevant_alignments = self.parsimonious_target_alignments

                details = ' '.join(map(str, self.all_indels_near_cuts))
            else:
                category = 'WT'
                subcategory = 'WT'
                self.relevant_alignments = self.parsimonious_target_alignments
        
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

        else:
            category = 'uncategorized'
            subcategory = 'uncategorized'
            details = 'n/a'

        return category, subcategory, details
    
    @memoized_property
    def read(self):
        if self.seq is None:
            return None
        else:
            return fastq.Read(self.name, self.seq, fastq.encode_sanger(self.qual))
    
    def realign_edges_to_primers(self, read_side):
        if self.seq is None:
            return []

        buffer_length = 5

        target_seq = self.target_info.target_sequence

        edge_als = []

        for amplicon_side in [5, 3]:
            primer = self.target_info.primers_by_side_of_target[amplicon_side]

            if amplicon_side == 5:
                amplicon_slice = idx[primer.start:primer.end + 1 + buffer_length]
            else:
                amplicon_slice = idx[primer.start - buffer_length:primer.end + 1]

            amplicon_side_seq = target_seq[amplicon_slice]

            if read_side == 5:
                read_slice = idx[:len(primer) + buffer_length]
            else:
                read_slice = idx[-(len(primer) + buffer_length):]

            if amplicon_side == 5:
                alignment_type = 'fixed_start'
            else:
                alignment_type = 'fixed_end'

            read = self.read[read_slice]
            if amplicon_side != read_side:
                read = read.reverse_complement()

            soft_clip_length = len(self.seq) - len(read)

            targets = [('amplicon_side', amplicon_side_seq)]
            temp_header = pysam.AlignmentHeader.from_references([n for n, s in targets], [len(s) for n, s in targets])

            als = sw.align_read(read, targets, 5, temp_header,
                                alignment_type=alignment_type,
                                max_alignments_per_target=1,
                                both_directions=False,
                                min_score_ratio=0,
                            )

            if len(als) > 0:
                al = als[0]
                if amplicon_side == 5:
                    ref_start_offset = primer.start
                    new_cigar = al.cigar + [(sam.BAM_CSOFT_CLIP, soft_clip_length)]
                else:
                    ref_start_offset = primer.start - buffer_length
                    new_cigar = [(sam.BAM_CSOFT_CLIP, soft_clip_length)] + al.cigar

                if read_side == 5:
                    primer_query_interval = interval.Interval(0, len(primer) - 1)
                elif read_side == 3:
                    # can't just use buffer_length as start in case read is shorter than primer + buffer_length
                    primer_query_interval = interval.Interval(len(read) - len(primer), np.inf)

                if amplicon_side != read_side:
                    al = sam.flip_alignment(al)

                edits_in_primer = sam.edit_distance_in_query_interval(al, primer_query_interval, ref_seq=amplicon_side_seq)
                if edits_in_primer <= 5:
                    al.reference_start = al.reference_start + ref_start_offset
                    al.cigar = sam.collapse_soft_clip_blocks(new_cigar)
                    if al.is_reverse:
                        seq = utilities.reverse_complement(self.seq)
                        qual = self.qual[::-1]
                    else:
                        seq = self.seq
                        qual = self.qual
                    al.query_sequence = seq
                    al.query_qualities = qual
                    al_dict = al.to_dict()
                    al_dict['ref_name'] = self.target_info.target
                    edge_al = pysam.AlignedSegment.from_dict(al_dict, self.target_info.header)
                    edge_als.append(edge_al)

        if len(edge_als) != 1:
            edge_al = None
        else:
            edge_al = edge_als[0]
            
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
        gap_als = []

        gap = self.gap_between_primer_alignments
        if len(gap) >= 4:
            seq_bytes = self.seq.encode()
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
    def possibly_imperfect_gap_alignments(self):
        gap_als = []
        if self.integration_interval.total_length >= 10:
            for al in self.target_alignments + self.donor_alignments:
                if (self.integration_interval - interval.get_covered(al)).total_length <= 2:
                    gap_als.append(al)

        return gap_als

    @memoized_property
    def gap_covered_by_target_alignment(self):
        all_gap_als = self.gap_alignments + self.possibly_imperfect_gap_alignments
        return any(al.reference_name == self.target_info.target for al in all_gap_als)

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
        primer_als = {5: None, 3: None}
        for side in [5, 3]:
            if len(self.all_primer_alignments[side]) == 1:
                primer_als[side] = self.all_primer_alignments[side][0]

        return primer_als
        
    @memoized_property
    def primer_strands(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        strands = {5: None, 3: None}
        for side in [5, 3]:
            al = self.primer_alignments[side]
            if al is not None:
                strands[side] = sam.get_strand(al)
        return strands
    
    @memoized_property
    def strand(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        strands = set(self.primer_strands.values())

        if None in strands:
            strands.remove(None)

        if len(strands) != 1:
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
        if ti.donor is None:
            return False
        else:
            donor_specific = ti.features[ti.donor, ti.donor_specific]
            als_to_check = self.parsimonious_and_gap_alignments
            return any(sam.overlaps_feature(al, donor_specific, False) for al in als_to_check)

    @memoized_property
    def single_merged_primer_alignment(self):
        ''' If the alignments from the primers are adjacent to each other on the query, merge them.
        If there is only one expected cut site, only try to perform a single merge. If there are
        more than one cut sites, merge all target alignments. '''

        primer_als = self.primer_alignments
        ref_seqs = self.target_info.reference_sequences

        if len(self.target_info.sgRNAs) <= 1:
            if primer_als[5] is not None and primer_als[3] is not None:
                merged = sam.merge_adjacent_alignments(primer_als[5], primer_als[3], ref_seqs)
            else:
                merged = None
        else:
            merged = sam.merge_multiple_adjacent_alignments(self.parsimonious_target_alignments, ref_seqs)
            if merged is not None:
                primers = self.target_info.primers_by_side_of_target.values()
                # Prefer to have the primers annotated on the strand they anneal to,
                # so don't require strand match here.
                reaches_primers = all(sam.overlaps_feature(merged, primer, False) for primer in primers)
                if not reaches_primers:
                    merged = None

        return merged
    
    @memoized_property
    def has_integration(self):
        covered = self.covered_by_primers_alignments
        start_covered = covered is not None and covered.start <= 10
        if not start_covered:
            return False
        else:
            if self.single_merged_primer_alignment is None:
                return True
            else:
                return False

    @memoized_property
    def mismatches_near_cut(self):
        merged_primer_al = self.single_merged_primer_alignment
        if merged_primer_al is None:
            return []
        else:
            mismatches = []
            tuples = sam.aligned_tuples(merged_primer_al, self.target_info.target_sequence)
            for true_read_i, read_b, ref_i, ref_b, qual in tuples:
                if ref_i is not None and true_read_i is not None:
                    if read_b != ref_b and ref_i in self.near_cut_intervals:
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
    def all_indels_near_cuts(self):
        indels_near_cuts = []
        for indel in self.indels:
            if indel.kind == 'D':
                d = indel
                d_interval = interval.Interval(min(d.starts_ats), max(d.starts_ats) + d.length - 1)
                if d_interval & self.near_cut_intervals:
                    indels_near_cuts.append(d)
            elif indel.kind == 'I':
                ins = indel
                if any(sa in self.near_cut_intervals for sa in ins.starts_afters):
                    indels_near_cuts.append(ins)

        return indels_near_cuts

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
    def parsimonious_and_gap_alignments(self):
        ''' identification of gap_alignments requires further processing of parsimonious alignments '''
        return sam.make_nonredundant(interval.make_parsimonious(self.parsimonious_alignments + self.gap_alignments))

    @memoized_property
    def parsimonious_target_alignments(self):
        return [al for al in self.parsimonious_and_gap_alignments if al.reference_name == self.target_info.target]

    @memoized_property
    def parsimonious_donor_alignments(self):
        return [al for al in self.parsimonious_and_gap_alignments if al.reference_name == self.target_info.donor]

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

        if closest_donor[5] is None and closest_donor[3] is None:
            return {5: False, 3: False}

        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            # The donor doesn't share homology arms with the target.
            return {5: False, 3: False}

        target_contains_full_arm = {
            5: (HAs[5]['target'].end - from_primer[5].reference_end <= 10
                if from_primer[5] is not None else False),
            3: (from_primer[3].reference_start - HAs[3]['target'].start <= 10
                if from_primer[3] is not None else False),
        }

        donor_contains_arm_external = {
            5: closest_donor[5].reference_start - HAs[5]['donor'].start <= 10,
            3: HAs[3]['donor'].end - (closest_donor[3].reference_end - 1) <= 10,
        }

        # Dilemma here: insisting on 20 nts past the edge of the HA filter out mismatch-containing
        # false positives at the expense of short, error-free true positives. Need to incorporate
        # check for mismatches.
        donor_contains_arm_internal = {
            5: closest_donor[5].reference_end - 1 - HAs[5]['donor'].end >= 0,
            3: HAs[3]['donor'].start - closest_donor[3].reference_start >= 0,
        }

        donor_past_HA = {
            5: sam.crop_al_to_ref_int(closest_donor[5], HAs[5]['donor'].end + 1, np.inf),
            3: sam.crop_al_to_ref_int(closest_donor[3], 0, HAs[3]['donor'].start - 1),
        }
        
        donor_past_HA_length = {
            side: donor_past_HA[side].query_alignment_length if donor_past_HA[side] is not None else 0
            for side in [5, 3]
        }
        
        donor_past_HA_edit_distance = {
            side: sam.total_edit_distance(donor_past_HA[side], self.target_info.donor_sequence)
            for side in [5, 3]
        }

        donor_contains_full_arm = {
            side: donor_contains_arm_external[side] and \
                  donor_contains_arm_internal[side] and \
                  donor_past_HA_length[side] >= 5 and \
                  donor_past_HA_edit_distance[side] / donor_past_HA_length[side] < 0.1
            for side in [5, 3]
        }
            
        target_external_edge_query = {
            5: (sam.closest_query_position(HAs[5]['target'].start, from_primer[5])
                if from_primer[5] is not None else None),
            3: (sam.closest_query_position(HAs[3]['target'].end, from_primer[3])
                if from_primer[3] is not None else None),
        }
        
        donor_external_edge_query = {
            5: sam.closest_query_position(HAs[5]['donor'].start, closest_donor[5]),
            3: sam.closest_query_position(HAs[3]['donor'].end, closest_donor[3]),
        }

        arm_overlaps = {
            side: (abs(target_external_edge_query[side] - donor_external_edge_query[side]) <= 10
                   if target_external_edge_query[side] is not None else False)
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
    def donor_relative_to_arm(self):
        ''' How much of the donor is integrated relative to the edges of the HAs? '''
        HAs = self.target_info.homology_arms

        # convention: positive if there is extra in the integration, negative if truncated
        relative_to_arm = {
            'internal': {
                5: ((HAs[5]['donor'].end + 1) - self.edge_r[5]
                    if self.edge_r[5] is not None else None),
                3: (self.edge_r[3] - (HAs[3]['donor'].start - 1)
                    if self.edge_r[3] is not None else None),
            },
            'external': {
                5: (HAs[5]['donor'].start - self.edge_r[5]
                    if self.edge_r[5] is not None else None),
                3: (self.edge_r[3] - HAs[3]['donor'].end
                    if self.edge_r[3] is not None else None),
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
    def donor_integration_is_blunt(self):
        donor_length = len(self.target_info.donor_sequence)

        is_blunt = {
            5: self.edge_r[5] is not None and self.edge_r[5] <= 1,
            3: self.edge_r[3] is not None and self.edge_r[3] >= donor_length - 2,
        }

        return is_blunt

    @memoized_property
    def donor_integration_contains_full_HA(self):
        HAs = self.target_info.homology_arms
        if 'donor' not in HAs[5] or 'donor' not in HAs[3]:
            return {5: False, 3: False}
        if not self.has_integration:
            return {5: False, 3: False}

        full_HA = {}
        for side in [5, 3]:
            offset = self.donor_relative_to_arm['external'][side]
            
            full_HA[side] = offset is not None and offset >= 0

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
            side: (sam.crop_al_to_ref_int(flanking_al[side], mask_start[side], mask_end[side])
                   if flanking_al[side] is not None else None
                  )
            for side in [5, 3]
        }

        if self.strand == '+':
            if covered[5] is not None:
                start = interval.get_covered(covered[5]).end + 1
            else:
                start = 0

            if covered[3] is not None:
                end = interval.get_covered(covered[3]).start - 1
            else:
                end = len(self.seq) - 1

        elif self.strand == '-':
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
        if self.primer_alignments[5] is None or self.primer_alignments[3] is None or self.strand is None:
            return interval.Interval.empty()

        left_covered = interval.get_covered(self.primer_alignments[5])
        right_covered = interval.get_covered(self.primer_alignments[3])
        if self.strand == '+':
            between_primers = interval.Interval(left_covered.start, right_covered.end)
        elif self.strand == '-':
            between_primers = interval.Interval(right_covered.start, left_covered.end)

        gap = between_primers - left_covered - right_covered
        
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
        integration_donor_als = []
        for al in self.parsimonious_donor_alignments:
            covered = interval.get_covered(al)
            if (self.integration_interval - covered).total_length == 0:
                # If a single donor al covers the whole integration, use just it.
                integration_donor_als = [al]
                break
            else:
                covered_integration = self.integration_interval & interval.get_covered(al)
                # Ignore als that barely extend past the homology arms.
                if len(covered_integration) >= 5:
                    integration_donor_als.append(al)

        if len(integration_donor_als) == 0:
            summary = 'other'

        elif len(integration_donor_als) == 1:
            covered = interval.get_disjoint_covered(self.parsimonious_target_alignments + integration_donor_als)
            uncovered_length = len(self.seq) - covered.total_length
            if uncovered_length > 10:
                summary = 'other'
            else:
                donor_al = integration_donor_als[0]
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
                if (gap - covered).total_length <= 3:
                    edit_distance = sam.edit_distance_in_query_interval(al, gap)
                    error_rate = edit_distance / len(gap)
                    if error_rate < 0.1:
                        covering_als.append(al)
                    
            if len(covering_als) == 0:
                covering_als = None

            return covering_als
        
    @memoized_property
    def one_sided_covering_als(self):
        all_covering_als = {
            'nonspecific_amplification': None,
            'genomic_insertion': None,
            'h': None,
            'nh': None,
        }
        
        if self.strand == '+':
            primer_al = self.primer_alignments[5]
        elif self.strand == '-':
            primer_al = self.primer_alignments[3]
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
        
        covered_by_normal = interval.get_disjoint_covered(self.alignments)
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

        if self.strand is None:
            return not_primer_length

        for target_side in [5, 3]:
            if (target_side == 5 and self.strand == '+') or (target_side == 3 and self.strand == '-'):
                read_side = 'left'
            elif (target_side == 3 and self.strand == '+') or (target_side == 5 and self.strand == '-'):
                read_side = 'right'

            al = self.primer_alignments[target_side]
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

        if self.strand is None:
            return primer_interval

        for target_side in [5, 3]:
            if (target_side == 5 and self.strand == '+') or (target_side == 3 and self.strand == '-'):
                read_side = 'left'
            elif (target_side == 3 and self.strand == '+') or (target_side == 5 and self.strand == '-'):
                read_side = 'right'

            al = self.primer_alignments[target_side]
            if al is None:
                primer_interval[read_side] = None
                continue

            primer = self.target_info.primers_by_side_of_target[target_side]
            just_primer_al = sam.crop_al_to_ref_int(al, primer.start, primer.end)
            start, end = sam.query_interval(just_primer_al)
            if read_side == 'left':
                primer_interval[read_side] = interval.Interval(0, end)
            elif read_side == 'right':
                primer_interval[read_side] = interval.Interval(start, len(self.seq) - 1)

        return primer_interval

    @memoized_property
    def nonspecific_amplification(self):
        if not self.primer_alignments_reach_edges:
            return None

        not_primer_length = self.extra_query_in_primer_als
        primer_interval = self.just_primer_interval

        # If alignments from the primers extend substantially into the read,
        # don't consider this nonspecific amplification. 

        if not_primer_length['left'] >= 20 or not_primer_length['right'] >= 20:
            return None

        need_to_cover = self.whole_read - primer_interval['left'] - primer_interval['right']

        covering_als = []
        for al in self.supplemental_alignments:
            covered = interval.get_covered(al)
            if len(need_to_cover - covered) == 0:
                covering_als.append(al)
                
        if len(covering_als) == 0:
            covering_als = None
            
        return covering_als

    @memoized_property
    def uncategorized_relevant_alignments(self):
        sources = [
            self.parsimonious_and_gap_alignments,
            self.nonhomologous_donor_alignments,
            self.extra_alignments,
        ]
        flattened = [al for source in sources for al in source]
        parsimonious = interval.make_parsimonious(flattened)

        covered = interval.get_disjoint_covered(parsimonious)
        supp_als = []

        def novel_length(supp_al):
            return (interval.get_covered(supp_al) - covered).total_length

        supp_als = interval.make_parsimonious(self.nonredundant_supplemental_alignments)
        supp_als = sorted(supp_als, key=novel_length, reverse=True)[:10]

        final = parsimonious + supp_als

        return final

    @memoized_property
    def donor_microhomology(self):
        if self.junction_summary_per_side == {5: 'HDR', 3: 'imperfect'}:
            target_al = self.primer_alignments[3]
        elif self.junction_summary_per_side == {5: 'imperfect', 3: 'HDR'}:
            target_al = self.primer_alignments[5]
        else:
            target_al = None
            
        if len(self.parsimonious_donor_alignments) == 1:
            donor_al = self.parsimonious_donor_alignments[0]
        else:
            donor_al = None
            
        if target_al is None or donor_al is None:
            MH_nts = None
        else:
            als = {
                'target': target_al,
                'donor': donor_al,
            }

            covered = {k: interval.get_covered(v) for k, v in als.items()}
            sides = {
                'left': min(covered, key=covered.get),
                'right': max(covered, key=covered.get),
            }
            
            initial_overlap = covered[sides['left']] & covered[sides['right']]
                
            if initial_overlap:
                # Trim back mismatches or indels in or near the overlap.
                mismatch_buffer_length = 5
                
                bad_read_ps = {
                    'left': set(),
                    'right': set(),
                }

                for side in ['left', 'right']:
                    al = als[sides[side]]
                    for read_p, *rest in get_mismatch_info(al, self.target_info):
                        bad_read_ps[side].add(read_p)

                    for kind, info in get_indel_info(al):
                        if kind == 'deletion':
                            read_p, length = info
                            bad_read_ps[side].add(read_p)
                        elif kind == 'insertion':
                            starts_at, ends_at = info
                            bad_read_ps[side].update([starts_at, ends_at])

                covered_trimmed = {}

                left_buffer_start = initial_overlap.start - mismatch_buffer_length

                left_illegal_ps = [p for p in bad_read_ps['left'] if p >= left_buffer_start]

                if left_illegal_ps:
                    old_start = covered[sides['left']].start
                    new_end = int(np.floor(min(left_illegal_ps))) - 1
                    covered_trimmed['left'] = interval.Interval(old_start, new_end)
                else:
                    covered_trimmed['left'] = covered[sides['left']]

                right_buffer_end = initial_overlap.end + mismatch_buffer_length

                right_illegal_ps = [p for p in bad_read_ps['right'] if p <= right_buffer_end]

                if right_illegal_ps:
                    new_start = int(np.ceil(max(right_illegal_ps))) + 1
                    old_end = covered[sides['right']].end
                    covered_trimmed['right'] = interval.Interval(new_start, old_end)
                else:
                    covered_trimmed['right'] = covered[sides['right']]
            else:
                covered_trimmed = {side: covered[sides[side]] for side in ['left', 'right']}

            if interval.are_disjoint(covered_trimmed['left'], covered_trimmed['right']):
                gap = covered_trimmed['right'].start - covered_trimmed['left'].end - 1
                MH_nts = -gap
            else:   
                overlap = covered_trimmed['left'] & covered_trimmed['right']

                MH_nts = overlap.total_length

        return MH_nts

class NonoverlappingPairLayout():
    def __init__(self, R1_als, R2_als, target_info):
        self.target_info = target_info
        self.layouts = {
            'R1': Layout(R1_als, target_info, mode='illumina'),
            'R2': Layout(R2_als, target_info, mode='illumina'),
        }
        if self.layouts['R1'].name != self.layouts['R2'].name:
            raise ValueError
        
        self.name = self.layouts['R1'].name
        
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
            primer_als = self.layouts[which].primer_alignments
            sides = set(s for s, al in primer_als.items() if al is not None)
            if len(sides) == 1:
                side = sides.pop()
            else:
                side = None

            target_sides[which] = side

        return target_sides

    @memoized_property
    def strand(self):
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
    def inferred_length(self):
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
        if kind == 'h' and self.inferred_length > 0:
            self.length = self.inferred_length

            self.relevant_alignments = {
                'R1': self.layouts['R1'].parsimonious_target_alignments + self.layouts['R1'].parsimonious_donor_alignments,
                'R2': self.layouts['R2'].parsimonious_target_alignments + self.layouts['R2'].parsimonious_donor_alignments,
            }

            junctions = set(self.junctions.values())

            if 'blunt' in junctions and 'uncategorized' not in junctions:
                category = 'blunt misintegration'
                subcategory = f'5\' {self.junctions[5]}, 3\' {self.junctions[3]}'
                details = self.bridging_strand[kind]
            elif junctions == set(['imperfect', 'HDR']):
                category = 'incomplete HDR'
                subcategory = f'5\' {self.junctions[5]}, 3\' {self.junctions[3]}'
                details = self.bridging_strand[kind]
            elif junctions == set(['imperfect']):
                category = 'complex misintegration'
                subcategory = f'5\' {self.junctions[5]}, 3\' {self.junctions[3]}'
                details = self.bridging_strand[kind]

            else:
                self.length = -1
                category = 'malformed layout'
                subcategory = 'non-overlapping'
                details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif kind == 'nh' and self.inferred_length > 0:
            self.length = self.inferred_length

            category = 'non-homologous donor'
            subcategory = 'simple'
            details = 'n/a'
            self.relevant_alignments = {
                'R1': self.layouts['R1'].parsimonious_target_alignments + self.layouts['R1'].nonhomologous_donor_alignments,
                'R2': self.layouts['R2'].parsimonious_target_alignments + self.layouts['R2'].nonhomologous_donor_alignments,
            }
            
        elif kind == 'nonspecific_amplification' and self.inferred_length > 0:
            R1_primer = self.layouts['R1'].primer_alignments[5]
            R2_primer = self.layouts['R2'].primer_alignments[3]

            if R1_primer is not None and R2_primer is not None:
                self.length = self.inferred_length

                category = 'malformed layout'
                subcategory = 'nonspecific amplification'
                details = 'n/a'
                bridging_als = self.bridging_alignments['nonspecific_amplification']
                self.relevant_alignments = {
                    'R1': [R1_primer, bridging_als['R1']],
                    'R2': [R2_primer, bridging_als['R2']],
                }

            else:
                self.length = -1
                category = 'malformed layout'
                subcategory = 'non-overlapping'
                details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments
        
        elif kind == 'genomic_insertion' and self.inferred_length > 0:
            R1_primer = self.layouts['R1'].primer_alignments[5]
            R2_primer = self.layouts['R2'].primer_alignments[3]

            if R1_primer is not None and R2_primer is not None:
                self.length = self.inferred_length

                category = 'genomic insertion'
                subcategory = 'genomic insertion'
                details = 'n/a'
                bridging_als = self.bridging_alignments['genomic_insertion']
                self.relevant_alignments = {
                    'R1': [R1_primer, bridging_als['R1']],
                    'R2': [R2_primer, bridging_als['R2']],
                }
            else:
                self.length = -1
                category = 'malformed layout'
                subcategory = 'non-overlapping'
                details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments
            
        else:
            self.length = -1

            category = 'malformed layout'
            subcategory = 'non-overlapping'
            details = 'n/a'

            self.relevant_alignments = self.uncategorized_relevant_alignments
        
        if self.strand == '-':
            self.relevant_alignments = {
                'R1': self.relevant_alignments['R2'],
                'R2': self.relevant_alignments['R1'],
            }
            
        return category, subcategory, details
    
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

def get_mismatch_info(alignment, target_info):
    mismatches = []

    tuples = []
    if target_info.reference_sequences.get(alignment.reference_name) is None:
        for read_p, ref_p, ref_b in alignment.get_aligned_pairs(with_seq=True):
            if read_p != None and ref_p != None:
                read_b = alignment.query_sequence[read_p]
                tuples.append((read_p, read_b, ref_p, ref_b))

    else:
        reference = target_info.reference_sequences[alignment.reference_name]
        for read_p, ref_p in alignment.get_aligned_pairs():
            if read_p != None and ref_p != None:
                read_b = alignment.query_sequence[read_p]
                ref_b = reference[ref_p]
                
                tuples.append((read_p, read_b, ref_p, ref_b))

    for read_p, read_b, ref_p, ref_b in tuples:
        read_b = read_b.upper()
        ref_b = ref_b.upper()

        if read_b != ref_b:
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
            first_edge = sam.total_read_nucs(alignment.cigar[:i])
            second_edge = first_edge + length
            starts_at, ends_at = sorted(sam.true_query_position(p, alignment) for p in [first_edge, second_edge])
            indels.append(('insertion', (starts_at, ends_at)))

    return indels

category_order = [
    ('WT',
        ('WT',
        ),
    ),
    ('simple indel',
        ('insertion',
         'deletion',
         'deletion <50 nt',
         'deletion >=50 nt',
        ),
    ),
    ('complex indel',
         ('deletion plus insertion',
          'templated insertion',
          'multiple indels',
         ),
    ),
    ('HDR',
        ('HDR',
        ),
    ),
    ('blunt misintegration',
        ("5' HDR, 3' blunt",
         "5' blunt, 3' HDR",
         "5' blunt, 3' blunt",
         "5' blunt, 3' imperfect",
         "5' imperfect, 3' blunt",
        ),
    ),
    ('incomplete HDR',
        ("5' HDR, 3' imperfect",
         "5' imperfect, 3' HDR",
         "5' imperfect, 3' imperfect", # placeholder
        ),
    ),
    ('complex misintegration',
        ("5' imperfect, 3' imperfect",
         'other',
        ),
    ),
    ('concatenated misintegration',
        ('HDR',
         '5\' blunt',
         '3\' blunt',
         '5\' and 3\' blunt',
         'uncategorized',
        ),
    ),
    ('non-homologous donor',
        ('simple',
         'complex',
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
        ('nonspecific amplification',
         'non-overlapping',
         'extra copy of primer',
         'missing a primer',
         'too short',
         'primer far from read edge',
         'primers not in same orientation',
         'no alignments detected',
        ),
    ),
]

full_index = []
for cat, subcats in category_order:
    for subcat in subcats:
        full_index.append((cat, subcat))
        
full_index = pd.MultiIndex.from_tuples(full_index) 

categories = [c for c, scs in category_order]
subcategories = dict(category_order)

def order(outcome):
    if isinstance(outcome, tuple):
        category, subcategory = outcome

        try:
            return (categories.index(category),
                    subcategories[category].index(subcategory),
                )
        except:
            print(category, subcategory)
            raise
    else:
        category = outcome
        try:
            return categories.index(category)
        except:
            print(category)
            raise

def outcome_to_sanitized_string(outcome):
    if isinstance(outcome, tuple):
        c, s = order(outcome)
        return f'category{c:03d}_subcategory{s:03d}'
    else:
        c = order(outcome)
        return f'category{c:03d}'

def sanitized_string_to_outcome(sanitized_string):
    match = re.match('category(\d+)_subcategory(\d+)', sanitized_string)
    if match:
        c, s = map(int, match.groups())
        category, subcats = category_order[c]
        subcategory = subcats[s]
        return category, subcategory
    else:
        match = re.match('category(\d+)', sanitized_string)
        if not match:
            raise ValueError(sanitized_string)
        c = int(match.group(1))
        category, subcats = category_order[c]
        return category