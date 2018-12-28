from collections import Counter, defaultdict

import numpy as np
import pysam

from sequencing import interval, sam, utilities, sw, fastq
from sequencing.utilities import memoized_property

from knockin.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs

class Layout(object):
    def __init__(self, alignments, target_info):
        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.qual = np.array(sam.get_original_qual(alignment))
        
        self.primary_ref_names = set(self.target_info.reference_sequences)

        self.required_sw = False
        
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
    def simple_layout(self):
        ''' True if a single alignment to target covers essentially the whole read
        (or up to amplicon primers).'''
        t_als = self.parsimonious_target_alignments
        
        if len(t_als) != 1:
            is_simple = False
        else:
            t_al = t_als[0]
            start, end = sam.query_interval(t_al)

            missing_from = {
                'start': start,
                'end': len(self.seq) - end - 1,
            }

            simple_end = {
                'start': missing_from['start'] <= 20,
                'end': missing_from['end'] <= 20 or self.extends_to_primers(t_al),
            }

            is_simple = all(simple_end.values())
            
        return is_simple
    
    @memoized_property
    def Q30_fractions(self):
        at_least_30 = self.qual >= 30
        fracs = {
            'all': np.mean(at_least_30),
            'second_half': np.mean(at_least_30[len(at_least_30) // 2:]),
        }
        return fracs

    @memoized_property
    def base_calls(self):
        SNPs = self.target_info.SNPs
        position_to_name = {SNPs['target'][name]['position']: name for name in SNPs['target']}
        variable_locii = {name: [] for name in SNPs['target']}

        other = []

        for al in self.parsimonious_target_alignments:
            for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, self.target_info.target_sequence):
                if ref_i in position_to_name:
                    name = position_to_name[ref_i]

                    if SNPs['target'][name]['strand'] == '-':
                        read_b = utilities.reverse_complement(read_b)

                    variable_locii[name].append((read_b, qual))

                else:
                    if read_b != '-' and ref_b != '-' and read_b != ref_b:
                        snv = SNV(ref_i, read_b, qual)
                        other.append(snv)

        other = SNVs(other)
        return variable_locii, other

    @memoized_property
    def variable_locii_summary(self):
        SNPs = self.target_info.SNPs
        variable_locii, other = self.base_calls
        
        genotype = {}

        donor_seen = False

        for name in sorted(SNPs['target']):
            bs = defaultdict(list)

            for b, q in variable_locii[name]:
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
                        donor_seen = True

        details = ''.join(genotype[name] for name in sorted(SNPs['target']))

        return donor_seen, details

    @memoized_property
    def indels(self):
        ti = self.target_info

        around_cut_interval = ti.around_cuts(5)

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
        
        if len(ti.cut_afters) > 1:
            raise NotImplementedError('more than one cut')

        cut_after = ti.cut_afters[0]
        before_cut = utilities.reverse_complement(ti.target_sequence[:cut_after + 1])[:25]
        after_cut = ti.target_sequence[cut_after + 1:][:25]

        new_targets = [
            ('edge_before_cut', before_cut),
            ('edge_after_cut', after_cut),
        ]

        full_refs = list(self.original_header.references) + [name for name, seq in new_targets]
        full_lengths = list(self.original_header.lengths) + [len(seq) for name, seq in new_targets]

        expanded_header = pysam.AlignmentHeader.from_references(full_refs, full_lengths)

        edge_alignments = sw.align_read(self.read, new_targets, 3, expanded_header, both_directions=False, alignment_type='query_end')

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
                                      max_alignments_per_target=8,
                                      mismatch_penalty=-8,
                                      indel_penalty=-60,
                                      min_score_ratio=0,
                                     )

        return stringent_als

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
            
            details = {
                'gap_before': gap_before,
                'gap_after': gap_after,
                'edit_distance': total_edit_distance,
                'chr': genomic_al.reference_name,
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
        ti = self.target_info

        if len(ti.cut_afters) > 1:
            raise NotImplementedError('more than one cut')

        cut_after = ti.cut_afters[0]

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
                }

                possible_inserts.append(details)

        def is_valid(d):
            return (d['gap_before'] <= 5 and 
                    d['gap_after'] <= 5 and
                    d['edit_distance'] <= 5 and
                    d['length'] >= 5 and
                    (d['target_bounds'][5] is not None and d['target_bounds'][3] is not None)
                   )
        valid = [d for d in possible_inserts if is_valid(d)]

        return valid
    
    def categorize(self):
        standard_alignments = self.parsimonious_target_alignments + self.donor_alignments

        if all(al.is_unmapped for al in self.alignments):
            category = 'bad sequence'
            subcategory = 'no alignments detected'
            details = 'n/a'

            self.relevant_alignments = []

        elif self.simple_layout:
            _, other_SNVs = self.base_calls
            donor_seen, variable_locii_details = self.variable_locii_summary

            if donor_seen:
                category = 'donor'
                details = variable_locii_details

                one_base_dels = [indel for indel, near_cut in self.indels if indel.kind == 'D' and indel.length == 1]
                other_indels = [indel for indel in self.indels_near_cut if not (indel.kind == 'D' and indel.length == 1)]

                if len(other_indels) == 0:
                    if len(one_base_dels) == 0:
                        subcategory = 'clean'
                    else:
                        subcategory = 'synthesis errors'

                elif len(other_indels) == 1:
                    indel = other_indels[0]
                    if indel.kind == 'I':
                        subcategory = 'insertion'
                    elif indel.kind == 'D':
                        subcategory = 'deletion'

                else:
                    subcategory = 'multiple indels'

                self.relevant_alignments = standard_alignments

            else:
                if len(self.indels_near_cut) == 0:
                    category = 'wild type'

                    if len(other_SNVs) == 0:
                        subcategory = 'wild type'
                        details = 'n/a'
                    else:
                        subcategory = 'mismatches'
                        details = str(other_SNVs)
                    
                    self.relevant_alignments = standard_alignments

                elif len(self.indels_near_cut) == 1:
                    indel = self.indels_near_cut[0]

                    if indel.kind == 'D':
                        category = 'deletion'
                    
                        nearby = interval.Interval(min(indel.starts_ats) - 5, max(indel.starts_ats) + 5)
                        if any(SNV.position in nearby for SNV in other_SNVs):
                            subcategory = 'mismatch nearby'
                        else:
                            subcategory = 'clean'

                    elif indel.kind == 'I':
                        category = 'insertion'
                        subcategory = 'insertion'

                    details = self.indels_string

                    self.relevant_alignments = standard_alignments

                else:
                    if self.Q30_fractions['all'] < 0.5:
                        category = 'bad sequence'
                        subcategory = 'low quality'
                        details = '{:0.2f}'.format(self.Q30_fractions['all'])

                    elif self.Q30_fractions['second_half'] < 0.5:
                        category = 'bad sequence'
                        subcategory = 'low quality tail'
                        details = '{:0.2f}'.format(self.Q30_fractions['second_half'])

                    else:
                        category = 'uncategorized'
                        subcategory = 'uncategorized'
                        details = 'n/a'

                    self.relevant_alignments = self.alignments

        elif self.donor_insertion is not None:
            def details_to_string(details):
                fields = [
                    details['strand'],
                    details['target_bounds'][5],
                    details['target_bounds'][3],
                    details['gap_before'],
                    details['gap_after'],
                    details['edit_distance'],
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

            self.relevant_alignments = best_explanation['full_alignments']

        elif self.genomic_insertion is not None:
            def details_to_string(details):
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

            best_explanation = self.genomic_insertion[0]

            category = 'genomic insertion'

            # which supplemental index did it come from?
            subcategory = best_explanation['chr'].split('_')[0]

            details = details_to_string(best_explanation)

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

    @utilities.memoized_property
    def longest_polyG(self):
        locations = utilities.homopolymer_lengths(self.seq, 'G')

        if locations:
            max_length = max(length for p, length in locations)
        else:
            max_length = 0

        return max_length

    def alignments_to_plot(self):
        category, subcategory, details = self.categorize()

        if category == 'endogenous':
            #to_plot = [al for al in self.alignments if al.reference_name == subcategory]
            to_plot = self.alignments
            
        elif category in ('no indel', 'indel'):
            to_plot = self.parsimonious_target_alignments + \
                      [al for al in self.alignments if al.reference_name == self.target_info.donor]
            
        elif category == 'uncategorized':
            to_plot = self.primary_alignments + self.nonredundant_supplemental_alignments
            
        else:
            to_plot = self.alignments

        return to_plot

category_order = [
    ('wild type',
        ('wild type',
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
