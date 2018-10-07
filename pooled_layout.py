from collections import Counter, defaultdict

import numpy as np

from sequencing import interval, sam, utilities
from sequencing.utilities import memoized_property

from . import target_info

class Layout(object):
    def __init__(self, alignments, target_info):
        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.qual = np.array(sam.get_original_qual(alignment))
        
        self.primary_ref_names = set(self.target_info.reference_sequences)
        
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
            covered = interval.get_disjoint_covered(als)

            upstream, downstream = sorted(als, key=lambda al: al.reference_start)
            gap = downstream.reference_start - (upstream.reference_end - 1)

            if gap == 1 and len(covered.intervals) == 2:
                # If there are two disjoint alignments that are adjacent on
                # the reference, merge them into one insertion-containing alignment.
                first, second = covered.intervals

                insertion_length = second.start - first.end - 1

                cigar = upstream.cigar[:-1] + [(sam.BAM_CINS, insertion_length)] + downstream.cigar[1:]

                upstream.cigar = cigar
                als = [upstream]
            
            else:
                merged = sam.merge_adjacent_alignments(upstream, downstream, self.target_info.reference_sequences)
                if merged is not None:
                    als = [merged]
        
        return als
    
    @memoized_property
    def target_edge_alignments(self):
        ''' Get target alignments that make it to the read edges. '''
        edge_alignments = {5: [], 3:[]}

        for al in self.target_alignments:
            covered = interval.get_covered(al)
            
            if covered.start == 0:
                edge_alignments[5].append(al)
            
            if covered.end == len(self.seq) - 1:
                edge_alignments[3].append(al)

        for edge in [5, 3]:
            if len(edge_alignments[edge]) != 1:
                edge_alignments[edge] = None
            else:
                edge_alignments[edge] = edge_alignments[edge][0]

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
    def endogenous_up_to(self):
        up_to = {
            'HBG1': 0,
            'HBB': 0,
            'HBBP1': 0,
            'HBD': 0,
        }
        for name in up_to:
            relevant = [al for al in self.alignments if al.reference_name == name]
            if relevant:
                up_to[name] = max(al.reference_end for al in relevant)

        return up_to

    @memoized_property
    def specific_to_endogenous(self):
        specific = {}
        for name, boundary in [('HBG1', 520), ('HBB', 530), ('HBBP1', 530), ('HBD', 670)]:
            specific[name] = self.endogenous_up_to[name] > boundary
        return specific

    @memoized_property
    def base_calls_at_variable_locii(self):
        fingerprint = self.target_info.fingerprints[self.target_info.target]
        base_calls = {p: [] for (strand, p), b in fingerprint}
        strands = {p: strand for (strand, p), b in fingerprint}

        for al in self.parsimonious_target_alignments:
            for read_p, ref_p in al.get_aligned_pairs():
                if ref_p in base_calls:
                    if read_p is None:
                        b = '-'
                        q = -1
                    else:
                        b = al.query_sequence[read_p]
                        q = al.query_qualities[read_p]

                        if strands[ref_p] == '-':
                            b = utilities.reverse_complement(b)

                    base_calls[ref_p].append((b, q))

        return base_calls

    @memoized_property
    def variable_locii_summary(self):
        fingerprint = self.target_info.fingerprints[self.target_info.target]
        base_calls = self.base_calls_at_variable_locii
        
        genotype = ''
        ambiguous = False

        for (strand, p), b in fingerprint:
            bs = defaultdict(list)

            for b, q in base_calls[p]:
                bs[b].append(q)

            if len(bs) != 1:
                ambiguous = True
                break
            else:
                b, qs = list(bs.items())[0]

                if b != '-' and not any(q >= 30 for q in qs):
                    ambiguous = True
                    break

                genotype += b

        if ambiguous:
            genotype = 'ambiguous'

        return genotype

    @memoized_property
    def indels(self):
        primer = self.target_info.features[self.target_info.target, 'forward primer']

        indels = []
        for al in self.parsimonious_target_alignments:
            for i, (cigar_op, length) in enumerate(al.cigar):
                if cigar_op == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before
                    ends_at = starts_at + length - 1

                    in_primer = (starts_at >= primer.start) and (ends_at < primer.end)

                    kind = 'D'
                    details = (starts_at, length)

                elif cigar_op == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    in_primer = primer.start <= starts_after < primer.end

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    kind = 'I'
                    details = (starts_after, insertion)
                    
                else:
                    continue

                kind, details = self.target_info.expand_degenerate_indel((kind, details))

                indels.append((kind, details, in_primer))

        return indels

    @memoized_property
    def indels_string(self):
        indels = [(k, d) for k, d, in_primer in self.indels if not in_primer]
        reps = [target_info.degenerate_indel_to_string(*indel) for indel in indels]
        string = ' '.join(sorted(reps))
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
    def genomic_insertion(self):
        target_seq = self.target_info.target_sequence

        possible_insertions = []

        edge_als = self.target_edge_alignments 
        edge_covered = {}
        target_bounds = {}

        if edge_als[5] is None:
            return None
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
                
            target_bounds[5] = sam.closest_ref_position(left_switch_after, edge_als[5], which_side='before')
            genomic_bounds[5] =  sam.closest_ref_position(left_switch_after + 1, genomic_al, which_side='after')   
            
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
                
            cropped_left_al = sam.crop_al_to_query_int(edge_als[5], -np.inf, left_switch_after)
            target_bounds[5] = sam.reference_edges(cropped_left_al)[3]

            if edge_als[3] is None:
                cropped_right_al = None
                target_bounds[3] = None
            else:
                cropped_right_al = sam.crop_al_to_query_int(edge_als[3], right_switch_after + 1, np.inf)
                target_bounds[3] = sam.reference_edges(cropped_right_al)[5]

            cropped_genomic_al = sam.crop_al_to_query_int(genomic_al, left_switch_after + 1, right_switch_after)
            genomic_bounds = sam.reference_edges(cropped_genomic_al)   
                
            non_overlapping = genomic_covered - edge_covered[5] - edge_covered[3]
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
            }
            
            possible_insertions.append(details)

        is_valid = lambda d: d['gap_before'] <= 5 and d['gap_after'] <= 5 and d['edit_distance'] <= 5 
        valid = [d for d in possible_insertions if is_valid(d)]

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
    def donor_insertion(self):
        target_seq = self.target_info.target_sequence
        donor_seq = self.target_info.donor_sequence

        edge_als = self.target_edge_alignments

        edge_covered = {}
        if edge_als[5] is not None:
            edge_covered[5] = interval.get_covered(edge_als[5])
        else:
            return None
            #edge_covered[5] = interval.Interval(-np.inf, -1)

        if edge_als[3] is not None:
            edge_covered[3] = interval.get_covered(edge_als[3])
        else:
            return None
            #edge_covered[3] = interval.Interval(len(self.seq), np.inf)

        covered_from_edges = interval.get_disjoint_covered(edge_als.values())

        possible_inserts = []

        for donor_al in self.donor_alignments:
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

                cropped_left_al = sam.crop_al_to_query_int(edge_als[5], -np.inf, left_switch_after)

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

                cropped_right_al = sam.crop_al_to_query_int(edge_als[3], right_switch_after + 1, np.inf)

                cropped_donor_al = sam.crop_al_to_query_int(donor_al, left_switch_after + 1, right_switch_after)

                target_bounds[5] = sam.reference_edges(cropped_left_al)[3]
                target_bounds[3] = sam.reference_edges(cropped_right_al)[5]

                insert_bounds =  sam.reference_edges(cropped_donor_al)
                
                non_overlapping = insert_covered - edge_covered[5] - edge_covered[3]
                middle_edits = sam.edit_distance_in_query_interval(donor_al, non_overlapping, donor_seq)
                total_edit_distance = left_min_edits + middle_edits + right_min_edits

                details = {
                    'gap_before': gap_before,
                    'gap_after': gap_after,
                    'edit_distance': total_edit_distance,
                    'strand': sam.get_strand(donor_al),
                    'donor_bounds': insert_bounds,
                    'target_bounds': target_bounds,
                    'length': abs(insert_bounds[5] - insert_bounds[3]) + 1,
                    'cropped_alignments': [cropped_left_al, cropped_donor_al, cropped_right_al],
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

        if len(valid) == 0:
            valid = None

        return valid
    
    def offtarget_priming(self):
        ''' Check if a nonredundant supplemental alignment overlaps with the same
        part of the read that maps to the amplicon primer.
        '''
        n_s_als = self.nonredundant_supplemental_alignments
        if not n_s_als:
            return False

        n_s_covered = interval.get_disjoint_covered(n_s_als)
        primer = self.target_info.features[self.target_info.target, 'forward primer']
        for t_al in self.parsimonious_target_alignments:
            to_primer = sam.crop_al_to_ref_int(t_al, primer.start, primer.end)
            if interval.get_covered(to_primer) & n_s_covered:
                return True

        return False
    
    def categorize(self):
        if all(al.is_unmapped for al in self.alignments):
            category = 'bad sequence'
            subcategory = 'no alignments detected'
            details = 'n/a'

        elif self.simple_layout:
            # Ignore indels in primer
            indels = [(k, d) for k, d, in_primer in self.indels if not in_primer]

            if len(indels) == 0:
                category = 'no indel'

                genotype = self.variable_locii_summary

                if genotype == self.target_info.wild_type_locii:
                    subcategory = 'wild type'
                elif genotype == self.target_info.donor_locii:
                    subcategory = 'donor'
                else:
                    subcategory = 'other'

                if genotype == 'ambiguous':
                    details = 'ambiguous'
                else:
                    details = ''.join('_' if b == wt else b for b, wt in zip(genotype, self.target_info.wild_type_locii))

            elif len(indels) == 1:
                category = 'indel'

                kind, _, = indels[0]

                if kind == 'D':
                    subcategory = 'deletion'
                elif kind == 'I':
                    subcategory = 'insertion'

                details = self.indels_string

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
                    category = 'indel'
                    subcategory = 'multiple'
                    details = self.indels_string

        elif self.donor_insertion is not None:
            def priority(details):
                key_order = [
                    'gap_before',
                    'gap_after',
                    'edit_distance',
                ]
                return [details[k] for k in key_order]

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

            best_explanation = sorted(self.donor_insertion, key=priority)[0]

            category = 'donor insertion'
            subcategory = 'donor insertion'
            details = details_to_string(best_explanation)

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
            subcategory = 'genomic insertion'
            details = details_to_string(best_explanation)

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

            elif self.offtarget_priming():
                category = 'endogenous'
                subcategory = 'other offtarget'
                details = 'n/a'

            else:
                category = 'uncategorized'
                subcategory = 'uncategorized'
                details = 'n/a'

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

def string_to_indels(string):
    indels = []
    for field in string.split(' '):
        kind, rest = field.split(':')
        if kind == 'D':
            starts, length = rest.split(',')
            length = int(length)

            details = length

        elif kind == 'I':
            starts, seqs = rest.split(',')
            if '|' in seqs:
                seqs = seqs.strip('{}').split('|')
            else:
                seqs = [seqs]

            details = seqs

        if '|' in starts:
            starts = [int(s) for s in starts.strip('{}').split('|')]
        else:
            starts = [int(starts)]


        indels.append((kind, starts, details))

    return indels

category_order = [
    ('no indel',
        ('wild type',
         'donor',
         'other',
        ),
    ),
    ('indel',
        ('deletion',
         'large deletion',
         'insertion',
         'multiple',
        ),
    ),
    ('uncategorized',
        ('uncategorized',
        ),
    ),
    ('genomic insertion',
        ('genomic insertion',
        ),
    ),
    ('donor insertion',
        ('donor insertion',
        ),
    ),
    ('endogenous',
        ('HBG1',
         'HBB',
         'HBBP1',
         'HBD',
         'other offtarget',
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
