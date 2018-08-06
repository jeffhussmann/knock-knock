from collections import Counter, defaultdict

import numpy as np

from sequencing import interval, sam, utilities

class Layout(object):
    def __init__(self, alignments, target_info):
        self.alignments = [al for al in alignments if not al.is_unmapped]
        self.target_info = target_info
        
        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.qual = np.array(sam.get_original_qual(alignment))
        
        self.primary_ref_names = set(self.target_info.reference_sequences)
        
    @utilities.memoized_property
    def target_alignments(self):
        t_als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.target
        ]
        
        return t_als
    
    @utilities.memoized_property
    def primary_alignments(self):
        p_als = [
            al for al in self.alignments
            if al.reference_name in self.primary_ref_names
        ]
        
        return p_als
    
    @utilities.memoized_property
    def supplemental_alignments(self):
        s_als = [
            al for al in self.alignments
            if al.reference_name not in self.primary_ref_names
        ]
        
        return s_als
    
    @utilities.memoized_property
    def phiX_alignments(self):
        als = [
            al for al in self.alignments
            if al.reference_name == 'phiX'
        ]
        
        return als
    
    @utilities.memoized_property
    def parsimonious_target_alignments(self):
        als = interval.make_parsimonious(self.target_alignments)
                             
        # If there are two target alignments that cover the beginning and end
        # of the read but have a large insertion in between, merge them into
        # one insertion-containing alignment.
        if len(als) == 2:
            covered = interval.get_disjoint_covered(als)
            if covered.start < 10 and len(self.seq) - covered.end < 10:
                upstream, downstream = sorted(als, key=lambda al: al.reference_start)
                gap = downstream.reference_start - (upstream.reference_end - 1)

                if gap == 1:
                    first, second = covered.intervals
                    insertion_length = second.start - first.end - 1

                    cigar = upstream.cigar[:-1] + [(sam.BAM_CINS, insertion_length)] + downstream.cigar[1:]

                    upstream.cigar = cigar
                    als = [upstream]
        
        return als

    def extends_past_PAS(self, al):
        if (self.target_info.target, 'PAS') in self.target_info.features:
            PAS = self.target_info.features[self.target_info.target, 'PAS']
            return al.reference_start <= PAS.start <= al.reference_end
        else:
            return False

    @utilities.memoized_property
    def simple_layout(self):
        ''' True if a single alignment to target covers essentially the whole read.'''
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
                'end': missing_from['end'] <= 20 or self.extends_past_PAS(t_al),
            }

            is_simple = all(simple_end.values())
            
        return is_simple
    
    @utilities.memoized_property
    def Q30_fractions(self):
        at_least_30 = self.qual >= 30
        fracs = {
            'all': np.mean(at_least_30),
            'second_half': np.mean(at_least_30[len(at_least_30) // 2:]),
        }
        return fracs

    @utilities.memoized_property
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

    @utilities.memoized_property
    def specific_to_endogenous(self):
        specific = {}
        for name, boundary in [('HBG1', 520), ('HBB', 530), ('HBBP1', 530), ('HBD', 670)]:
            specific[name] = self.endogenous_up_to[name] > boundary
        return specific

    @utilities.memoized_property
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

    @utilities.memoized_property
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

                if not any(q >= 30 for q in qs):
                    ambiguous = True
                    break

                genotype += b

        if ambiguous:
            genotype = 'ambiguous'

        return genotype

    @utilities.memoized_property
    def indels(self):
        primer = self.target_info.features[self.target_info.target, 'forward primer']

        indels = []
        for al in self.parsimonious_target_alignments:
            for i, (kind, length) in enumerate(al.cigar):
                if kind == sam.BAM_CDEL:
                    nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_at = al.reference_start + nucs_before
                    ends_at = starts_at + length - 1

                    in_primer = (starts_at >= primer.start) and (ends_at < primer.end)

                    info = ('D', (starts_at, length), in_primer)

                elif kind == sam.BAM_CINS:
                    ref_nucs_before = sam.total_reference_nucs(al.cigar[:i])
                    starts_after = al.reference_start + ref_nucs_before - 1

                    in_primer = primer.start <= starts_after < primer.end

                    read_nucs_before = sam.total_read_nucs(al.cigar[:i])
                    insertion = al.query_sequence[read_nucs_before:read_nucs_before + length]

                    ref_seq = self.target_info.reference_sequences[self.target_info.target]

                    info = ('I', (starts_after, insertion), in_primer)
                    
                else:
                    continue

                indels.append(info)

        return indels

    @utilities.memoized_property
    def large_deletion_info(self):
        large_deletion = None

        t_als = self.parsimonious_target_alignments
        if len(t_als) == 2:
            first, second = t_als

            if sam.get_strand(first) == sam.get_strand(second):
                strand = sam.get_strand(first)

                left, right = sorted([first, second], key=sam.query_interval)
                (q_left_start, q_left_end), (q_right_start, q_right_end) = map(sam.query_interval, (left, right))

                # logic copied from sam.merge_adjacent_alignements
                query_overlap = q_left_end - (q_right_start - 1)
                if query_overlap > 0:
                    if strand == '+':
                        five_edge = left.reference_end - 1
                        three_edge = right.reference_start
                    else:
                        five_edge = right.reference_end - 1
                        three_edge = left.reference_start

                    if three_edge > five_edge:
                        large_deletion = (five_edge, three_edge, query_overlap)

        return large_deletion

    def indels_to_string(self, indels):
        reps = []
        for kind, details, in_primer in indels:
            rep = self.target_info.degenerate_indels.get((kind, details))
            if rep is None:
                rep = '{0}:{1},{2}'.format(kind, *details)
            reps.append(rep)

        string = ' '.join(sorted(reps))
        return string
    
    @utilities.memoized_property
    def nonredundant_supplemental_alignments(self):
        covered_from_target = interval.get_disjoint_covered(self.parsimonious_target_alignments)
        
        nonredundant = []
        
        for al in self.supplemental_alignments:
            covered  = interval.get_covered(al)
            novel_length = (covered - covered_from_target).total_length
            #print(covered, covered_from_target, covered - covered_from_target)
            #print(novel_length)
            if novel_length > 15:
                nonredundant.append(al)

        return nonredundant

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
        if any(self.specific_to_endogenous.values()):
            for name in ['HBG1', 'HBB', 'HBBP1', 'HBD']:
                if self.specific_to_endogenous[name]:
                    category = 'endogenous'
                    subcategory = name
                    details = 'n/a'
                    break

        elif self.simple_layout:
            # Ignore indels in primer
            indels = [(k, d, in_primer) for k, d, in_primer in self.indels if not in_primer]

            if len(indels) == 0:
                category = 'no indel'

                genotype = self.variable_locii_summary
                fingerprints = self.target_info.fingerprints
                wild_type = ''.join([b for _, b in fingerprints[self.target_info.target]])
                donor = ''.join([b for _, b in fingerprints[self.target_info.donor]])

                if genotype == wild_type:
                    subcategory = 'wild type'
                elif genotype == donor:
                    subcategory = 'donor'
                else:
                    subcategory = 'other'

                details = genotype

            elif len(indels) == 1:
                category = 'indel'

                kind, _, _ = indels[0]

                if kind == 'D':
                    subcategory = 'deletion'
                elif kind == 'I':
                    subcategory = 'insertion'

                details = self.indels_to_string(indels)

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
                    details = self.indels_to_string(indels)

        elif self.large_deletion_info is not None:
            category = 'indel'
            subcategory = 'large deletion'
            details = 'D:{0}-{1} ({2})'.format(*self.large_deletion_info)

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
