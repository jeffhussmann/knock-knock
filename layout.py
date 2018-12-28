import numpy as np
from collections import defaultdict

from sequencing import sam, interval, utilities
from knockin.target_info import DegenerateDeletion, DegenerateInsertion

memoized_property = utilities.memoized_property

class Layout(object):
    def __init__(self, alignments, target_info):
        self.target_info = target_info

        split_als = []
        for al in alignments:
            if al.reference_name == self.target_info.target:
                split_als.extend(sam.split_at_deletions(al, 2))
            elif al.reference_name == self.target_info.donor:
                split_als.extend(sam.split_at_deletions(al, 2))
            else:
                split_als.append(al)

        self.alignments = split_als
        
        alignment = alignments[0]
        self.name = alignment.query_name
        self.seq = sam.get_original_seq(alignment)
        self.qual = np.array(sam.get_original_qual(alignment))
        
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
        
        elif not self.is_mostly_covered:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'

        elif not self.has_integration:
            if self.scar_near_cut is not None:
                if len(self.scar_near_cut) > 1:
                    category = 'uncategorized'
                    subcategory = 'uncategorized'
                else:
                    category = 'indel'
                    indel = self.scar_near_cut[0]
                    if indel.kind == 'D':
                        subcategory = 'deletion'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                details = self.scar_string
            else:
                category = 'WT'
                subcategory = 'WT'

        elif self.integration_summary == 'donor':
            if self.junction_summary_per_side[5] == 'HDR' and self.junction_summary_per_side[3] == 'HDR':
                category = 'HDR'
                subcategory = 'HDR'
            else:
                category = 'misintegration'
                subcategory = '5\' {}, 3\' {}'.format(self.junction_summary_per_side[5], self.junction_summary_per_side[3])

        elif self.integration_summary == 'concatamer':
            category = 'concatamer'
            subcategory = self.junction_summary

        elif self.integration_summary in ['donor with indel', 'other', 'unexpected length', 'unexpected source']:
            category = 'uncategorized'
            subcategory = 'uncategorized'

        else:
            print(self.integration_summary)

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
        
        elif not self.is_mostly_covered:
            category = 'malformed layout'
            subcategory = 'primer far from read edge'

        else:
            if self.scar_near_cut is not None:
                category = 'indel'
                if len(self.scar_near_cut) > 1:
                    subcategory = 'complex indel'
                else:
                    indel = self.scar_near_cut[0]
                    if indel.kind == 'D':
                        if indel.length < 50:
                            subcategory = 'deletion <50 nt'
                        else:
                            subcategory = 'deletion >=50 nt'
                    elif indel.kind == 'I':
                        subcategory = 'insertion'

                details = self.scar_string
            else:
                category = 'WT'
                subcategory = 'WT'

        return category, subcategory, details

    @memoized_property
    def all_primer_alignments(self):
        ''' Get all alignments that contain the amplicon primers. '''
        als = {}
        for side in [5, 3]:
            primer = self.target_info.primers[side]
            als[side] = [al for al in self.alignments if sam.overlaps_feature(al, primer)]

        return als

    @memoized_property
    def extra_copy_of_primer(self):
        ''' Check if too many alignments containing either primer were found. '''
        return len(self.all_primer_alignments[5]) > 1 or len(self.all_primer_alignments[3]) > 1
    
    @memoized_property
    def missing_a_primer(self):
        ''' Check if either primer was not found in an alignments. '''
        return len(self.all_primer_alignments[5]) == 0 or len(self.all_primer_alignments[3]) == 0
        
    @memoized_property
    def primer_alignments(self):
        ''' Get the single alignment containing each primer. '''
        if self.extra_copy_of_primer or self.missing_a_primer:
            return None
        else:
            return {side: self.all_primer_alignments[side][0] for side in [5, 3]}
        
    @memoized_property
    def primer_strands(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        if self.primer_alignments is None:
            return None
        else:
            return {side: sam.get_strand(self.primer_alignments[side]) for side in [5, 3]}
    
    @memoized_property
    def strand(self):
        ''' Get which strand each primer-containing alignment mapped to. '''
        if self.primer_strands is None:
            return None
        elif self.primer_strands[5] != self.primer_strands[3]:
            return None
        else:
            return self.primer_strands[5]

    @memoized_property
    def covered_from_primers(self):
        ''' How much of the read is covered by alignments containing the primers? '''
        assert self.primer_strands[5] == self.primer_strands[3]
        return interval.get_disjoint_covered([self.primer_alignments[5], self.primer_alignments[3]])

    @memoized_property
    def is_mostly_covered(self):
        ''' TODO: this is misnamed - should be something like 'covers_beginning_and_end'
        '''
        return (self.covered_from_primers.start <= 10 and
                len(self.seq) - self.covered_from_primers.end <= 10
               )

    @memoized_property
    def has_integration(self):
        return self.merged_primer_alignment is None

    @memoized_property
    def merged_primer_alignment(self):
        primer_als = self.primer_alignments

        if self.is_mostly_covered:
            merged = sam.merge_adjacent_alignments(primer_als[5], primer_als[3], self.target_info.reference_sequences)
        else:
            merged = None

        return merged

    @memoized_property
    def scar_near_cut(self):
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
    def scar_string(self):
        if self.scar_near_cut is None:
            scar_string = None
        else:
            scar_string = ' '.join(map(str, self.scar_near_cut))

        return scar_string

    @memoized_property
    def donor_alignments(self):
        als = [
            al for al in self.alignments
            if al.reference_name == self.target_info.donor
        ]
        return als

    @memoized_property
    def parsimonius_alignments(self):
        return interval.make_parsimonious(self.alignments)

    @memoized_property
    def parsimonious_donor_alignments(self):
        als = [
            al for al in self.parsimonius_alignments
            if al.reference_name == self.target_info.donor
        ]
        return als

    @memoized_property
    def closest_donor_alignment_to_edge(self):
        ''' Identify the alignments to the donor closest to edge of the read
        that has the 5' and 3' amplicon primer. '''
        donor_als = self.donor_alignments

        if len(donor_als) > 0:
            closest = {}

            left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
            right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)

            if self.strand == '+':
                closest[5] = left_most
                closest[3] = right_most
            else:
                closest[5] = right_most
                closest[3] = left_most
        else:
            closest = {5: None, 3: None}

        return closest

    @memoized_property
    def clean_handoff(self):
        ''' Check if target sequence cleanly transitions to donor sequence at
        each junction between the two, with one full length copy of the relevant
        homology arm and no large indels (i.e. not from sequencing errors) near
        the internal edge.
        '''
        if len(self.donor_alignments) == 0:
            return {5: False, 3: False}

        from_primer = self.primer_alignments
        HAs = self.target_info.homology_arms
        closest_donor = self.closest_donor_alignment_to_edge

        target_contains_full_arm = {
            5: HAs['target', 5].end - from_primer[5].reference_end <= 10,
            3: from_primer[3].reference_start - HAs['target', 3].start <= 10,
        }

        donor_contains_arm_external = {
            5: closest_donor[5].reference_start - HAs['donor', 5].start <= 10,
            3: HAs['donor', 3].end - (closest_donor[3].reference_end - 1) <= 10,
        }

        donor_contains_arm_internal = {
            5: closest_donor[5].reference_end - 1 - HAs['donor', 5].end >= 20,
            3: HAs['donor', 3].start - closest_donor[3].reference_start >= 20,
        }

        donor_contains_full_arm = {
            side: donor_contains_arm_external[side] and donor_contains_arm_internal[side]
            for side in [5, 3]
        }
            
        target_external_edge_query = {
            5: sam.closest_query_position(HAs['target', 5].start, from_primer[5]),
            3: sam.closest_query_position(HAs['target', 3].end, from_primer[3]),
        }
        
        donor_external_edge_query = {
            5: sam.closest_query_position(HAs['donor', 5].start, closest_donor[5]),
            3: sam.closest_query_position(HAs['donor', 3].end, closest_donor[3]),
        }

        arm_overlaps = {
            side: abs(target_external_edge_query[side] - donor_external_edge_query[side]) <= 10
            for side in [5, 3]
        }

        junction = {
            5: HAs['donor', 5].end,
            3: HAs['donor', 3].start,
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
        edge_r = {
            5: [],
            3: [],
        }

        for al in self.parsimonious_donor_alignments:
            cropped = sam.crop_al_to_query_int(al, self.integration_interval.start, self.integration_interval.end)
            start = cropped.reference_start
            end = cropped.reference_end - 1
            if self.strand == '+':
                edge_r[5].append(start)
                edge_r[3].append(end)
            else:
                edge_r[3].append(start)
                edge_r[5].append(end)

        for side in [5, 3]:
            if len(edge_r[side]) != 1:
                # placeholder
                edge_r[side] = [None]

            edge_r[side] = edge_r[side][0]

        return edge_r

    @memoized_property
    def donor_relative_to_arm(self):
        ''' How much of the donor is integrated relative to the edges of the HAs? '''
        HAs = self.target_info.homology_arms

        # convention: positive if there is extra in the integration, negative if truncated
        relative_to_arm = {
            'internal': {
                5: (HAs['donor', 5].end + 1) - self.edge_r[5],
                3: self.edge_r[3] - (HAs['donor', 3].start - 1),
            },
            'external': {
                5: HAs['donor', 5].start - self.edge_r[5],
                3: self.edge_r[3] - HAs['donor', 3].end,
            },
        }

        return relative_to_arm
    
    @memoized_property
    def donor_relative_to_cut(self):
        ''' Distance on query between base aligned to donor before/after cut
        and start of target alignment.
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
            mask_end[5] = HAs['donor', 5].end
        else:
            mask_end[5] = cut_after

        if self.clean_handoff[3]:
            mask_start[3] = HAs['donor', 3].start
        else:
            mask_start[3] = cut_after + 1

        covered = {
            side: sam.crop_al_to_ref_int(flanking_al[side], mask_start[side], mask_end[side])
            for side in [5, 3]
        }

        disjoint_covered = interval.get_disjoint_covered([covered[5], covered[3]])

        return interval.Interval(disjoint_covered[0].end + 1, disjoint_covered[-1].start - 1)

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

        target_blunt = self.target_to_at_least_cut
        
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

        e_coli_alignments = [al for al in self.parsimonius_alignments if al.reference_name == 'e_coli_K12']

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

        if self.strand == '+':
            key = lambda al: interval.get_covered(al).start
            reverse = False
        else:
            key = lambda al: interval.get_covered(al).end
            reverse = True

        five_to_three = sorted(p_donor_als, key=key, reverse=reverse)
        junctions_clean = []

        for before, after in zip(five_to_three[:-1], five_to_three[1:]):
            adjacent = interval.are_adjacent(interval.get_covered(before), interval.get_covered(after))

            missing_before = HAs['donor', 3].end - (before.reference_end - 1)
            missing_after = after.reference_start - HAs['donor', 5].start

            clean = adjacent and (missing_before == 0) and (missing_after == 0)

            junctions_clean.append(clean)

        if all(junctions_clean):
            return len(junctions_clean) + 1
        else:
            return 0
    
    @memoized_property
    def indels(self):
        indels = []

        al = self.merged_primer_alignment

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
        (
         "5' HDR, 3' NHEJ",
         "5' NHEJ, 3' HDR",
         "5' HDR, 3' truncated",
         "5' truncated, 3' HDR",
         "5' NHEJ, 3' truncated",
         "5' truncated, 3' NHEJ",
         "5' NHEJ, 3' NHEJ",
         "5' truncated, 3' truncated",
        ),
    ),
    ('uncategorized',
        ('uncategorized',
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
    return (categories.index(category),
            subcategories[category].index(subcategory),
           )