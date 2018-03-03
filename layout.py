import pysam
import numpy as np

import Sequencing.utilities as utilities
import Sequencing.sam as sam
import Sequencing.interval as interval

from collections import defaultdict

def overlaps_feature(alignment, feature):
    same_reference = alignment.reference_name == feature.seqname
    num_overlapping_bases = alignment.get_overlap(feature.start, feature.end)
    return same_reference and (num_overlapping_bases > 0)

def identify_alignments_from_primers(als, primers, cut_after):
    query_length = als[0].query_length

    all_als_from_primers = {
        side: [al for al in als if overlaps_feature(al, primers[side])]
        for side in [5, 3]
    }

    als_from_primers = None
    strand = None

    if len(all_als_from_primers[5]) > 1 or len(all_als_from_primers[3]) > 1:
        outcome = ('malformed layout', 'extra copies of primer')

    elif len(all_als_from_primers[5]) == 0 or len(all_als_from_primers[3]) == 0:
        outcome = ('malformed layout', 'missing a primer')

    else:
        als_from_primers = {side: all_als_from_primers[side][0] for side in [5, 3]}

        strands = {side: sam.get_strand(als_from_primers[side]) for side in [5, 3]}
        if strands[5] != strands[3]:
            outcome = ('malformed layout', 'primers not in same orientation')

        else:
            strand = strands[5]
            # How much of the read is covered by alignments containing the primers?
            covered = interval.get_disjoint_covered([als_from_primers[5], als_from_primers[3]])

            if covered.start > 10 or query_length - covered.end > 10:
                outcome = ('malformed layout', 'primer far from read edge')

            else:
                no_integration = False

                if als_from_primers[5] == als_from_primers[3]:
                    merged = als_from_primers[5]
                    no_integration = True

                elif len(covered) == 1:
                    # The number of disjoint intervals is 1 - i.e. it is a
                    # single connected interval.
                    if als_from_primers[5].reference_end < als_from_primers[3].reference_start:
                        merged = sam.merge_adjacent_alignments(als_from_primers[5], als_from_primers[3])
                        no_integration = True

                if no_integration:
                    largest_deletion = largest_deletion_nearby(merged, cut_after, 10)
                    if largest_deletion == 0:
                        outcome = ('no integration', 'no scar')
                    else:
                        outcome = ('no integration', 'deletion near cut')

                else:
                    outcome = 'integration'

    return outcome, als_from_primers, strand

def check_for_clean_handoffs(als_from_primers, strand, donor_als, HAs):
    # Identify the alignments to the donor closest to edge of the read
    # that has the 5' and 3' PCR primer.
    closest_donor_al_to_edges = {}
    left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
    right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)
    if strand == '+':
        closest_donor_al_to_edges[5] = left_most
        closest_donor_al_to_edges[3] = right_most
    else:
        closest_donor_al_to_edges[5] = right_most
        closest_donor_al_to_edges[3] = left_most

    target_contains_full_arm = {
        5: HAs['target', 5].end - als_from_primers[5].reference_end <= 10,
        3: als_from_primers[3].reference_start - HAs['target', 3].start <= 10,
    }

    donor_contains_arm_external = {
        5: closest_donor_al_to_edges[5].reference_start - HAs['donor', 5].start <= 10,
        3: HAs['donor', 3].end - (closest_donor_al_to_edges[3].reference_end - 1) <= 10,
    }

    donor_contains_arm_internal = {
        5: closest_donor_al_to_edges[5].reference_end - 1 - HAs['donor', 5].end >= 20,
        3: HAs['donor', 3].start - closest_donor_al_to_edges[3].reference_start >= 20,
    }

    donor_contains_full_arm = {
        side: donor_contains_arm_external[side] and donor_contains_arm_internal[side]
        for side in [5, 3]
    }
        
    target_external_edge_query = {
        5: sam.closest_query_position(HAs['target', 5].start, als_from_primers[5]),
        3: sam.closest_query_position(HAs['target', 3].end, als_from_primers[3]),
    }
    
    donor_external_edge_query = {
        5: sam.closest_query_position(HAs['donor', 5].start, closest_donor_al_to_edges[5]),
        3: sam.closest_query_position(HAs['donor', 3].end, closest_donor_al_to_edges[3]),
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
        side: largest_indel_nearby(closest_donor_al_to_edges[side], junction[side], 10)
        for side in [5, 3]
    }

    clean_handoffs = {}
    for side in [5, 3]:
        clean_handoffs[side] = (
            target_contains_full_arm[side] and
            donor_contains_full_arm[side] and
            arm_overlaps[side] and
            max_indel_near_junction[side] <= 2
        )

    return clean_handoffs, closest_donor_al_to_edges

def check_for_blunt_compatibility(als_from_primers):
    target_to_at_least_cut = {
        5: als_from_primers[5].reference_end - 1 >= cut_after,
        3: als_from_primers[3].reference_start <= (cut_after + 1),
    }

def get_integration_interval(als_from_primers, closest_donor_al_to_edges, clean_handoffs, HAs, cut_after):
    flanking_al = {}
    mask_start = {5: -np.inf}
    mask_end = {3: np.inf}
    for side in [5, 3]:
        if clean_handoffs[side]:
            flanking_al[side] = closest_donor_al_to_edges[side]
        else:
            flanking_al[side] = als_from_primers[side]

    if clean_handoffs[5]:
        mask_end[5] = HAs['donor', 5].end
    else:
        mask_end[5] = cut_after

    if clean_handoffs[3]:
        mask_start[3] = HAs['donor', 3].start
    else:
        mask_start[3] = cut_after + 1

    covered = {
        side: sam.crop_al_to_ref_int(flanking_al[side], mask_start[side], mask_end[side])
        for side in [5, 3]
    }

    disjoint_covered = interval.get_disjoint_covered([covered[5], covered[3]])
    integration_interval = interval.Interval(disjoint_covered[0].end + 1, disjoint_covered[-1].start - 1)

    return integration_interval

def largest_deletion_nearby(alignment, ref_pos, window):
    ref_pos_to_block = sam.get_ref_pos_to_block(alignment)
    nearby = range(ref_pos - window, ref_pos + window)
    blocks = [ref_pos_to_block.get(r, (0, 0)) for r in nearby]
    deletions = [l for k, l in blocks if k == sam.BAM_CDEL]
    if deletions:
        largest_deletion = max(deletions)
    else:
        largest_deletion = 0

    return largest_deletion

def largest_insertion_nearby(alignment, ref_pos, window):
    nearby = sam.crop_al_to_ref_int(alignment, ref_pos - window, ref_pos + window)
    largest_insertion = sam.max_block_length(nearby, {sam.BAM_CINS})
    return largest_insertion

def largest_indel_nearby(alignment, ref_pos, window):
    largest_deletion = largest_deletion_nearby(alignment, ref_pos, window)
    largest_insertion = largest_insertion_nearby(alignment, ref_pos, window)
    return max(largest_deletion, largest_insertion)
    
def characterize_layout(als, target):
    if all(al.is_unmapped for al in als):
        return ('malformed layout', 'no alignments detected')

    quals = als[0].query_qualities

    outcome, als_from_primers, strand = identify_alignments_from_primers(als, target.primers, target.cut_after)

    if outcome != 'integration':
        print('\t', outcome)
        return outcome
    
    donor_als = [al for al in als if al.reference_name == target.donor]
    clean_handoffs, closest_donor_al_to_edges = check_for_clean_handoffs(als_from_primers, strand, donor_als, target.homology_arms)

    integration_interval = get_integration_interval(als_from_primers, closest_donor_al_to_edges, clean_handoffs, target.homology_arms, target.cut_after)

    print(clean_handoffs)
    print(integration_interval)
    return ('test', 'test')

    als_from_primers[5] = sam.crop_al_to_ref_int(als_from_primers[5], -np.inf, cut_after)
    als_from_primers[3] = sam.crop_al_to_ref_int(als_from_primers[3], cut_after + 1, np.inf)

    
    target_edge_relative_to_cut = {
        5: als_from_primers[5].reference_end - 1 - cut_after,
        3: als_from_primers[3].reference_start - (cut_after + 1),
    }
    
    target_q = {
        5: sam.true_query_position(als_from_primers[5].query_alignment_end - 1, als_from_primers[5]),
        3: sam.true_query_position(als_from_primers[3].query_alignment_start, als_from_primers[3]),
    }
    
    target_blunt = {}
    for side in [5, 3]:
        blunt = False
        offset = target_edge_relative_to_cut[side]
        if offset == 0:
            blunt = True
        elif abs(offset) <= 3:
            q = target_q[side]
            if min(quals[q - 3: q + 4]) <= 30:
                blunt = True
                
        target_blunt[side] = blunt
    
    # Mask off the parts of the read explained by the primer-containing alignments.
    # If these alignments go to exactly the cut site, use the cut as the boundary.
    # Otherwise, use the homology arm edge.
    
    if target_blunt[5]:
        mask_end = cut_after
    else:
        mask_end = HAs[target, 5].end
        
    has_primer[5] = sam.crop_al_to_ref_int(has_primer[5], 0, mask_end)
    
    if target_blunt[3]:
        mask_start = cut_after + 1
    else:
        mask_start = HAs[target, 3].start
        
    has_primer[3] = sam.crop_al_to_ref_int(has_primer[3], mask_start, np.inf)
    
    covered_from_primers = interval.get_disjoint_covered([has_primer[5], has_primer[3]])
    integration_interval = interval.Interval(covered_from_primers[0].end + 1, covered_from_primers[-1].start - 1)

    parsimonious = interval.make_parsimoninous(als)
    donor_als = [al for al in parsimonious if al.reference_name == donor]

    if len(donor_als) == 0:
        e_coli_als = [al for al in parsimonious if al.reference_name == 'e_coli_K12']
        if len(e_coli_als) == 1:
            e_coli_al, = e_coli_als
            e_coli_covered = interval.get_covered(e_coli_al)
            if e_coli_covered.start - integration_interval.start <= 10 and integration_interval.end - e_coli_covered.end <= 10:
                return ('non-GFP integration', 'e coli')
            else:
                return ('non-GFP integration', 'e coli')
        else:
            return ('non-GFP integration', 'uncategorized')

    five_most = sam.restrict_alignment_to_query_interval(five_most, integration_interval.start, integration_interval.end)
    three_most = sam.restrict_alignment_to_query_interval(three_most, integration_interval.start, integration_interval.end)

    # 5 and 3 are both positive if there is extra in the insert and negative if truncated
    donor_relative_to_arm_internal = {
        5: (HAs[donor, 5].end + 1) - five_most.reference_start,
        3: ((three_most.reference_end - 1) + 1) - HAs[donor, 3].start,
    }
    
    donor_relative_to_arm_external = {
        5: five_most.reference_start - HAs[donor, 5].start,
        3: three_most.reference_end - 1 - HAs[donor, 3].end,
    }
    
    donor_q = {
        5: sam.true_query_position(five_most.query_alignment_start, five_most),
        3: sam.true_query_position(three_most.query_alignment_end - 1, three_most),
    }
    
    donor_blunt = {}
    for side in [5, 3]:
        blunt = False
        offset = donor_relative_to_arm_external[side]
        
        if offset == 0:
            blunt = True
        elif abs(offset) <= 3:
            q = donor_q[side]
            if min(quals[q - 3:q + 4]) <= 30:
                blunt = True
                
        donor_blunt[side] = blunt
        
    junction_status = []
    
    for side in [5, 3]:
        if target_blunt[side] and donor_blunt[side]:
            junction_status.append('{0}\' blunt'.format(side))
        elif clean_handoffs[side]:
            pass
        else:
            junction_status.append('{0}\' uncategorized'.format(side))
    
    if len(junction_status) == 0:
        junction_description = ' '
    else:
        junction_description = ', '.join(junction_status)
    
    if len(donor_als) == 1:
        if clean_handoffs[5] and clean_handoffs[3]:
            insert_description = 'expected'
        elif donor_relative_to_arm_internal[5] < 0 or donor_relative_to_arm_internal[3] < 0:
            insert_description = 'truncated'
        elif donor_relative_to_arm_internal[5] > 0 or donor_relative_to_arm_internal[3] > 0:
            insert_description = 'extended'
        else:
            insert_description = 'uncategorized insertion'
        
        if sam.get_strand(donor_als[0]) != strand:
            insert_description = 'flipped ' + insert_description
            
    else:
        # Concatamer?
        if strand == '+':
            key = lambda al: interval.get_covered(al).start
            reverse = False
        else:
            key = lambda al: interval.get_covered(al).end
            reverse = True

        five_to_three = sorted(donor_als, key=key, reverse=reverse)
        concatamer_junctions = []
        for before_junction, after_junction in zip(five_to_three[:-1], five_to_three[1:]):
            adjacent = interval.are_adjacent(interval.get_covered(before_junction), interval.get_covered(after_junction))
            missing_before = before_junction.reference_end - 1 - HAs[donor, 3].end
            missing_after = after_junction.reference_start - HAs[donor, 5].start
            clean = adjacent and (missing_before == 0) and (missing_after == 0)
            concatamer_junctions.append(clean)

        if all(concatamer_junctions):
            #insert_description += ', {0}-mer GFP'.format(len(concatamer_junctions) + 1)
            insert_description = 'concatamer'.format(len(concatamer_junctions) + 1)
        else:
            insert_description = 'uncategorized insertion'
    
        if any(sam.get_strand(donor_al) != strand for donor_al in donor_als):
            insert_description = 'flipped ' + insert_description

    outcome = (insert_description, junction_description)
    return outcome

def count_outcomes(bam_by_name_fn, target):
    bam_fh = pysam.AlignmentFile(bam_by_name_fn)
    alignment_groups = utilities.group_by(bam_fh, lambda al: al.query_name)

    outcomes = defaultdict(list)

    group_i = 0
    for name, als in alignment_groups:
        group_i += 1
        if group_i > 10:
            break
        
        print(name)
        outcome = characterize_layout(als, target)
        outcomes[outcome].append(name)

        
    bam_fh.close()
        
    return outcomes
