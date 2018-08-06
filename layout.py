import numpy as np

from sequencing import sam, interval

def characterize_layout(als, target_info):
    if all(al.is_unmapped for al in als):
        layout_info = {
            'outcome': {
                'description': ('malformed layout', 'no alignments detected'),
                'sort_order': (600, 500),
            },
            'malformed': True,
        }
        return layout_info

    quals = als[0].query_qualities
    if als[0].is_reverse:
        quals = quals[::-1]

    layout_info = {
        'alignments': {
            'all': als,
            'parsimonious': interval.make_parsimoninous(als),
        },
        'quals': quals,
    }

    identify_flanking_target_alignments(layout_info, target_info)
    
    if layout_info['has_integration']:

        check_for_clean_handoffs(layout_info, target_info)

        identify_integration_interval(layout_info, target_info)

        check_flanking_for_blunt_compatibility(layout_info, target_info)

        characterize_integration_edges(layout_info, target_info)

        summarize_junctions(layout_info, target_info)

        characterize_integration(layout_info, target_info)

    summarize_outcome(layout_info)

    return layout_info

def overlaps_feature(alignment, feature):
    same_reference = alignment.reference_name == feature.seqname
    num_overlapping_bases = alignment.get_overlap(feature.start, feature.end)
    return same_reference and (num_overlapping_bases > 0)

def identify_flanking_target_alignments(layout_info, target_info):
    als = layout_info['alignments']['all']
    primers = target_info.primers
    cut_after = target_info.cut_after
    query_length = als[0].query_length

    all_als_from_primers = {
        side: [al for al in als if overlaps_feature(al, primers[side])]
        for side in [5, 3]
    }

    als_from_primers = None
    strand = None
    has_integration = False

    if len(all_als_from_primers[5]) > 1 or len(all_als_from_primers[3]) > 1:
        layout_info['malformed'] = '100: extra copies of primer'

    elif len(all_als_from_primers[5]) == 0 or len(all_als_from_primers[3]) == 0:
        layout_info['malformed'] = '200: missing a primer'

    else:
        als_from_primers = {side: all_als_from_primers[side][0] for side in [5, 3]}

        strands = {side: sam.get_strand(als_from_primers[side]) for side in [5, 3]}
        if strands[5] != strands[3]:
            layout_info['malformed'] = '300: primers not in same orientation'

        else:
            strand = strands[5]
            # How much of the read is covered by alignments containing the primers?
            covered = interval.get_disjoint_covered([als_from_primers[5], als_from_primers[3]])

            if covered.start > 10 or query_length - covered.end > 10:
                layout_info['malformed'] = '400: primer far from read edge'

            else:
                has_integration = True

                if als_from_primers[5] == als_from_primers[3]:
                    merged = als_from_primers[5]
                    has_integration = False

                elif len(covered) == 1:
                    # The number of disjoint intervals is 1 - i.e. it is a
                    # single connected interval.
                    if als_from_primers[5].reference_end < als_from_primers[3].reference_start:
                        merged = sam.merge_adjacent_alignments(als_from_primers[5], als_from_primers[3])
                        has_integration = False

                if not has_integration:
                    layout_info['scar'] = max_indel_nearby(merged, cut_after, 10)

    layout_info['has_integration'] = has_integration
    layout_info['strand'] = strand
    layout_info['alignments']['from_primer'] = als_from_primers
    return layout_info

def check_for_clean_handoffs(layout_info, target_info):
    # Identify the alignments to the donor closest to edge of the read
    # that has the 5' and 3' PCR primer.
    closest_donor_to_edge = {}

    donor_als = [
        al for al in layout_info['alignments']['all']
        if al.reference_name == target_info.donor
    ]

    if len(donor_als) == 0:
        layout_info['clean_handoffs'] = {5: False, 3: False}
        layout_info['alignments']['closest_donor_to_edge'] = {5: None, 3: None}
        return layout_info


    left_most = min(donor_als, key=lambda al: interval.get_covered(al).start)
    right_most = max(donor_als, key=lambda al: interval.get_covered(al).end)

    if layout_info['strand'] == '+':
        closest_donor_to_edge[5] = left_most
        closest_donor_to_edge[3] = right_most
    else:
        closest_donor_to_edge[5] = right_most
        closest_donor_to_edge[3] = left_most

    from_primer = layout_info['alignments']['from_primer']
    HAs = target_info.homology_arms

    target_contains_full_arm = {
        5: HAs['target', 5].end - from_primer[5].reference_end <= 10,
        3: from_primer[3].reference_start - HAs['target', 3].start <= 10,
    }

    donor_contains_arm_external = {
        5: closest_donor_to_edge[5].reference_start - HAs['donor', 5].start <= 10,
        3: HAs['donor', 3].end - (closest_donor_to_edge[3].reference_end - 1) <= 10,
    }

    donor_contains_arm_internal = {
        5: closest_donor_to_edge[5].reference_end - 1 - HAs['donor', 5].end >= 20,
        3: HAs['donor', 3].start - closest_donor_to_edge[3].reference_start >= 20,
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
        5: sam.closest_query_position(HAs['donor', 5].start, closest_donor_to_edge[5]),
        3: sam.closest_query_position(HAs['donor', 3].end, closest_donor_to_edge[3]),
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
        side: max_indel_nearby(closest_donor_to_edge[side], junction[side], 10)
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

    layout_info['clean_handoffs'] = clean_handoffs
    layout_info['alignments']['closest_donor_to_edge'] = closest_donor_to_edge
    return layout_info

def check_flanking_for_blunt_compatibility(layout_info, target_info):
    from_primer = layout_info['alignments']['from_primer']
    cut_after = target_info.cut_after

    target_to_at_least_cut = {
        5: from_primer[5].reference_end - 1 >= cut_after,
        3: from_primer[3].reference_start <= (cut_after + 1),
    }

    layout_info['target_to_at_least_cut'] = target_to_at_least_cut
    return layout_info

def characterize_integration_edges(layout_info, target_info):
    # 'int_int' short for 'integration_interval'
    int_int = layout_info['integration_interval']
    HAs = target_info.homology_arms
    quals = layout_info['quals']

    donor_als = [
        al for al in layout_info['alignments']['parsimonious']
        if al.reference_name == target_info.donor
    ]

    if layout_info['strand'] == '+':
        edge_q = {
            5: int_int.start,
            3: int_int.end,
        }
    else:
        edge_q = {
            5: int_int.end,
            3: int_int.start,
        }

    edge_r = {
        5: [],
        3: [],
    }

    for al in donor_als:
        q_to_r = {
            sam.true_query_position(q, al): r
            for q, r in al.aligned_pairs
            if r is not None and q is not None
        }

        for side in [5, 3]:
            if edge_q[side] in q_to_r:
                edge_r[side].append(q_to_r[edge_q[side]])

    for side in [5, 3]:
        if len(edge_r[side]) != 1:
            # placeholder
            edge_r[side] = [-1000]

        edge_r[side] = edge_r[side][0]

    # convention: positive if there is extra in the integration, negative if truncated
    relative_to_arm_internal = {
        5: (HAs['donor', 5].end + 1) - edge_r[5],
        3: edge_r[3] - (HAs['donor', 3].start - 1),
    }
    
    relative_to_arm_external = {
        5: HAs['donor', 5].start - edge_r[5],
        3: edge_r[3] - HAs['donor', 3].end,
    }
    
    donor_blunt = {}
    for side in [5, 3]:
        blunt = False
        offset = relative_to_arm_external[side]
        
        if offset == 0:
            blunt = True
        elif abs(offset) <= 3:
            q = edge_q[side]
            if min(quals[q - 3:q + 4]) <= 30:
                blunt = True
                
        donor_blunt[side] = blunt

    layout_info['integration_blunt'] = donor_blunt
    layout_info['donor_relative_to_arm_internal'] = relative_to_arm_internal

def identify_integration_interval(layout_info, target_info):
    alignments = layout_info['alignments']
    clean_handoffs = layout_info['clean_handoffs']
    HAs = target_info.homology_arms
    cut_after = target_info.cut_after

    flanking_al = {}
    mask_start = {5: -np.inf}
    mask_end = {3: np.inf}
    for side in [5, 3]:
        if clean_handoffs[side]:
            flanking_al[side] = alignments['closest_donor_to_edge'][side]
        else:
            flanking_al[side] = alignments['from_primer'][side]

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

    layout_info['integration_interval'] = integration_interval
    return layout_info

def summarize_junctions(layout_info, target_info):
    junction_status = {}

    target_blunt = layout_info['target_to_at_least_cut']
    donor_blunt = layout_info['integration_blunt']
    clean_handoffs = layout_info['clean_handoffs']
    
    for side in [5, 3]:
        if target_blunt[side] and donor_blunt[side]:
            junction_status[side] = 'NHEJ'
        elif clean_handoffs[side]:
            junction_status[side] = 'HDR'
        else:
            junction_status[side] = 'uncategorized'
            
    if (junction_status[5] == 'HDR' and
        junction_status[3] == 'HDR'):
        description = '100: HDR'

    elif (junction_status[5] == 'NHEJ' and
          junction_status[3] == 'HDR'):
        description = "200: 5' NHEJ"
    
    elif (junction_status[5] == 'HDR' and
          junction_status[3] == 'NHEJ'):
        description = "300: 3' NHEJ"
    
    elif (junction_status[5] == 'NHEJ' and
          junction_status[3] == 'NHEJ'):
        description = "400: 5' and 3' NHEJ"

    else:
        description = '900: uncategorized'

    junction_status['description'] = description

    layout_info['junction'] = junction_status

def characterize_integration(layout_info, target_info):
    int_int = layout_info['integration_interval']
    parsimonious_als = layout_info['alignments']['parsimonious']
    strand = layout_info['strand']
    donor_relative_to_arm_internal = layout_info['donor_relative_to_arm_internal']
    junction_status = layout_info['junction']

    donor_als = [
        al for al in parsimonious_als
        if al.reference_name == target_info.donor
    ]

    if len(donor_als) == 0:
        source = '900: uncategorized'

        e_coli_als = [al for al in parsimonious_als if al.reference_name == 'e_coli_K12']
        if len(e_coli_als) == 1:
            covered = interval.get_covered(e_coli_als[0])
            if covered.start - int_int.start <= 10 and int_int.end - covered.end <= 10:
                source = '200: e coli'

        layout_info['integration'] = ('unexpected', source)
        return layout_info

    if any(sam.get_strand(al) != strand for al in donor_als):
        layout_info['integration'] = ('unexpected', '100: flipped')
        return layout_info

    if len(donor_als) == 1:
        if junction_status[5] != 'uncategorized' and junction_status[3] != 'uncategorized':
            description = 'full length'
        else:
            fields = []
            for side in [5, 3]:
                if junction_status[side] == 'uncategorized':
                    if donor_relative_to_arm_internal[side] < 0:
                        fields += ["{0}' truncated".format(side)]
                    elif donor_relative_to_arm_internal[side] > 0:
                        fields += ["{0}' extended".format(side)]

            if len(fields) > 0:
                description = '100: ' + ', '.join(fields)
            else:
                description = '900: uncategorized'

    else:
        check_for_concatamer(layout_info, target_info)

        if 'concatamer' in layout_info:
            description = 'concatamer'

        else:
            #TODO: check for plasmid extensions around the boundary
            description = '900: uncategorized'

    layout_info['integration'] = description

    return layout_info

def summarize_outcome(layout_info):
    if 'malformed' in layout_info:
        outcome = ('600: malformed layout', layout_info['malformed'])

    elif not layout_info['has_integration']:
        if layout_info['scar'] > 0:
            outcome = ('100: no integration', '200: scar near cut')
        else:
            outcome = ('100: no integration', '100: no scar')

    else:
        if len(layout_info['integration']) == 2:
            _, description = layout_info['integration']
            outcome = ('500: unexpected integration', description)

        elif layout_info['integration'] == 'full length':
            outcome = ('200: full length', layout_info['junction']['description'])

        elif layout_info['integration'] == 'concatamer':
            outcome = ('300: concatamer', layout_info['junction']['description'])

        else:
            outcome = ('400: unexpected length', layout_info['integration'])

    pairs = [s.split(': ') for s in outcome]
    description = tuple(d for p, d in pairs)
    sort_order = tuple(int(p) for p, d in pairs)
    layout_info['outcome'] = {
        'sort_order': sort_order,
        'description': description,
    }

def check_for_concatamer(layout_info, target_info):
    parsimonious_als = layout_info['alignments']['parsimonious']
    strand = layout_info['strand']
    HAs = target_info.homology_arms

    donor_als = [
        al for al in parsimonious_als
        if al.reference_name == target_info.donor
    ]

    if len(donor_als) <= 1:
        return layout_info

    if strand == '+':
        key = lambda al: interval.get_covered(al).start
        reverse = False
    else:
        key = lambda al: interval.get_covered(al).end
        reverse = True

    five_to_three = sorted(donor_als, key=key, reverse=reverse)
    junctions_clean = []

    for before, after in zip(five_to_three[:-1], five_to_three[1:]):
        adjacent = interval.are_adjacent(interval.get_covered(before), interval.get_covered(after))

        missing_before = HAs['donor', 3].end - (before.reference_end - 1)
        missing_after = after.reference_start - HAs['donor', 5].start

        clean = adjacent and (missing_before == 0) and (missing_after == 0)

        junctions_clean.append(clean)


    if all(junctions_clean):
        layout_info['concatamer'] = len(junctions_clean) + 1

    return layout_info

def max_del_nearby(alignment, ref_pos, window):
    ref_pos_to_block = sam.get_ref_pos_to_block(alignment)
    nearby = range(ref_pos - window, ref_pos + window)
    blocks = [ref_pos_to_block.get(r, (0, 0)) for r in nearby]
    dels = [l for k, l in blocks if k == sam.BAM_CDEL]
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
    ('no integration',
        ('no scar',
         'scar near cut',
        ),
    ),
    ('full length',
        ('HDR',
         '5\' NHEJ',
         '3\' NHEJ',
         '5\' and 3\' NHEJ',
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
    ('unexpected length',
        ('5\' truncated',
         '3\' truncated',
         '5\' extended',
         '3\' extended',
         '5\' truncated, 3\' truncated',
         '5\' extended, 3\' extended',
         '5\' truncated, 3\' extended',
         '5\' extended, 3\' truncated',
         'uncategorized',
        ),
    ),
    ('unexpected integration',
        ('flipped',
         'e coli',
         'uncategorized',
        ),
    ),
    ('malformed layout',
        ('extra copies of primer',
         'missing a primer',
         'primer far from read edge',
         'no alignments detected',
        ),
    ),
]
