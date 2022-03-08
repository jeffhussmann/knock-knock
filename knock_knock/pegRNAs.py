import copy

from collections import defaultdict

import pandas as pd
import pysam

import Bio.SeqUtils

from hits import gff, interval, sam, sw, utilities

from knock_knock import target_info

def read_csv(csv_fn, process=True):
    df = pd.read_csv(csv_fn, index_col='name', comment='#')

    if process:
        component_order = ['protospacer', 'scaffold', 'extension']

        for component in component_order:
            df[component] = df[component].str.upper()

        full_sequences = []
        for _, row in df.iterrows():
            full_sequence = ''.join([row[component] for component in component_order])
            full_sequences.append(full_sequence)

        df['full_sequence'] = full_sequences

        return df.to_dict(orient='index')
    
    else:
        return df

default_feature_colors = {
    'RTT': '#c7b0e3',
    'PBS': '#85dae9',
    'protospacer': '#ff9ccd',
    'scaffold': '#b7e6d7',
    'overlap': '#9EAFD2',
    'extension': '#777777',
}

def PBS_name(pegRNA_name):
    return f'{pegRNA_name}_PBS'

def extract_pegRNA_name(PBS_name):
    return PBS_name.rsplit('_', 1)[0]

def infer_features(pegRNA_name,
                   pegRNA_components,
                   target_name,
                   target_sequence,
                   effector,
                  ):
    '''
    Identifies primer binding site (PBS) and reverse transcription template (RTT) regions
    of a pegRNA/target sequence pair by locating the pegRNA protospacer in the target, then
    finding the longest exact match in the extension region of the pegRNA for the sequence
    immediately upstream of the nick location in the target.

    Returns dictionaries of hits.gff Features for the pegRNA and the target.
    '''

    protospacer = pegRNA_components['protospacer']
    pegRNA_sequence = pegRNA_components['full_sequence']
    
    # Identify which strand of the target sequence the protospacer is
    # present on.

    strands = set()
    
    if protospacer in target_sequence:
        strands.add('+')
    
    if utilities.reverse_complement(protospacer) in target_sequence:
        strands.add('-')

    if len(strands) == 0:
        # Initial base of protospacer might not match.
        protospacer = protospacer[1:]

        if protospacer in target_sequence:
            strands.add('+')
        
        if utilities.reverse_complement(protospacer) in target_sequence:
            strands.add('-')
    
    if len(strands) != 1:
        raise ValueError(f'protospacer not present on exactly 1 strand ({len(strands)})')
        
    strand = strands.pop()
    
    # Confirm that there is a PAM next to the protospacer in the target.    

    if strand == '+':
        ps_5 = target_sequence.index(protospacer)
    else:
        ps_5 = target_sequence.index(utilities.reverse_complement(protospacer)) + len(protospacer) - 1

    PAM_pattern = effector.PAM_pattern

    # Note: implicitly assumes effector.PAM_side == 3.
    if strand == '+':
        PAM_slice = slice(ps_5 + len(protospacer), ps_5 + len(protospacer) + len(PAM_pattern))
        PAM_transform = utilities.identity
        cut_after = PAM_slice.start + effector.cut_after_offset[0]
        
    else:
        PAM_slice = slice(ps_5 - len(protospacer) + 1 - len(PAM_pattern), ps_5 - len(protospacer) + 1)
        PAM_transform = utilities.reverse_complement
        # cut_after calculation is confusing. See similiar code in knock_knock.target_info.TargetInfo.cut_afters
        cut_after = PAM_slice.stop - 1 - effector.cut_after_offset[0] - 1

    PAM = PAM_transform(target_sequence[PAM_slice])
    pattern, *matches = Bio.SeqUtils.nt_search(PAM, PAM_pattern)
    
    if 0 not in matches:
        raise ValueError(f'bad PAM: {PAM} next to {protospacer} (strand {strand})')


    # Identify the PBS region of the pegRNA by finding a match to
    # the sequence of the target immediately before the nick.    
    
    header = pysam.AlignmentHeader.from_references([pegRNA_name, 'target'], [len(pegRNA_sequence), len(target_sequence)])
    # 'ref' is pegRNA and 'query' is target
    mapper = sw.SeedAndExtender(pegRNA_sequence.encode(), 8, header, pegRNA_name)

    target_bytes = target_sequence.encode()

    # Note: assumes PBS perfectly matches at least seed_length nts at the nick.
    seed_length = 7

    if strand == '+':
        seed_start = cut_after + 1 - seed_length
        before_nick_start = 0
        before_nick_end = cut_after
    else:
        before_nick_start = cut_after + 1
        before_nick_end = len(target_sequence) - 1
        seed_start = cut_after + 1

    alignments = mapper.seed_and_extend(target_bytes, seed_start, seed_start + seed_length, pegRNA_name)
    
    valid_alignments = [al for al in alignments if sam.get_strand(al) != strand]
    
    if len(valid_alignments) != 1:
        seed_sequence = target_bytes[seed_start:seed_start + seed_length]
        raise ValueError(f'{pegRNA_name}: not exactly one valid PBS location for {seed_sequence} ({len(valid_alignments)})')
        
    PBS_alignment = valid_alignments[0]

    # Restrict the PBS to not extend past the nick.
    PBS_alignment = sam.crop_al_to_query_int(PBS_alignment, before_nick_start, before_nick_end)

    # Build GFF features of the pegRNA components.

    starts = {}
    starts['protospacer'] = 0
    starts['scaffold'] = starts['protospacer'] + len(pegRNA_components['protospacer'])
    starts['extension'] = starts['scaffold'] + len(pegRNA_components['scaffold'])
    
    ends = {name: starts[name] + len(pegRNA_components[name]) for name in starts}
    
    starts['PBS'] = PBS_alignment.reference_start
    ends['PBS'] = starts['PBS'] + PBS_alignment.query_alignment_length
    
    starts['RTT'] = starts['extension']
    ends['RTT'] = starts['PBS']
    
    # Need to annotate PBS and RTT strands like this to enable
    # identification of shared features.
    strands = {
        'protospacer': '+',
        'scaffold': '+',
        'extension': '+',
        'PBS': '-',
        'RTT': '-',
    }

    # Update pegRNA_components.
    for name in ['PBS', 'RTT']:
        pegRNA_components[name] = pegRNA_sequence[starts[name]:ends[name]]

    pegRNA_features = {
        (pegRNA_name, name): gff.Feature.from_fields(seqname=pegRNA_name,
                                                     start=starts[name],
                                                     end=ends[name] - 1,
                                                     feature='misc', 
                                                     strand=strands[name],
                                                     attribute_string=gff.make_attribute_string({
                                                         'ID': name,
                                                         'color': default_feature_colors[name],
                                                     }),
                                                    )
        for name in starts
    }

    # Build a GFF feature for the PBS in the target.

    target_PBS_start, target_PBS_end = sam.query_interval(PBS_alignment)
    target_PBS_name = f'{pegRNA_name}_PBS'
    target_PBS_feature = gff.Feature.from_fields(seqname=target_name,
                                                 start=target_PBS_start,
                                                 end=target_PBS_end,
                                                 strand=strand,
                                                 feature='misc', 
                                                 attribute_string=gff.make_attribute_string({
                                                     'ID': target_PBS_name,
                                                     'color': default_feature_colors['PBS'],
                                                 }),
                                                )

    target_features = {
        (target_name, target_PBS_name): target_PBS_feature,
    }
    
    return pegRNA_features, target_features

def infer_SNV_features(ti):
    ''' Compatibility with code initially designed for HDR screens wants
    pegRNAs to be annotated with 'homology arms' and 'SNPs'.
    One HA is the PBS, the other is the RTT, and SNPs are features.
    '''

    new_features = {}

    SNV_positions_on_target = defaultdict(list)
                
    for pegRNA_name in ti.pegRNA_names:
        names = {
            'target': ti.target,
            'pegRNA': pegRNA_name,
        }

        features = {
            ('target', 'PBS'): ti.features[ti.target, f'{names["pegRNA"]}_PBS'],
            ('pegRNA', 'PBS'): ti.features[names['pegRNA'], 'PBS'],
            ('pegRNA', 'RTT'): ti.features[names['pegRNA'], 'RTT'],
            ('pegRNA', 'scaffold'): ti.features[names['pegRNA'], 'scaffold'],
        }

        strands = {
            'target': features['target', 'PBS'].strand,
            'pegRNA': '-',
        }

        seqs = {
            ('pegRNA', name): ti.feature_sequence(names['pegRNA'], name)
            for name in ['RTT', 'scaffold']
        }

        # ti.feature_sequence uses strand to RC, so RTT will be RC'ed but not scaffold.
        if features['pegRNA', 'scaffold'].strand == '+':
            seqs['pegRNA', 'scaffold'] = utilities.reverse_complement(seqs['pegRNA', 'scaffold'])

        starts = {
            ('pegRNA', 'RTT'): features['pegRNA', 'RTT'].end,
        }
        ends = {}

        if strands['target'] == '+':
            starts['target', 'RTT'] = features['target', 'PBS'].end + 1
            ends['target', 'RTT'] = starts['target', 'RTT'] + len(features['pegRNA', 'RTT'])

            starts['target', 'scaffold'] = ends['target', 'RTT'] + 1
            ends['target', 'scaffold'] = starts['target', 'scaffold'] + len(features['pegRNA', 'scaffold'])

        else:
            ends['target', 'RTT'] = features['target', 'PBS'].start # Note: ends is exclusive here, so no - 1
            starts['target', 'RTT'] = ends['target', 'RTT'] - len(features['pegRNA', 'RTT'])

            ends['target', 'scaffold'] = starts['target', 'RTT'] # Note: ends is exclusive here, so no - 1
            starts['target', 'scaffold'] = ends['target', 'scaffold'] - len(features['pegRNA', 'scaffold'])

        for name in ['RTT', 'scaffold']:
            seqs['target', name] = ti.target_sequence[starts['target', name]:ends['target', name]]
            if features['target', 'PBS'].strand == '-':
                seqs['target', name] = utilities.reverse_complement(seqs['target', name])

        # pegRNA sequences should always be provided as 5'-3' RNA
        # and therefore have the RTT feature on the - strand. 
        if features['pegRNA', 'RTT'].strand != '-':
            raise ValueError(str(features['pegRNA', 'RTT']))

        SNP_offsets = []

        for i, (pegRNA_b, target_b) in enumerate(zip(seqs['pegRNA', 'RTT'], seqs['target', 'RTT'])):
            if pegRNA_b != target_b:
                SNP_offsets.append(i)
                
        for offset in SNP_offsets:
            SNP_name = f'SNP_{names["pegRNA"]}_{offset}'

            positions = {
                'pegRNA': starts['pegRNA', 'RTT'] - offset,
            }

            if strands['target'] == '+':
                positions['target'] = starts['target', 'RTT'] + offset
            else:
                positions['target'] = ends['target', 'RTT'] - offset - 1

            SNV_base = ti.reference_sequences[names['pegRNA']][positions['pegRNA']]
            # A pegRNA for a forward strand protospacer provides a SNP base that is
            # the opposite of its given strand.
            if strands['target'] == '+':
                SNV_strand = '-'
                SNV_base = utilities.reverse_complement(SNV_base)
            else:
                SNV_strand = '+'

            SNV_positions_on_target[positions['target']].append(
                (names['pegRNA'], positions['pegRNA'], SNV_strand, SNV_base)
            )

            for seq_name in names:
                feature = gff.Feature.from_fields(seqname=names[seq_name],
                                                  start=positions[seq_name],
                                                  end=positions[seq_name],
                                                  strand=strands[seq_name],
                                                 )
                feature.attribute['ID'] = SNP_name
            
                new_features[names[seq_name], SNP_name] = feature

        # When interpreting alignments to pegRNAs, it is useful to know
        # the point in pegRNA sequence at which it first diverges from
        # genomic sequence. Annotate the region of the pegRNA past this point
        # (that is, before it in 5'-to-3' sequence) with a feature
        # named f'after_first_difference_{pegRNA_name}'.

        pegRNA_seq = seqs['pegRNA', 'RTT'] + seqs['pegRNA', 'scaffold']
        target_seq = seqs['target', 'RTT'] + seqs['target', 'scaffold']

        for offset, (pegRNA_b, target_b) in enumerate(zip(pegRNA_seq, target_seq)):
            if pegRNA_b != target_b:
                break

        first_differnce_position = starts['pegRNA', 'RTT'] - offset
        feature = gff.Feature.from_fields(seqname=names['pegRNA'],
                                          start=0,
                                          end=first_differnce_position,
                                          strand='-',
                                         )
        name = f'after_first_difference_{names["pegRNA"]}'
        feature.attribute['ID'] = name
        new_features[names["pegRNA"], name] = feature
            
        HA_PBS_name = f'HA_PBS_{names["pegRNA"]}'
        HA_RTT_name = f'HA_RTT_{names["pegRNA"]}'

        HA_PBS = copy.deepcopy(features['target', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[ti.target, HA_PBS_name] = HA_PBS

        HA_RTT = gff.Feature.from_fields(seqname=ti.target,
                                        start=starts['target', 'RTT'],
                                        end=ends['target', 'RTT'],
                                        strand=HA_PBS.strand,
                                        )
        HA_RTT.attribute['ID'] = HA_RTT_name
        new_features[ti.target, HA_RTT_name] = HA_RTT

        HA_PBS = copy.deepcopy(features['pegRNA', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[names['pegRNA'], HA_PBS_name] = HA_PBS

        HA_RTT = copy.deepcopy(features['pegRNA', 'RTT'])
        HA_RTT.attribute['ID'] = HA_RTT_name
        new_features[names['pegRNA'], HA_RTT_name] = HA_RTT

    SNVs = defaultdict(dict)

    for target_position, pegRNA_list in SNV_positions_on_target.items():
        t = ti.reference_sequences[ti.target][target_position]

        for pegRNA_name, position, strand, d in pegRNA_list:
            name = f'SNV_{target_position}_{t}-{d}'

            SNVs[ti.target][name] = {
                'position': target_position,
                'strand': '+',
                'base': t,
            }

            SNVs[pegRNA_name][name] = {
                'position': position,
                'strand': strand,
                'base': d,
            }

    return new_features, SNVs

def infer_twin_pegRNA_features(ti):

    target_seq = ti.reference_sequences[ti.target]
    primers = ti.primers_by_side_of_target

    pegRNA_names = ti.pegRNA_names_by_side_of_target

    pegRNA_seqs = {side: ti.reference_sequences[ti.pegRNA_names_by_side_of_target[side]] for side in [5, 3]}

    target_PBSs = {side: ti.features[ti.target, ti.PBS_names_by_side_of_target[side]] for side in [5, 3]}

    pegRNA_RTTs = {side: ti.features[ti.pegRNA_names_by_side_of_target[side], 'RTT'] for side in [5, 3]}

    overlap_features = {}
    overlap_seqs = {}
    intended_edit_seqs = {}

    is_prime_del = False

    through_PBS = {
        5: target_seq[primers[5].start:target_PBSs[5].end + 1],
        3: target_seq[target_PBSs[3].start:primers[3].end + 1]
    }

    RTed = {
        5: utilities.reverse_complement(pegRNA_seqs[5][pegRNA_RTTs[5].start:pegRNA_RTTs[5].end + 1]),
        3: pegRNA_seqs[3][pegRNA_RTTs[3].start:pegRNA_RTTs[3].end + 1],
    }

    target_with_RTed = {
        5: through_PBS[5] + RTed[5],
        3: RTed[3] + through_PBS[3],
    }

    # Align the RT'ed part of the 5' pegRNA to the target+RT'ed sequence
    # from the 3' side.

    for length in range(1, len(RTed[5]) + 1):
        suffix = RTed[5][-length:]
        try:
            start = target_with_RTed[3].index(suffix)
        except ValueError:
            length = length - 1
            break

    # How much of the RTed sequence from 5 lines up with non-RTed sequence from 3?
    # Answer is how much of [start, start + length - 1] overlaps with [len(RTed[3]), len(target_with_RTed[3]) - 1]
    # If any, these pegRNAs are a prime del strategy.

    non_RTed_interval = interval.Interval(start, start + length - 1) & interval.Interval(len(RTed[3]), len(target_with_RTed[3]) - 1)
    if len(non_RTed_interval) > 0:
        is_prime_del = True

    # How much of the RTed sequence from 5 lines up with RTed sequence from 3?
    # Since the lined up part begins at index 'start' in target_with_RTed[3], the answer
    # is how much of [start, start + length - 1] overlaps with [0, len(RTed[3]) - 1]

    overlap_interval = interval.Interval(start, start + length - 1) & interval.Interval(0, len(RTed[3]) - 1)

    if len(overlap_interval) > 0:
        overlap_seqs[5] = target_with_RTed[3][overlap_interval.start:overlap_interval.end + 1]
        overlap_interval_on_pegRNA = interval.Interval(pegRNA_RTTs[3].start + overlap_interval.start, pegRNA_RTTs[3].start + overlap_interval.end)

        overlap_feature = gff.Feature.from_fields(seqname=pegRNA_names[3],
                                                  feature='overlap',
                                                  start=overlap_interval_on_pegRNA.start,
                                                  end=overlap_interval_on_pegRNA.end,
                                                  strand='+',
                                                  attribute_string=gff.make_attribute_string({
                                                      'ID': 'overlap',
                                                      'color': default_feature_colors['overlap'],
                                                      'short_name': 'overlap',
                                                  }),
                                                 )

        overlap_features[pegRNA_names[3], 'overlap'] = overlap_feature

    else:
        overlap_seqs[5] = ''

    intended_edit_seqs[5] = target_with_RTed[5][:-length] + target_with_RTed[3][start:]


    # Align the RT'ed part of the 3' pegRNA to the target+RT'ed sequence
    # from the 5' side.

    for length in range(1, len(RTed[3]) + 1):
        prefix = RTed[3][:length]
        try:
            start = target_with_RTed[5].index(prefix)
        except ValueError:
            length = length - 1
            break

    # How much of the RTed sequence from 3 lines up with non-RTed sequence from 5?
    # Answer is how much of [start, start + length - 1] overlaps with [0, len(through_PBS[5]) - 1]
    # If any, these pegRNAs are a prime del strategy.

    non_RTed_interval = interval.Interval(start, start + length - 1) & interval.Interval(0, len(through_PBS[5]) - 1)
    if len(non_RTed_interval) > 0:
        is_prime_del = True

    # How much of the RTed sequence from 3 lines up with RTed sequence from 5?
    # Since the lined up part begins at index 'start' in target_with_RTed[5], the answer
    # is how much of [start, start + length - 1] overlaps with [len(through_PBS[5]), len(target_with_RTed[5]) - 1]

    overlap_interval = interval.Interval(start, start + length - 1) & interval.Interval(len(through_PBS[5]), len(target_with_RTed[5]) - 1)

    if len(overlap_interval) > 0:
        overlap_seqs[3] = target_with_RTed[5][overlap_interval.start:overlap_interval.end + 1]

        overlap_interval_on_pegRNA = interval.Interval(pegRNA_RTTs[5].start + (len(target_with_RTed[5]) - 1 - overlap_interval.end),
                                                       pegRNA_RTTs[5].start + (len(target_with_RTed[5]) - 1 - overlap_interval.start),
                                                      )

        overlap_feature = gff.Feature.from_fields(seqname=pegRNA_names[5],
                                                  feature='overlap',
                                                  start=overlap_interval_on_pegRNA.start,
                                                  end=overlap_interval_on_pegRNA.end,
                                                  strand='-',
                                                  attribute_string=gff.make_attribute_string({
                                                      'ID': 'overlap',
                                                      'color': default_feature_colors['overlap'],
                                                      'short_name': 'overlap',
                                                  }),
                                                 )


        overlap_features[pegRNA_names[5], 'overlap'] = overlap_feature

    else:
        overlap_seqs[3] = ''

    intended_edit_seqs[3] = target_with_RTed[5][:start + length] + target_with_RTed[3][length:]

    if overlap_seqs[5] != overlap_seqs[3]:
        raise ValueError('inconsistent overlaps inferred')

    if intended_edit_seqs[5] != intended_edit_seqs[3]:
        raise ValueError('inconsistent intended edits inferred')
    else:
        intended_edit_seq = intended_edit_seqs[5]

    # Check if the intended edit is a deletion.

    unedited_seq = target_seq[primers[5].start:primers[3].end + 1]

    deletion = None

    if len(intended_edit_seq) < len(unedited_seq):
        for num_matches_at_start, (intended_b, edited_b) in enumerate(zip(unedited_seq, intended_edit_seq)):
            if intended_b != edited_b:
                break

        # If the sequence following the first difference exactly
        # matches the end of the wild type amplicon, the intended
        # edit is a deletion.
        if unedited_seq.endswith(intended_edit_seq[num_matches_at_start + 1:]):
            deletion_length = len(unedited_seq) - len(intended_edit_seq)

            deletion_start = primers[5].start + num_matches_at_start

            deletion = target_info.DegenerateDeletion([deletion_start], deletion_length)
            deletion = ti.expand_degenerate_indel(deletion)
            
    return deletion, overlap_features, is_prime_del

def infer_twin_prime_overlap_old(pegRNA_components):
    '''
    Identify the stretch of exactly matching sequence at the 3'
    ends of the reverse transcription products of two pegRNAs used
    in a twin prime strategy.
    '''
    if len(pegRNA_components) != 2:
        raise ValueError(f'Can only infer pegRNA overlap for 2 pegRNAs ({len(pegRNA_components)} provided)')

    first_seq, second_seq = [pegRNA_components[name]['extension'] for name in pegRNA_components]
    second_seq_rc = utilities.reverse_complement(second_seq)

    # twin pegRNAs should have exactly matching sequence at the 3' ends
    # of their RT products, which corresponds to the beginning of the RTT
    # in one of the (5'-to-3') pegRNA sequences matching the end of the 
    # reverse-complemented RTT of the other (5'-to-3') pegRNA sequence. 
    for start in range(len(first_seq)):
        second_suffix = second_seq_rc[start:]
        first_prefix = first_seq[:len(second_suffix)]
        if first_prefix == second_suffix:
            break

    overlap_length = len(first_prefix)

    overlap_features = {}

    # Arbitrarily assign strands to the two overlap features.
    # It is only importantly that they be opposite.

    for (pegRNA_name, components), strand in zip(pegRNA_components.items(), ['+', '-']):
        RTT_start = components['full_sequence'].index(components['extension'])
        overlap_start = RTT_start
        # Note: gff ends are inclusive, hence - 1 here.
        overlap_end = overlap_start + overlap_length - 1
        overlap_feature = gff.Feature.from_fields(seqname=pegRNA_name,
                                                  feature='overlap',
                                                  start=overlap_start,
                                                  end=overlap_end,
                                                  strand=strand,
                                                  attribute_string=gff.make_attribute_string({
                                                      'ID': 'overlap',
                                                      'color': default_feature_colors['overlap'],
                                                      'short_name': 'overlap',
                                                  }),
                                                 )


        overlap_features[pegRNA_name, 'overlap'] = overlap_feature

    return overlap_length, overlap_features

def infer_twin_pegRNA_intended_deletion_old(ti):
    primers = ti.primers_by_side_of_read
    pegRNA_names = ti.pegRNA_names_by_side_of_read

    if set(pegRNA_names) != {'left', 'right'}:
        return None, None

    PBS_names = {
        side: PBS_name(pegRNA_names[side])
        for side in ['left', 'right']
    }

    seqs = {
        'target': ti.reference_sequences[ti.target],
        'left_pegRNA': ti.reference_sequences[pegRNA_names['left']],
        'right_pegRNA': ti.reference_sequences[pegRNA_names['right']],
    }

    if ti.sequencing_direction == '+':
        before_pegRNAs = seqs['target'][primers['left'].start:ti.features[ti.target, PBS_names['left']].start]
    else:
        before_pegRNAs = seqs['target'][ti.features[ti.target, PBS_names['left']].end + 1:primers['left'].end + 1]
        before_pegRNAs = utilities.reverse_complement(before_pegRNAs)
        
    from_left_pegRNA = seqs['left_pegRNA'][ti.features[pegRNA_names['left'], 'overlap'].start:ti.features[pegRNA_names['left'], 'PBS'].end + 1]
    from_left_pegRNA = utilities.reverse_complement(from_left_pegRNA)
    
    from_right_pegRNA = seqs['right_pegRNA'][ti.features[pegRNA_names['right'], 'overlap'].end + 1: ti.features[pegRNA_names['right'], 'PBS'].end + 1]
        
    if ti.sequencing_direction == '+':
        after_pegRNAs = seqs['target'][ti.features[ti.target, PBS_names['right']].end + 1:primers['right'].end + 1]
    else:
        after_pegRNAs = seqs['target'][primers['right'].start:ti.features[ti.target, PBS_names['right']].start]
        after_pegRNAs = utilities.reverse_complement(after_pegRNAs)
        
    intended_seq = before_pegRNAs + from_left_pegRNA + from_right_pegRNA + after_pegRNAs

    if len(intended_seq) >= len(ti.wild_type_amplicon_sequence):
        # Not a deletion if it doesn't reduce length. Without this sanity check,
        # logic below would be more complicated.
        deletion_feature = None
        deletion = None
    else:
        for i, (edited_b, original_b) in enumerate(zip(intended_seq, ti.wild_type_amplicon_sequence)):
            if edited_b != original_b:
                break
                
        num_matches_at_start = i
                
        remaining = intended_seq[num_matches_at_start + 1:]

        # If the sequence following the first difference exactly
        # matches the end of the wild type amplicon, the intended
        # edit is a deletion.

        if ti.wild_type_amplicon_sequence.endswith(remaining):
            deletion_length = len(ti.wild_type_amplicon_sequence) - len(intended_seq)

            if ti.sequencing_direction == '+':
                deletion_start = primers['left'].start + num_matches_at_start
                deletion_end = deletion_start + deletion_length - 1
            else:
                deletion_end = primers['left'].end - num_matches_at_start
                deletion_start = deletion_end - deletion_length + 1

            deletion_feature = gff.Feature.from_fields(start=deletion_start,
                                                       end=deletion_end,
                                                       ID=f'intended_deletion_{pegRNA_names["left"]}_{pegRNA_names["right"]}',
                                                      )
            deletion = target_info.DegenerateDeletion([deletion_start], deletion_length)
            deletion = ti.expand_degenerate_indel(deletion)
        else:
            deletion_feature = None
            deletion = None

    return deletion, deletion_feature