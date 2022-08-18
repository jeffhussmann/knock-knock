import copy
import logging

from collections import defaultdict

import pandas as pd
import pysam

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
    'protospacer': 'lightgrey',
    'scaffold': '#b7e6d7',
    'overlap': '#9EAFD2',
    'extension': '#777777',
    'insertion': '#b1ff67',
    'deletion': 'darkgrey',
}

def PBS_name(pegRNA_name):
    return f'{pegRNA_name}_PBS'

def protospacer_name(pegRNA_name):
    return f'{pegRNA_name}_protospacer'

def extract_pegRNA_name(PBS_name):
    return PBS_name.rsplit('_', 1)[0]

def infer_features(pegRNA_name,
                   pegRNA_components,
                   target_name,
                   target_sequence,
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
    effector = target_info.effectors[pegRNA_components['effector']]
    
    # Identify which strand of the target sequence the protospacer is present on.

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
        protospacer_start = target_sequence.index(protospacer)
    else:
        protospacer_start = target_sequence.index(utilities.reverse_complement(protospacer))

    protospacer_end = protospacer_start + len(protospacer) - 1

    target_protospacer_name = protospacer_name(pegRNA_name)
    target_protospacer_feature = gff.Feature.from_fields(seqname=target_name,
                                                         start=protospacer_start,
                                                         end=protospacer_end,
                                                         strand=strand,
                                                         feature='sgRNA', 
                                                         attribute_string=gff.make_attribute_string({
                                                             'ID': target_protospacer_name,
                                                             'color': default_feature_colors['protospacer'],
                                                             'effector': effector.name,
                                                         }),
                                                        )

    cut_after = effector.cut_afters(target_protospacer_feature)[strand]

    if not effector.PAM_matches_pattern(target_protospacer_feature, target_sequence):
        raise ValueError(f'bad PAM next to {target_protospacer_name} (strand {strand})')

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
    
    extension_ref_p_start = len(pegRNA_components['protospacer'] + pegRNA_components['scaffold'])
    extension_ref_interval = interval.Interval(extension_ref_p_start, len(pegRNA_sequence) - 1)
    def overlaps_extension(al):
        return interval.get_covered_on_ref(al) & extension_ref_interval

    valid_alignments = [al for al in alignments if sam.get_strand(al) != strand and overlaps_extension(al)]

    def priority_key(al):
        # Prioritize longer matches, then matches closer to the 3' end.
        return (al.query_alignment_length, al.reference_start)

    valid_alignments = sorted(valid_alignments, key=priority_key, reverse=True)

    if len(valid_alignments) != 1:
        seed_sequence = target_bytes[seed_start:seed_start + seed_length]
        starts = [al.reference_start for al in valid_alignments]
        warning_message = [f'{pegRNA_name}: not exactly one valid PBS alignment for {seed_sequence}'] + \
        [f'\tlength: {al.query_alignment_length}, start in pegRNA: {al.reference_start}, start in target: {al.query_alignment_start}' for al in valid_alignments]
        logging.warning('\n'.join(warning_message))
        
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

    # Build PBS feature on the target.

    target_PBS_start, target_PBS_end = sam.query_interval(PBS_alignment)
    target_PBS_name = PBS_name(pegRNA_name)
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
        (target_name, target_protospacer_name): target_protospacer_feature,
    }
    
    return pegRNA_features, target_features

def infer_edit_features(pegRNA_names,
                        target_name,
                        existing_features,
                        reference_sequences,
                       ):
    ''' Requires features to already include results from infer_features.
    
    Compatibility with code initially designed for HDR screens wants
    pegRNAs to be annotated with 'homology arms' and 'SNPs'.
    One HA is the PBS, the other is the RT, and SNPs are features.
    '''

    target_sequence = reference_sequences[target_name]

    new_features = {}

    SNV_positions_on_target = defaultdict(list)
                
    for pegRNA_name in pegRNA_names:
        names = {
            'target': target_name,
            'pegRNA': pegRNA_name,
        }

        features = {
            ('target', 'PBS'): existing_features[names['target'], PBS_name(names['pegRNA'])],
            ('target', 'protospacer'): existing_features[names['target'], protospacer_name(names['pegRNA'])],
            ('pegRNA', 'PBS'): existing_features[names['pegRNA'], 'PBS'],
            ('pegRNA', 'RTT'): existing_features[names['pegRNA'], 'RTT'],
            ('pegRNA', 'scaffold'): existing_features[names['pegRNA'], 'scaffold'],
        }

        strands = {
            'target': features['target', 'PBS'].strand,
            'pegRNA': '-',
        }

        seqs = {
            ('pegRNA', name): features['pegRNA', name].sequence(reference_sequences)
            for name in ['RTT', 'scaffold']
        }

        # feature sequence lookup uses strand to RC, so RTT will be RC'ed but not scaffold.
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
            seqs['target', name] = target_sequence[starts['target', name]:ends['target', name]]
            if features['target', 'PBS'].strand == '-':
                seqs['target', name] = utilities.reverse_complement(seqs['target', name])

        # pegRNA sequences should always be provided as 5'-3' RNA
        # and therefore have the RTT feature on the - strand. 
        if features['pegRNA', 'RTT'].strand != '-':
            raise ValueError(str(features['pegRNA', 'RTT']))

        # Determine if this pegRNA programs an insertion or deletion.
        # For an insertion, some suffix of the RT'ed sequence
        # should exactly match the target sequence immediately 
        # after the nick.
        # For a deletion, the entire RT'ed sequence should exactly
        # match the target sequence somewhere downstream of the nick.
        # (These assumptions may need to be relaxed.)

        pegRNA_seq = reference_sequences[pegRNA_name]

        pegRNA_RTT = existing_features[pegRNA_name, 'RTT']

        is_programmed_insertion = False
        is_programmed_deletion = False

        protospacer = features['target', 'protospacer']
        effector = target_info.effectors[protospacer.attribute['effector']]
        cut_after = effector.cut_afters(protospacer)[protospacer.strand]

        min_complementary_length_after_insertion = 8
        if protospacer.strand == '+':
            RTed = seqs['pegRNA', 'RTT']

            target_after_nick = target_sequence[cut_after + 1:]

            # Start at 1 to insist on non-empty insertion.
            for i in range(1, len(RTed) - min_complementary_length_after_insertion):
                possible_match = RTed[i:]
                if target_after_nick.startswith(possible_match):
                    is_programmed_insertion = True
                    break

            # Start at 1 to insist on non-empty deletion.
            if RTed in target_after_nick[1:]:
                is_programmed_deletion = True

            if is_programmed_insertion:
                starts['target', 'HA_RT'] = cut_after + 1
                ends['target', 'HA_RT'] = starts['target', 'HA_RT'] + len(possible_match) - 1
            elif is_programmed_deletion:
                deletion_start = cut_after + 1
                deletion_length = target_after_nick.index(RTed)
                starts['target', 'HA_RT'] = cut_after + 1 + deletion_length
                ends['target', 'HA_RT'] = starts['target', 'HA_RT'] + len(RTed) - 1
            else:
                starts['target', 'HA_RT'] = cut_after + 1
                ends['target', 'HA_RT'] = starts['target', 'HA_RT'] + len(RTed) - 1
                
        else:
            RTed = utilities.reverse_complement(seqs['pegRNA', 'RTT'])

            target_before_nick = target_sequence[:cut_after + 1]

            # Start at 1 to insist on non-empty insertion.
            for i in range(1, len(RTed) - min_complementary_length_after_insertion):
                possible_match = RTed[:len(RTed) - i]
                if target_before_nick.endswith(possible_match):
                    is_programmed_insertion = True
                    break

            # Only go up to -1 to insist on non-empty deletion.
            if RTed in target_before_nick[:-1]:
                is_programmed_deletion = True

            if is_programmed_insertion:
                ends['target', 'HA_RT'] = cut_after
                starts['target', 'HA_RT'] = ends['target', 'HA_RT'] - len(possible_match) + 1
            elif is_programmed_deletion:
                # Note: this is untested.
                deletion_start = target_before_nick.index(RTed) + len(RTed)
                deletion_length = cut_after - deletion_start + 1
                ends['target', 'HA_RT'] = deletion_start - 1
                starts['target', 'HA_RT'] = ends['target', 'HA_RT'] - len(RTed) + 1
            else:
                ends['target', 'HA_RT'] = cut_after
                starts['target', 'HA_RT'] = ends['target', 'HA_RT'] - len(RTed) + 1

        if is_programmed_insertion and is_programmed_deletion:
            raise ValueError('pegRNA programs both insertion and deletion')

        deletion = None

        if is_programmed_insertion:
            starts['pegRNA', 'HA_RT'] = features['pegRNA', 'RTT'].start
            ends['pegRNA', 'HA_RT'] = starts['pegRNA', 'HA_RT'] + len(possible_match) - 1
            
            insertion_length = len(RTed) - len(possible_match)
            starts['pegRNA', 'insertion'] = ends['pegRNA', 'HA_RT'] + 1
            ends['pegRNA', 'insertion'] = starts['pegRNA', 'insertion'] + insertion_length - 1
            
            insertion_name = f'insertion_{pegRNA_name}'
            insertion = gff.Feature.from_fields(seqname=names['pegRNA'],
                                                start=starts['pegRNA', 'insertion'],
                                                end=ends['pegRNA', 'insertion'],
                                                strand='-',
                                                ID=insertion_name,
                                               )
            insertion.attribute['color'] = default_feature_colors['insertion']
            new_features[names['pegRNA'], insertion_name] = insertion

        elif is_programmed_deletion:
            starts['pegRNA', 'HA_RT'] = pegRNA_RTT.start
            ends['pegRNA', 'HA_RT'] = starts['pegRNA', 'HA_RT'] + len(RTed) - 1

            starts['target', 'deletion'] = deletion_start
            ends['target', 'deletion'] = deletion_start + deletion_length - 1

            deletion = target_info.DegenerateDeletion([deletion_start], deletion_length)

            deletion_name = f'deletion_{pegRNA_name}'
            deletion_feature = gff.Feature.from_fields(seqname=names['target'],
                                                       start=starts['target', 'deletion'],
                                                       end=ends['target', 'deletion'],
                                                       strand='+',
                                                       ID=deletion_name,
                                                      )
            deletion_feature.attribute['color'] = default_feature_colors['deletion']
            new_features[names['target'], deletion_name] = deletion_feature

        else:
            starts['pegRNA', 'HA_RT'] = pegRNA_RTT.start
            ends['pegRNA', 'HA_RT'] = starts['pegRNA', 'HA_RT'] + len(RTed) - 1
                    
            # If neither a programmed insertion nor deletion was found, annotate SNVs.

            for offset, (pegRNA_b, target_b) in enumerate(zip(seqs['pegRNA', 'RTT'], seqs['target', 'RTT'])):
                if pegRNA_b != target_b:
                    SNP_name = f'SNP_{names["pegRNA"]}_{offset:03d}'

                    positions = {
                        'pegRNA': starts['pegRNA', 'RTT'] - offset,
                    }

                    if strands['target'] == '+':
                        positions['target'] = starts['target', 'RTT'] + offset
                    else:
                        positions['target'] = ends['target', 'RTT'] - offset - 1

                    SNV_base = reference_sequences[names['pegRNA']][positions['pegRNA']]
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
                                                          ID=SNP_name,
                                                         )
                    
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
        name = f'after_first_difference_{names["pegRNA"]}'
        feature = gff.Feature.from_fields(seqname=names['pegRNA'],
                                          start=0,
                                          end=first_differnce_position,
                                          strand='-',
                                          ID=name,
                                         )
        new_features[names['pegRNA'], name] = feature


        HA_PBS_name = f'HA_PBS_{names["pegRNA"]}'
        HA_RT_name = f'HA_RT_{names["pegRNA"]}'

        # Make target HA features.

        HA_PBS = copy.deepcopy(features['target', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[target_name, HA_PBS_name] = HA_PBS

        HA_RT = gff.Feature.from_fields(seqname=target_name,
                                        start=starts['target', 'HA_RT'],
                                        end=ends['target', 'HA_RT'],
                                        strand=HA_PBS.strand,
                                        ID=HA_RT_name,
                                       )
        HA_RT.attribute['color'] = default_feature_colors['RTT']
        new_features[target_name, HA_RT_name] = HA_RT

        # Make pegRNA HA features.

        HA_PBS = copy.deepcopy(features['pegRNA', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[names['pegRNA'], HA_PBS_name] = HA_PBS

        HA_RT = gff.Feature.from_fields(seqname=names['pegRNA'],
                                        start=starts['pegRNA', 'HA_RT'],
                                        end=ends['pegRNA', 'HA_RT'],
                                        strand='-',
                                        ID=HA_RT_name,
                                       )
        HA_RT.attribute['color'] = default_feature_colors['RTT']
        new_features[names['pegRNA'], HA_RT_name] = HA_RT

    SNVs = defaultdict(dict)

    for target_position, pegRNA_list in SNV_positions_on_target.items():
        t = target_sequence[target_position]

        for pegRNA_name, position, strand, d in pegRNA_list:
            name = f'SNV_{target_position}_{t}-{d}'

            SNVs[target_name][name] = {
                'position': target_position,
                'strand': '+',
                'base': t,
            }

            SNVs[pegRNA_name][name] = {
                'position': position,
                'strand': strand,
                'base': d,
            }

    return new_features, SNVs, deletion

def PBS_names_by_side_of_target(pegRNA_names,
                                target_name,
                                existing_features,
                               ):
    PBS_features_by_strand = {}

    for pegRNA_name in pegRNA_names:
        PBS_feature = existing_features[target_name, PBS_name(pegRNA_name)]
        if PBS_feature.strand in PBS_features_by_strand:
            raise ValueError('pegRNAs target same strand')
        else:
            PBS_features_by_strand[PBS_feature.strand] = PBS_feature

    if len(PBS_features_by_strand) == 2:
        if PBS_features_by_strand['+'].start > PBS_features_by_strand['-'].end:
            raise ValueError('pegRNAs not in PAM-in configuration')

    by_side = {}

    strand_to_side = {
        '+': 5,
        '-': 3,
    }

    for strand, side in strand_to_side.items():
        if strand in PBS_features_by_strand:
            by_side[side] = PBS_features_by_strand[strand].attribute['ID']

    return by_side

def pegRNA_names_by_side_of_target(pegRNA_names,
                                   target_name,
                                   existing_features,
                                  ):
    PBS_names = PBS_names_by_side_of_target(pegRNA_names, target_name, existing_features)
    return {side: extract_pegRNA_name(PBS_name) for side, PBS_name in PBS_names.items()}

def infer_twin_pegRNA_features(pegRNA_names,
                               target_name,
                               existing_features,
                               reference_sequences,
                              ):

    target_seq = reference_sequences[target_name]

    PBS_names_by_side = PBS_names_by_side_of_target(pegRNA_names, target_name, existing_features)
    pegRNA_names_by_side = pegRNA_names_by_side_of_target(pegRNA_names, target_name, existing_features)

    pegRNA_seqs = {side: reference_sequences[pegRNA_names_by_side[side]] for side in [5, 3]}

    target_PBSs = {side: existing_features[target_name, PBS_names_by_side[side]] for side in [5, 3]}

    pegRNA_RTTs = {side: existing_features[pegRNA_names_by_side[side], 'RTT'] for side in [5, 3]}

    new_features = {}
    overlap_seqs = {}
    intended_edit_seqs = {}

    is_prime_del = False

    through_PBS = {
        5: target_seq[:target_PBSs[5].end + 1],
        3: target_seq[target_PBSs[3].start:]
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

        overlap_feature = gff.Feature.from_fields(seqname=pegRNA_names_by_side[3],
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

        new_features[pegRNA_names_by_side[3], 'overlap'] = overlap_feature

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

        overlap_feature = gff.Feature.from_fields(seqname=pegRNA_names_by_side[5],
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


        new_features[pegRNA_names_by_side[5], 'overlap'] = overlap_feature

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

    unedited_seq = target_seq

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

            deletion_start = num_matches_at_start

            deletion = target_info.DegenerateDeletion([deletion_start], deletion_length)

            deletion_name = f'deletion_{pegRNA_names[0]}_{pegRNA_names[1]}'
            deletion_feature = gff.Feature.from_fields(seqname=target_name,
                                                       start=deletion_start,
                                                       end=deletion_start + deletion_length - 1,
                                                       strand='+',
                                                       ID=deletion_name,
                                                      )
            deletion_feature.attribute['color'] = default_feature_colors['deletion']

            new_features[target_name, deletion_name] = deletion_feature
            
    results = {
        'deletion': deletion,
        'new_features': new_features,
        'is_prime_del': is_prime_del,
        'intended_edit_seq': intended_edit_seq,
    }

    return results