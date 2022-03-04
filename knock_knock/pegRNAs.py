import copy

import pandas as pd
import pysam

import Bio.SeqUtils

from hits import gff, sam, sw, utilities

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

        if features['target', 'PBS'].strand == '+':
            starts['target', 'RTT'] = features['target', 'PBS'].end + 1
            ends['target', 'RTT'] = starts['target', 'RTT'] + len(features['pegRNA', 'RTT'])

            starts['target', 'scaffold'] = ends['target', 'RTT'] + 1
            ends['target', 'scaffold'] = starts['target', 'scaffold'] + len(features['pegRNA', 'scaffold'])

            target_offset_sign = 1

        else:
            ends['target', 'RTT'] = features['target', 'PBS'].start # Note: ends is exclusive here, so no - 1
            starts['target', 'RTT'] = ends['target', 'RTT'] - len(features['pegRNA', 'RTT'])

            ends['target', 'scaffold'] = starts['target', 'RTT'] # Note: ends is exclusive here, so no - 1
            starts['target', 'scaffold'] = ends['target', 'scaffold'] - len(features['pegRNA', 'scaffold'])

            target_offset_sign = -1

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
                'target': starts['target', 'RTT'] + offset * target_offset_sign,
                'pegRNA': starts['pegRNA', 'RTT'] - offset,
            }

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

    return new_features

def infer_twin_prime_overlap(pegRNA_components):
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

def infer_prime_del_intended_deletion(ti):
    pass

def infer_twin_pegRNA_intended_deletion(ti):
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