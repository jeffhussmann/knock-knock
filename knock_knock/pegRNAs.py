import copy
import logging

from collections import defaultdict

import Bio.Align
import pandas as pd
import pysam

from hits import gff, interval, sam, sw, utilities

import knock_knock.utilities
from knock_knock import target_info

def read_csv(csv_fn, process=True):
    df = knock_knock.utilities.read_and_sanitize_csv(csv_fn, index_col='name')

    if process:
        component_order = ['protospacer', 'scaffold', 'extension']

        for component in component_order:
            # Files only containing non-pegRNAs may omit scaffold
            # and extension columns.
            if component not in df.columns:
                df[component] = ''

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
    'HA_RT': '#c542f5',
    'PBS': '#85dae9',
    'protospacer': 'lightgrey',
    'scaffold': '#b7e6d7',
    'overlap': '#9eafd2',
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

def identify_protospacer_in_target(target_sequence, protospacer, effector):
    ''' Find an occurence of protospacer on either strand of target_sequence
    that has a PAM for effector positioned appropriately. If there is more
    than one such occurence, raise a ValueError. 
    Because the first nt of protospacer might be a non-matching G, first
    look for the whole protospacer. If no match is found, look for the
    protospacer with the first nt removed.
    '''

    if isinstance(effector, str):
        effector = target_info.effectors[effector]

    def find(protospacer_suffix):
        valid_features = []
        for strand, ps_seq in [('+', protospacer_suffix), ('-', utilities.reverse_complement(protospacer_suffix))]:
            protospacer_starts = utilities.find_all_substring_starts(target_sequence, ps_seq)
            
            for protospacer_start in protospacer_starts:
                protospacer_end = protospacer_start + len(ps_seq) - 1
                target_protospacer_feature = gff.Feature.from_fields(start=protospacer_start,
                                                                     end=protospacer_end,
                                                                     strand=strand,
                                                                     feature='sgRNA', 
                                                                     attribute_string=gff.make_attribute_string({
                                                                         'color': default_feature_colors['protospacer'],
                                                                         'effector': effector.name,
                                                                     }),
                                                                    )
                
                if effector.PAM_matches_pattern(target_protospacer_feature, target_sequence):
                    valid_features.append(target_protospacer_feature)

        return valid_features
                
    valid_features = find(protospacer)

    if len(valid_features) == 0:
        valid_features = find(protospacer[1:])

    if len(valid_features) != 1:
        raise ValueError(f'{len(valid_features)} valid locations for protospacer {protospacer} in target {target_sequence if len(target_sequence) < 1000 else ">1kb long"}')
    else:
        valid_feature = valid_features[0]
        return valid_feature

def get_RTT_aligner():
    aligner = Bio.Align.PairwiseAligner()

    # Idea: 'global' mode with no penalty for target right gaps
    # requires the entire flap to be aligned without penalizing
    # the inclusion of extra genomic sequence at the end of the
    # query.  

    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -3
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -0.1

    aligner.target_right_open_gap_score = 0
    aligner.target_right_extend_gap_score = 0

    return aligner

def trim_excess_target_from_alignment(alignment):
    # Trim off excess target sequence to make text representation
    # of the alignment clearer.  
    max_flap_column = alignment.inverse_indices[0][-1]
    trimmed_alignment = alignment[:, :max_flap_column + 1 + 10]
    return trimmed_alignment
    
class pegRNA:
    def __init__(self, name, components, target_name, target_sequence, max_deletion_length=None):
        self.name = name
        self.components = components
        self.target_name = target_name
        self.target_sequence = target_sequence
        self.target_bytes = target_sequence.encode()

        self.reference_sequences = {
            self.name: self.components['full_sequence'],
            self.target_name: self.target_sequence,
        }

        self.features = {}

        self.PBS_name = PBS_name(self.name)
        self.protospacer_name = protospacer_name(self.name)

        self.SNVs = None
        self.deletion = None
        self.insertion = None

        self.max_deletion_length = max_deletion_length

        self.infer_PBS_and_RTT_features()

    def infer_PBS_and_RTT_features(self):
        '''
        Identifies primer binding site (PBS) and reverse transcription template (RTT) regions
        of a pegRNA/target sequence pair by locating the pegRNA protospacer in the target, then
        finding the longest exact match in the extension region of the pegRNA for the sequence
        immediately upstream of the nick location in the target.

        Returns dictionaries of hits.gff Features for the pegRNA and the target.
        '''

        protospacer = self.components['protospacer']
        scaffold = self.components['scaffold']
        pegRNA_sequence = self.components['full_sequence']
        effector = target_info.effectors[self.components['effector']]

        target_protospacer_feature = identify_protospacer_in_target(self.target_sequence, protospacer, effector)
        target_protospacer_feature.attribute['ID'] = self.protospacer_name
        target_protospacer_feature.seqname = self.target_name
        strand = target_protospacer_feature.strand
        
        cut_afters = effector.cut_afters(target_protospacer_feature)
        try:
            cut_after = cut_afters[strand]
        except KeyError:
            # To support PE nuclease strategies, allow for blunt-cutting effectors.
            cut_after = cut_afters['both']

        # Identify the PBS region of the pegRNA by finding a match to
        # the sequence of the target immediately before the nick.    
        
        header = pysam.AlignmentHeader.from_references([self.name, 'target'], [len(pegRNA_sequence), len(self.target_sequence)])
        # 'ref' is pegRNA and 'query' is target
        mapper = sw.SeedAndExtender(pegRNA_sequence.encode(), 8, header, self.name)

        # Note: assumes PBS perfectly matches at least seed_length nts at the nick.
        seed_length = 7

        if strand == '+':
            seed_start = cut_after + 1 - seed_length
            before_nick_start = 0
            before_nick_end = cut_after
        else:
            before_nick_start = cut_after + 1
            before_nick_end = len(self.target_sequence) - 1
            seed_start = cut_after + 1

        alignments = mapper.seed_and_extend(self.target_bytes, seed_start, seed_start + seed_length, self.name)
        
        extension_ref_p_start = len(protospacer + scaffold)
        extension_ref_interval = interval.Interval(extension_ref_p_start, len(pegRNA_sequence) - 1)
        def overlaps_extension(al):
            return interval.get_covered_on_ref(al) & extension_ref_interval

        valid_alignments = [al for al in alignments if sam.get_strand(al) != strand and overlaps_extension(al)]

        def priority_key(al):
            # Prioritize longer matches, then matches closer to the 3' end.
            return (al.query_alignment_length, al.reference_start)

        valid_alignments = sorted(valid_alignments, key=priority_key, reverse=True)

        if len(valid_alignments) == 0:
            seed_sequence = self.target_bytes[seed_start:seed_start + seed_length]
            starts = [al.reference_start for al in valid_alignments]
            warning_message = [
                f'{self.name}: {len(valid_alignments)} valid PBS alignment(s) for {seed_sequence}:'
            ]
            for al in alignments:
                warning_message.append(
                    f'\tlength: {al.query_alignment_length}, start in pegRNA: {al.reference_start} {sam.get_strand(al)}, start in target: {al.query_alignment_start}'
                )
            warning_message = '\n'.join(warning_message)
            logging.warning(warning_message)
            
        PBS_alignment = valid_alignments[0]

        # Restrict the PBS to not extend past the nick.
        PBS_alignment = sam.crop_al_to_query_int(PBS_alignment, before_nick_start, before_nick_end)

        # Build GFF features of the pegRNA components.

        starts = {}
        starts['protospacer'] = 0
        starts['scaffold'] = starts['protospacer'] + len(protospacer)
        starts['extension'] = starts['scaffold'] + len(scaffold)
        
        ends = {name: starts[name] + len(self.components[name]) for name in starts}
        
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
            self.components[name] = pegRNA_sequence[starts[name]:ends[name]]

        self.features.update({
            (self.name, name): gff.Feature.from_fields(seqname=self.name,
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
        })

        # Build PBS feature on the target.

        target_PBS_start, target_PBS_end = sam.query_interval(PBS_alignment)
        target_PBS_feature = gff.Feature.from_fields(seqname=self.target_name,
                                                    start=target_PBS_start,
                                                    end=target_PBS_end,
                                                    strand=strand,
                                                    feature='misc', 
                                                    attribute_string=gff.make_attribute_string({
                                                        'ID': self.PBS_name,
                                                        'color': default_feature_colors['PBS'],
                                                    }),
                                                    )

        self.features.update({
            (self.target_name, self.PBS_name): target_PBS_feature,
            (self.target_name, self.protospacer_name): target_protospacer_feature,
        })

    @utilities.memoized_property
    def intended_flap_sequence(self):
        return utilities.reverse_complement(self.components['RTT'])

    @utilities.memoized_property
    def cut_after(self):
        protospacer = self.features[self.target_name, self.protospacer_name]
        effector = target_info.effectors[protospacer.attribute['effector']]
        try:
            cut_after = effector.cut_afters(protospacer)[protospacer.strand]
        except KeyError:
            # PE nuclease support
            cut_after = effector.cut_afters(protospacer)['both']

        return cut_after

    @utilities.memoized_property
    def target_downstream_of_nick(self):
        protospacer = self.features[self.target_name, self.protospacer_name]

        if protospacer.strand == '+':
            target_downstream_of_nick = self.target_sequence[self.cut_after + 1:]
        else:
            target_downstream_of_nick = utilities.reverse_complement(self.target_sequence[:self.cut_after + 1])

        if self.max_deletion_length is not None:
            target_downstream_of_nick = target_downstream_of_nick[:self.max_deletion_length + len(self.intended_flap_sequence)]

        return target_downstream_of_nick

    def align_RTT_to_target(self):
        ''' Align the intended flap sequence to genomic sequence downstream
        of the nick using Biopython's dynamic programming.
        ''' 

        aligner = get_RTT_aligner()

        alignments = aligner.align(self.intended_flap_sequence, self.target_downstream_of_nick)

        return alignments

    def print_best_alignments(self):
        alignments = self.align_RTT_to_target()
        for alignment in alignments:
            print(trim_excess_target_from_alignment(alignment))

    def extract_edits_from_alignment(self, verbose=False):
        ''' Align the intended flap sequence to genomic sequence downstream
        of the nick using Biopython's dynamic programming.
        Return representations of SNVs, deletions, insertions, and
        of the exactly-matching terminal HA_RT in coordinates relative to the 
        start of the flap and the nick, respectively. 
        ''' 

        alignments = self.align_RTT_to_target()

        best_alignment = alignments[0]

        # Trim off excess target sequence to make text representation
        # of the alignment clearer.  
        max_flap_column = best_alignment.inverse_indices[0][-1]
        best_alignment = best_alignment[:, :max_flap_column + 1 + 10]

        if verbose:
            print(best_alignment)

        SNVs = {}
        deletions = []
        insertions = []

        flap_subsequences, target_subsequences = best_alignment.aligned

        # flap_subsequences and target_subsequences are arrays of length-2 arrays
        # that indicate the start and (exclusive) ends of subsequences of the flap
        # and target that are aligned to each other.

        # Track mismatch positions within each subsequence to allow definition
        # of the exactly-matching HA_RT section of the final subsequence later.
        mismatches_in_subsequences = []

        # Compare the sequences of each aligned flap and target subsequences to
        # identify programmed SNVs.
        for flap_subsequence, target_subsequence in zip(flap_subsequences, target_subsequences):
            flap_ps = range(*flap_subsequence)
            target_ps = range(*target_subsequence)
            
            mismatches = [-1]
            
            for i, (flap_p, target_p) in enumerate(zip(flap_ps, target_ps)):
                flap_b = self.intended_flap_sequence[flap_p]
                target_b = self.target_downstream_of_nick[target_p]
                if flap_b != target_b:
                    mismatches.append(i)
                    
                    SNV_name = f'SNV_{target_p}_{target_b}_{flap_p}_{flap_b}'
                    SNVs[SNV_name] = {
                        'flap': flap_p,
                        'target_downstream': target_p,
                    }
                    
            mismatches_in_subsequences.append(mismatches)
            
        after_last_mismatch_offset = mismatches_in_subsequences[-1][-1] + 1

        # HA_RT is the part of the last subsequence following the last mismatch,
        # or the entire last subsequence if it doesn't contain a mismatch.

        HA_RT = {
            'flap': (flap_subsequences[-1][0] + after_last_mismatch_offset, flap_subsequences[-1][1] - 1),
            'target_downstream': (target_subsequences[-1][0] + after_last_mismatch_offset, target_subsequences[-1][1] - 1),
        }

        # Insertions are gaps between consecutive flap subsequences. If the
        # first subsequence doesn't start at 0, the flap begins with an insertion.

        if flap_subsequences[0][0] != 0:
            insertion = {
                'start_in_flap': 0,
                'end_in_flap': flap_subsequences[0][0] - 1,
                'starts_after_in_downstream': -1,
                'ends_before_in_downstream': 0,
            }
            insertions.append(insertion)
            
        for (_, left_flap_end), (right_flap_start, _), (_, left_target_end), (right_target_start, _) in zip(flap_subsequences, flap_subsequences[1:], target_subsequences, target_subsequences[1:]):
            if left_flap_end != right_flap_start:
                insertion = {
                    'start_in_flap': left_flap_end,
                    'end_in_flap': right_flap_start - 1,
                    'starts_after_in_downstream': left_target_end - 1,
                    'ends_before_in_downstream': right_target_start,
                }
                insertions.append(insertion)

        # Deletions are gaps between consecutive target subsequences. If the
        # first subsequence doesn't start at 0, sequence immediately after the
        # nick is deleted.

        if target_subsequences[0][0] != 0:
            deletions.append((0, target_subsequences[0][0] - 1))
                
        for (_, left_target_end), (right_target_start, _) in zip(target_subsequences, target_subsequences[1:]):
            if left_target_end != right_target_start:
                deletions.append((left_target_end, right_target_start - 1))
                
        return SNVs, deletions, insertions, HA_RT, (flap_subsequences, target_subsequences)

    def infer_edit_features(self):
        ''' 
        Note that in pooled screening contexts, self.max_deletion_length may need to be set.
        '''

        new_features = {}

        names = {
            'target': self.target_name,
            'pegRNA': self.name,
        }

        features = {
            ('target', 'PBS'): self.features[names['target'], self.PBS_name],
            ('target', 'protospacer'): self.features[names['target'], self.protospacer_name],
            ('pegRNA', 'PBS'): self.features[names['pegRNA'], 'PBS'],
            ('pegRNA', 'RTT'): self.features[names['pegRNA'], 'RTT'],
            ('pegRNA', 'scaffold'): self.features[names['pegRNA'], 'scaffold'],
        }

        strands = {
            'target': features['target', 'PBS'].strand,
            'pegRNA': '-',
        }

        seqs = {
            ('pegRNA', name): features['pegRNA', name].sequence(self.reference_sequences)
            for name in ['RTT', 'scaffold']
        }

        # pegRNA sequences should always be provided as 5'-3' RNA
        # and therefore have the RTT feature on the - strand. 
        if features['pegRNA', 'RTT'].strand != '-':
            raise ValueError(str(features['pegRNA', 'RTT']))

        if features['pegRNA', 'scaffold'].strand != '+':
            raise ValueError(str(features['pegRNA', 'scaffold']))

        # feature sequence lookup uses the value of the strand attribute to reverse
        # complement if appropriate, so the RTT sequence will be RC'ed but not the scaffold.
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
            seqs['target', name] = self.target_sequence[starts['target', name]:ends['target', name]]
            if features['target', 'PBS'].strand == '-':
                seqs['target', name] = utilities.reverse_complement(seqs['target', name])

        # Align the intended flap sequence to the target downstream of the nick.

        SNVs, deletions, insertions, HA_RT, _ = self.extract_edits_from_alignment()

        if len(SNVs) > 0:
            if len(deletions) == 0 and len(insertions) == 0:
                self.edit_type = 'SNVs'
            else:
                self.edit_type = 'combination'
        else:
            if len(deletions) == 1 and len(insertions) == 0:
                self.edit_type = 'deletion'
            elif len(deletions) == 0 and len(insertions) == 1:
                self.edit_type = 'insertion'
            else:
                self.edit_type = 'combination'

        # Convert from flap/downstream coordinates to pegRNA/target coordinates.

        def convert_flap_to_pegRNA_coordinates(flap_p):
            return features['pegRNA', 'RTT'].end - flap_p

        def convert_downstream_of_nick_to_target_coordinates(downstream_p):
            if strands['target'] == '+':
                return self.cut_after + 1 + downstream_p
            else:
                return self.cut_after - downstream_p

        starts['pegRNA', 'HA_RT'], ends['pegRNA', 'HA_RT'] = sorted(map(convert_flap_to_pegRNA_coordinates, HA_RT['flap']))
        starts['target', 'HA_RT'], ends['target', 'HA_RT'] = sorted(map(convert_downstream_of_nick_to_target_coordinates, HA_RT['target_downstream']))

        if len(insertions) > 0:
            if len(insertions) > 1:
                logging.warning('multiple insertions')
                #raise NotImplementedError
            
            insertion = insertions[0]
            flap_coords = (insertion['start_in_flap'], insertion['end_in_flap'])
            downstream_coords = (insertion['starts_after_in_downstream'], insertion['ends_before_in_downstream'])
            starts['pegRNA', 'insertion'], ends['pegRNA', 'insertion'] = sorted(map(convert_flap_to_pegRNA_coordinates, flap_coords))
            starts_after, _ = sorted(map(convert_downstream_of_nick_to_target_coordinates, downstream_coords))
                
            insertion_name = f'insertion_{self.name}'
            insertion = gff.Feature.from_fields(seqname=names['pegRNA'],
                                                start=starts['pegRNA', 'insertion'],
                                                end=ends['pegRNA', 'insertion'],
                                                strand='-',
                                                ID=insertion_name,
                                               )
            insertion.attribute['color'] = default_feature_colors['insertion']
            new_features[names['pegRNA'], insertion_name] = insertion

            insertion_starts_after_name = f'insertion_starts_after_{self.name}'
            insertion_starts_after = gff.Feature.from_fields(seqname=names['target'],
                                                             start=starts_after,
                                                             end=starts_after,
                                                             strand='+',
                                                             ID=insertion_starts_after_name,
                                                            )
            new_features[names['target'], insertion_starts_after_name] = insertion_starts_after

            self.insertion = insertion

        if len(deletions) > 0:
            if len(deletions) > 1:
                logging.warning('multiple deletions')
                #raise NotImplementedError

            deletion = deletions[0]
            starts['target', 'deletion'], ends['target', 'deletion'] = sorted(map(convert_downstream_of_nick_to_target_coordinates, deletion))
            deletion_length = ends['target', 'deletion'] - starts['target', 'deletion'] + 1

            self.deletion = target_info.DegenerateDeletion([starts['target', 'deletion']], deletion_length)

            deletion_name = f'deletion_{self.name}'
            deletion_feature = gff.Feature.from_fields(seqname=names['target'],
                                                       start=starts['target', 'deletion'],
                                                       end=ends['target', 'deletion'],
                                                       strand=strands['target'],
                                                       ID=deletion_name,
                                                      )
            deletion_feature.attribute['color'] = default_feature_colors['deletion']
            new_features[names['target'], deletion_name] = deletion_feature

        if len(SNVs) > 0:
            self.SNVs = {
                self.target_name: {},
                self.name: {},
                'flap': {},
                'target_downstream': {},
            }

            self.SNV_name_to_alternative_coordinates = {}

            for SNV in SNVs.values():
                positions = {
                    'pegRNA': convert_flap_to_pegRNA_coordinates(SNV['flap']),
                    'target': convert_downstream_of_nick_to_target_coordinates(SNV['target_downstream']),
                }

                if strands['target'] == '+':
                    pegRNA_strand = '-'
                else:
                    pegRNA_strand = '+'

                target_base_plus = self.target_sequence[positions['target']]
                pegRNA_base_plus = self.reference_sequences[names['pegRNA']][positions['pegRNA']]

                if pegRNA_strand == '+':
                    pegRNA_base_effective = pegRNA_base_plus
                    target_base_effective = target_base_plus
                else:
                    pegRNA_base_effective = utilities.reverse_complement(pegRNA_base_plus)
                    target_base_effective = utilities.reverse_complement(target_base_plus)

                SNV_name = f'SNV_{positions["target"]}_{target_base_plus}-{pegRNA_base_effective}'

                self.SNVs['flap'][SNV_name] = {
                    'position': SNV['flap'],
                    'base': self.intended_flap_sequence[SNV['flap']],
                }
                self.SNVs['target_downstream'][SNV_name] = {
                    'position': SNV['target_downstream'],
                    'base': self.target_downstream_of_nick[SNV['target_downstream']],
                }

                self.SNVs[self.target_name][SNV_name] = {
                    'position': positions['target'],
                    'strand': '+',
                    'base': target_base_plus,
                    'alternative_base': pegRNA_base_effective,
                }

                self.SNVs[self.name][SNV_name] = {
                    'position': positions['pegRNA'],
                    'strand': pegRNA_strand,
                    'base': pegRNA_base_plus,
                    'alternative_base': target_base_effective,
                }

                for seq_name in names:
                    feature = gff.Feature.from_fields(seqname=names[seq_name],
                                                      start=positions[seq_name],
                                                      end=positions[seq_name],
                                                      strand=strands[seq_name],
                                                      ID=SNV_name,
                                                     )
                
                    new_features[names[seq_name], SNV_name] = feature

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

        first_difference_position = starts['pegRNA', 'RTT'] - offset
        name = f'after_first_difference_{names["pegRNA"]}'
        feature = gff.Feature.from_fields(seqname=names['pegRNA'],
                                          start=0,
                                          end=first_difference_position,
                                          strand='-',
                                          ID=name,
                                         )
        new_features[names['pegRNA'], name] = feature


        HA_PBS_name = f'HA_PBS_{names["pegRNA"]}'
        HA_RT_name = f'HA_RT_{names["pegRNA"]}'

        # Make target HA features.

        HA_PBS = copy.deepcopy(features['target', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[self.target_name, HA_PBS_name] = HA_PBS

        HA_RT = gff.Feature.from_fields(seqname=self.target_name,
                                        start=starts['target', 'HA_RT'],
                                        end=ends['target', 'HA_RT'],
                                        strand=HA_PBS.strand,
                                        ID=HA_RT_name,
                                       )
        HA_RT.attribute['color'] = default_feature_colors['RTT']
        new_features[self.target_name, HA_RT_name] = HA_RT

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

        self.features.update(new_features)

class TwinPrimePegRNA_pair:
    def __init__(self):
        pass

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
            logging.warning('pegRNAs not in PAM-in configuration')

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

    offset_to_positions = defaultdict(dict)
    for offset in range(len(target_seq)):
        offset_to_positions[offset][target_name] = offset

    for offset_in_RTT in range(len(pegRNA_RTTs[5])):
        position = pegRNA_RTTs[5].start + offset_in_RTT
        offset = target_PBSs[5].end + len(pegRNA_RTTs[5]) - offset_in_RTT
        offset_to_positions[offset][pegRNA_names_by_side[5]] = position

    for offset_in_RTT in range(len(pegRNA_RTTs[3])):
        position = pegRNA_RTTs[3].start + offset_in_RTT
        offset = target_PBSs[3].start - len(pegRNA_RTTs[3]) + offset_in_RTT
        offset_to_positions[offset][pegRNA_names_by_side[3]] = position

    # Align the RT'ed part of the 5' pegRNA to the target+RT'ed sequence
    # from the 3' side.

    for length in range(1, len(RTed[5]) + 1):
        suffix = RTed[5][-length:]
        starts = utilities.find_all_substring_starts(target_with_RTed[3], suffix)
        if len(starts) > 0:
            # If there are multiple matches, prioritize the one closest to the start.
            start = starts[0]
        else:
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
        starts = utilities.find_all_substring_starts(target_with_RTed[5], prefix)
        if len(starts) > 0:
            # If there are multiple matches, prioritize the one closest to the end.
            start = starts[-1]
        else:
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

    if overlap_seqs[5] == '' and overlap_seqs[3] == '':
        # prime del with no insertion won't have overlap between RTTs, but will have overlap between
        # each PBS and the other pegRNA's RTT.
        seq_5 = pegRNA_seqs[5]
        seq_3_rc = utilities.reverse_complement(pegRNA_seqs[3])
        PBS_length_3 = len(target_PBSs[3])
        longest_match = 0
        for l in range(PBS_length_3, len(seq_5)):
            if seq_3_rc[:l] == seq_5[-l:]:
                longest_match = l

        if longest_match != 0:
            for side in [5, 3]:
                overlap_feature = gff.Feature.from_fields(seqname=pegRNA_names_by_side[side],
                                                          feature='overlap',
                                                          start=len(pegRNA_seqs[side]) - 1 - longest_match,
                                                          end=len(pegRNA_seqs[side]) - 1,
                                                          strand='+',
                                                          attribute_string=gff.make_attribute_string({
                                                              'ID': 'overlap',
                                                              'color': default_feature_colors['overlap'],
                                                              'short_name': 'overlap',
                                                          }),
                                                         )
                new_features[pegRNA_names_by_side[side], 'overlap'] = overlap_feature

    intended_edit_seqs[3] = target_with_RTed[5][:start + length] + target_with_RTed[3][length:]

    deletion = None
    SNVs = None

    if (overlap_seqs[5] != overlap_seqs[3]) or (intended_edit_seqs[5] != intended_edit_seqs[3]):
        intended_edit_seq = None
        logging.warning(f'Unable to infer a consistent intended edit for {"+".join(pegRNA_names)}')
        for side in [5, 3]:
            new_features.pop((pegRNA_names_by_side[side], 'overlap'), None)
    else:
        intended_edit_seq = intended_edit_seqs[5]

        # Check if the intended edit is a deletion.

        unedited_seq = target_seq

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

        elif len(intended_edit_seq) == len(unedited_seq):

            SNVs = {
                target_name: {},
                pegRNA_names[0]: {},
                pegRNA_names[1]: {},
            }

            for offset, (pegRNAs_b, target_b) in enumerate(zip(intended_edit_seq, unedited_seq)):
                if pegRNAs_b != target_b:

                    positions = offset_to_positions[offset]

                    SNV_name = f'SNV_{positions[target_name]}_{target_b}-{pegRNAs_b}'

                    SNVs[target_name][SNV_name] = {
                        'position': positions[target_name],
                        'strand': '+',
                        'base': target_b,
                        'alternative_base': pegRNAs_b,
                    }

                    feature = gff.Feature.from_fields(seqname=target_name,
                                                      start=positions[target_name],
                                                      end=positions[target_name],
                                                      strand='+',
                                                      ID=SNV_name,
                                                     )
                
                    new_features[target_name, SNV_name] = feature

                    if pegRNA_names_by_side[5] in positions:
                        pegRNA_name = pegRNA_names_by_side[5]

                        # Note: convention on pegRNA base strandedness is a constant source
                        # of confusion.
                        pegRNA_base_effective = utilities.reverse_complement(pegRNAs_b)
                        target_base_effective = utilities.reverse_complement(target_b)

                        SNVs[pegRNA_name][SNV_name] = {
                            'position': positions[pegRNA_name],
                            'strand': '-',
                            'base': pegRNA_base_effective,
                            'alternative_base': target_base_effective,
                        }

                        feature = gff.Feature.from_fields(seqname=pegRNA_name,
                                                          start=positions[pegRNA_name],
                                                          end=positions[pegRNA_name],
                                                          strand='-',
                                                          ID=SNV_name,
                                                         )
                    
                        new_features[pegRNA_name, SNV_name] = feature

                    if pegRNA_names_by_side[3] in positions:
                        pegRNA_name = pegRNA_names_by_side[3]

                        pegRNA_base_effective = pegRNAs_b
                        target_base_effective = target_b

                        SNVs[pegRNA_name][SNV_name] = {
                            'position': positions[pegRNA_name],
                            'strand': '+',
                            'base': pegRNA_base_effective,
                            'alternative_base': target_base_effective,
                        }

                        feature = gff.Feature.from_fields(seqname=pegRNA_name,
                                                          start=positions[pegRNA_name],
                                                          end=positions[pegRNA_name],
                                                          strand='+',
                                                          ID=SNV_name,
                                                         )
                    
                        new_features[pegRNA_name, SNV_name] = feature

    results = {
        'deletion': deletion,
        'SNVs': SNVs,
        'new_features': new_features,
        'is_prime_del': is_prime_del,
        'intended_edit_seq': intended_edit_seq,
    }

    return results