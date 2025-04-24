import copy
import logging

from collections import defaultdict

import Bio.Align
import numpy as np
import pysam
import RNA

from hits import gff, interval, sam, sw, utilities

import knock_knock.utilities
import knock_knock.integrases
import knock_knock.effector
from knock_knock import target_info

memoized_property = utilities.memoized_property

def read_csv(csv_fn, process=True):
    df = knock_knock.utilities.read_and_sanitize_csv(csv_fn, index_col='name')

    if process:
        component_order = ['protospacer', 'scaffold', 'extension']

        for component in component_order:
            # Files only containing non-pegRNAs may omit scaffold
            # and extension columns.
            if component not in df.columns:
                df[component] = ''

            df[component] = [s.strip().upper().replace('U', 'T') for s in df[component]]

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
    'DR_WT': '#75c6a9',
}

def PBS_name(pegRNA_name):
    return f'{pegRNA_name}_PBS'

def protospacer_name(pegRNA_name):
    return f'{pegRNA_name}_protospacer'

def extract_pegRNA_name(PBS_name):
    return PBS_name.rsplit('_', 1)[0]

def get_RTT_aligner(match_score=2,
                    mismatch_score=-3,
                    open_gap_score=-12,
                    extend_gap_score=-0.1,
                   ):

    aligner = Bio.Align.PairwiseAligner(
        match_score=match_score,
        mismatch_score=mismatch_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )

    # Idea: 'global' mode with no penalty for target right gaps
    # requires the entire flap to be aligned without penalizing
    # the inclusion of extra genomic sequence at the end of the
    # query.  

    aligner.mode = 'global'
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

        self.effector = knock_knock.effector.effectors[self.components['effector']]

        self.reference_sequences = {
            self.name: self.components['full_sequence'],
            self.target_name: self.target_sequence,
        }

        self.features = {}

        self.PBS_name = PBS_name(self.name)
        self.protospacer_name = protospacer_name(self.name)

        self.substitutions = None
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

        target_protospacer_feature = self.effector.identify_protospacer_in_target(self.target_sequence, protospacer)
        target_protospacer_feature.attribute['ID'] = self.protospacer_name
        target_protospacer_feature.seqname = self.target_name
        strand = target_protospacer_feature.strand
        
        cut_afters = self.effector.cut_afters(target_protospacer_feature)
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
        for seed_length in [7, 6, 5, 4]:
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

            if len(valid_alignments) > 0:
                break

        if len(valid_alignments) == 0:
            seed_sequence = self.target_bytes[seed_start:seed_start + seed_length]
            starts = [al.reference_start for al in valid_alignments]
            error_message = [
                f'{self.name}: {len(valid_alignments)} valid PBS alignment(s) for {seed_sequence}:'
            ]
            for al in alignments:
                error_message.append(
                    f'\tlength: {al.query_alignment_length}, start in pegRNA: {al.reference_start} {sam.get_strand(al)}, start in target: {al.query_alignment_start}'
                )
            error_message = '\n'.join(error_message)
            raise ValueError(error_message)
            
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

    @property
    def strand(self):
        return self.features[self.target_name, self.PBS_name].strand

    @memoized_property
    def intended_flap_sequence(self):
        return utilities.reverse_complement(self.components['RTT'])

    @memoized_property
    def cut_after(self):
        protospacer = self.features[self.target_name, self.protospacer_name]
        try:
            cut_after = self.effector.cut_afters(protospacer)[protospacer.strand]
        except KeyError:
            # PE nuclease support
            cut_after = self.effector.cut_afters(protospacer)['both']

        return cut_after

    @memoized_property
    def target_downstream_of_nick(self):
        protospacer = self.features[self.target_name, self.protospacer_name]

        if protospacer.strand == '+':
            target_downstream_of_nick = self.target_sequence[self.cut_after + 1:]
        else:
            target_downstream_of_nick = utilities.reverse_complement(self.target_sequence[:self.cut_after + 1])

        if self.max_deletion_length is not None:
            target_downstream_of_nick = target_downstream_of_nick[:self.max_deletion_length + len(self.intended_flap_sequence)]

        return target_downstream_of_nick

    @memoized_property
    def target_upstream_of_nick(self):
        protospacer = self.features[self.target_name, self.protospacer_name]

        if protospacer.strand == '+':
            target_upstream_of_nick = self.target_sequence[:self.cut_after + 1]
        else:
            target_upstream_of_nick = utilities.reverse_complement(self.target_sequence[self.cut_after + 1:])

        target_upstream_of_nick = target_upstream_of_nick[-20:]

        return target_upstream_of_nick

    def align_RTT_to_target(self):
        ''' Align the intended flap sequence to genomic sequence downstream
        of the nick using Biopython's dynamic programming.
        ''' 

        aligner = get_RTT_aligner()

        alignments = aligner.align(self.intended_flap_sequence, self.target_downstream_of_nick)

        min_aligned_nts = min(len(self.intended_flap_sequence), 5)

        best_alignment = alignments[0]

        gaps, identities, mismatches = best_alignment.counts()

        if identities + mismatches < min_aligned_nts:
            sequences = [self.intended_flap_sequence, self.target_downstream_of_nick]
            coordinates = np.array([[0, len(self.intended_flap_sequence), len(self.intended_flap_sequence)],
                                    [0, 0, len(self.target_downstream_of_nick)]
                                   ],
                                  )
            unaligned = Bio.Align.Alignment(sequences, coordinates)
            alignments = [unaligned]

        return alignments

    def print_best_alignments(self):
        alignments = self.align_RTT_to_target()
        for alignment in alignments:
            print(trim_excess_target_from_alignment(alignment))

    @memoized_property
    def programmed_substitution_target_ps(self):

        if self.substitutions is None:
            target_substitutions = {}
        else:
            target_substitutions = self.substitutions[self.target_name]

        ps = {substitution_details['position'] for _, substitution_details in target_substitutions.items()}

        return ps

    @memoized_property
    def edit_properties(self):
        ''' Align the intended flap sequence to genomic sequence downstream
        of the nick using Biopython's dynamic programming.
        Return representations of substitutions, deletions, insertions, and
        of the exactly-matching terminal HA_RT in coordinates relative to the 
        start of the flap and the nick, respectively. 
        ''' 

        alignments = self.align_RTT_to_target()

        best_alignment = trim_excess_target_from_alignment(alignments[0])

        substitutions = {}
        deletions = []
        insertions = []
        HA_RT = None

        flap_subsequences, target_subsequences = best_alignment.aligned
        
        # Do completely non-homologous flaps always produce len 0?
        if len(flap_subsequences) > 0 and len(target_subsequences) > 0:

            # flap_subsequences and target_subsequences are arrays of length-2 arrays
            # that indicate the start and (exclusive) ends of subsequences of the flap
            # and target that are aligned to each other.

            # Track mismatch positions within each subsequence to allow definition
            # of the exactly-matching HA_RT section of the final subsequence later.
            mismatches_in_subsequences = []

            # Compare the sequences of each aligned flap and target subsequences to
            # identify programmed substitutions.
            for flap_subsequence, target_subsequence in zip(flap_subsequences, target_subsequences):
                flap_ps = range(*flap_subsequence)
                target_ps = range(*target_subsequence)

                mismatches = [-1]
                
                for i, (flap_p, target_p) in enumerate(zip(flap_ps, target_ps)):
                    flap_b = self.intended_flap_sequence[flap_p]
                    target_b = self.target_downstream_of_nick[target_p]

                    if flap_b != target_b:
                        mismatches.append(i)
                        
                        substitution_name = f'substitution_{target_p}_{target_b}_{flap_p}_{flap_b}'

                        substitutions[substitution_name] = {
                            'flap': flap_p,
                            'target_downstream': target_p,
                            'description': f'+{target_p + 1}{target_b}→{flap_b}',
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

            for insertion in insertions:
                sequence = self.intended_flap_sequence[insertion['start_in_flap']:insertion['end_in_flap'] + 1]
                position = insertion['starts_after_in_downstream'] + 1
                insertion['description'] = f'+{position}ins{sequence}'

            # Deletions are gaps between consecutive target subsequences. If the
            # first subsequence doesn't start at 0, sequence immediately after the
            # nick is deleted.

            if target_subsequences[0][0] != 0:
                deletions.append((0, target_subsequences[0][0] - 1))
                    
            for (_, left_target_end), (right_target_start, _) in zip(target_subsequences, target_subsequences[1:]):
                if left_target_end != right_target_start:
                    deletions.append((left_target_end, right_target_start - 1))

        properties = {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'HA_RT': HA_RT,
            'HA_RT_length': (HA_RT['flap'][1] - HA_RT['flap'][0] + 1) if HA_RT is not None else 0,
            'flap_subsequences': flap_subsequences,
            'target_subsequences': target_subsequences,
        }
                
        return properties

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

        if len(self.edit_properties['substitutions']) > 0:
            if len(self.edit_properties['deletions']) == 0 and len(self.edit_properties['insertions']) == 0:
                self.edit_type = 'substitution(s)'
            else:
                self.edit_type = 'combination'
        else:
            if len(self.edit_properties['deletions']) == 1 and len(self.edit_properties['insertions']) == 0:
                self.edit_type = 'deletion'
            elif len(self.edit_properties['deletions']) == 0 and len(self.edit_properties['insertions']) == 1:
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

        if self.edit_properties['HA_RT'] is not None:
            starts['pegRNA', 'HA_RT'], ends['pegRNA', 'HA_RT'] = sorted(map(convert_flap_to_pegRNA_coordinates, self.edit_properties['HA_RT']['flap']))
            starts['target', 'HA_RT'], ends['target', 'HA_RT'] = sorted(map(convert_downstream_of_nick_to_target_coordinates, self.edit_properties['HA_RT']['target_downstream']))

        if len(self.edit_properties['insertions']) > 0:
            if len(self.edit_properties['insertions']) > 1:
                logging.warning('multiple insertions')
            
            insertion = self.edit_properties['insertions'][0]
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

        if len(self.edit_properties['deletions']) > 0:
            if len(self.edit_properties['deletions']) > 1:
                logging.warning(f'Inferred edit for {self.name} has multiple deletions')

            deletion = self.edit_properties['deletions'][0]
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

        if len(self.edit_properties['substitutions']) > 0:
            self.substitutions = {
                self.target_name: {},
                self.name: {},
                'flap': {},
                'target_downstream': {},
            }

            for substitution in self.edit_properties['substitutions'].values():
                positions = {
                    'pegRNA': convert_flap_to_pegRNA_coordinates(substitution['flap']),
                    'target': convert_downstream_of_nick_to_target_coordinates(substitution['target_downstream']),
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

                substitution_name = f'substitution_{positions["target"]}_{target_base_plus}-{pegRNA_base_effective}'

                self.substitutions['flap'][substitution_name] = {
                    'position': substitution['flap'],
                    'base': self.intended_flap_sequence[substitution['flap']],
                    'description': substitution['description'],
                }

                self.substitutions['target_downstream'][substitution_name] = {
                    'position': substitution['target_downstream'],
                    'base': self.target_downstream_of_nick[substitution['target_downstream']],
                    'description': substitution['description'],
                }

                self.substitutions[self.target_name][substitution_name] = {
                    'position': positions['target'],
                    'strand': '+',
                    'base': target_base_plus,
                    'alternative_base': pegRNA_base_effective,
                    'description': substitution['description'],
                }

                self.substitutions[self.name][substitution_name] = {
                    'position': positions['pegRNA'],
                    'strand': pegRNA_strand,
                    'base': pegRNA_base_plus,
                    'alternative_base': target_base_effective,
                    'description': substitution['description'],
                }

                for seq_name in names:
                    feature = gff.Feature.from_fields(seqname=names[seq_name],
                                                      start=positions[seq_name],
                                                      end=positions[seq_name],
                                                      strand=strands[seq_name],
                                                      ID=substitution_name,
                                                     )
                
                    new_features[names[seq_name], substitution_name] = feature

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

        if self.edit_properties['HA_RT'] is not None:
            target_HA_RT = gff.Feature.from_fields(seqname=self.target_name,
                                                   start=starts['target', 'HA_RT'],
                                                   end=ends['target', 'HA_RT'],
                                                   strand=HA_PBS.strand,
                                                   ID=HA_RT_name,
                                                  )
            target_HA_RT.attribute['color'] = default_feature_colors['RTT']
            new_features[self.target_name, HA_RT_name] = target_HA_RT

        # Make pegRNA HA features.

        HA_PBS = copy.deepcopy(features['pegRNA', 'PBS'])
        HA_PBS.attribute['ID'] = HA_PBS_name
        new_features[names['pegRNA'], HA_PBS_name] = HA_PBS

        if self.edit_properties['HA_RT'] is not None:
            pegRNA_HA_RT = gff.Feature.from_fields(seqname=names['pegRNA'],
                                                   start=starts['pegRNA', 'HA_RT'],
                                                   end=ends['pegRNA', 'HA_RT'],
                                                   strand='-',
                                                   ID=HA_RT_name,
                                                  )
            pegRNA_HA_RT.attribute['color'] = default_feature_colors['RTT']
            new_features[names['pegRNA'], HA_RT_name] = pegRNA_HA_RT

        self.features.update(new_features)

    @memoized_property
    def edit_description(self):
        if self.edit_properties['HA_RT'] is None:
            edit_description = 'no homology'
        else:
            strings = []

            for name, details in self.edit_properties['substitutions'].items():
                strings.append((details['target_downstream'], details['description']))

            for insertion in self.edit_properties['insertions']:
                strings.append((insertion['start_in_flap'], insertion['description']))

            for deletion in self.edit_properties['deletions']: 
                sequence = self.target_downstream_of_nick[deletion[0]:deletion[1] + 1]
                position = deletion[0] + 1
                strings.append((deletion[0], f'+{position}del{sequence}'))

            strings = [s for p, s in sorted(strings)]

            edit_description = ','.join(strings)

        return edit_description

    @memoized_property
    def RTT_structure(self):
        fc = RNA.fold_compound(self.components['RTT'])
        (propensity, ensemble_energy) = fc.pf()
        basepair_probs = fc.bpp()
        array = np.array(basepair_probs)[1:, 1:]
        sym_array = array + array.T
        flipped_total_bpps = sym_array.sum(axis=1)[::-1]
        flipped_propensity = propensity[::-1].translate(str.maketrans('(){}', ')()('))

        return flipped_total_bpps, flipped_propensity

    @memoized_property
    def extension_structure(self):
        fc = RNA.fold_compound(self.components['extension'])
        (propensity, ensemble_energy) = fc.pf()
        basepair_probs = fc.bpp()
        array = np.array(basepair_probs)[1:, 1:]
        sym_array = array + array.T
        flipped_total_bpps = sym_array.sum(axis=1)[::-1]
        flipped_propensity = propensity[::-1].translate(str.maketrans('(){}', ')()('))

        return flipped_total_bpps, flipped_propensity

    @memoized_property
    def substitution_string_to_edit_description(self):
        substitutions = self.substitutions

        substitution_name_to_target_order = {substitution_name: i for i, substitution_name in enumerate(sorted(substitutions[self.target_name]))}

        substitution_names_in_flap_order = sorted(substitutions['flap'], key=lambda substitution_name: substitutions['flap'][substitution_name]['position'])
        substitution_name_to_flap_order = {substitution_name: i for i, substitution_name in enumerate(substitution_names_in_flap_order)}

        substitution_string_to_edit_description = {}

        for substitution_subset in utilities.powerset(substitutions['flap']):
            if len(substitution_subset) == 0:
                continue

            chars = ['_' for _ in substitutions['flap']]
            for name in substitution_subset:
                chars[substitution_name_to_target_order[name]] = substitutions[self.target_name][name]['alternative_base']
            
            substitution_string = (''.join(chars))
            
            subset_in_flap_order = sorted(substitution_subset, key=substitution_name_to_flap_order.get)
            
            description = ','.join(substitutions[self.target_name][name]['description'] for name in subset_in_flap_order)
            
            substitution_string_to_edit_description[substitution_string] = description

        return substitution_string_to_edit_description

def get_pegRNAs_by_strand(pegRNAs):
    pegRNAs_by_strand = {}

    for pegRNA_ in pegRNAs:
        if pegRNA_.strand in pegRNAs_by_strand:
            raise ValueError('pegRNAs target same strand')
        else:
            pegRNAs_by_strand[pegRNA_.strand] = pegRNA_
    
    return pegRNAs_by_strand

def get_PBS_features_by_strand(pegRNAs):
    by_strand = {}

    for strand, pegRNA_ in get_pegRNAs_by_strand(pegRNAs).items():
        by_strand[strand] = pegRNA_.features[pegRNA_.target_name, pegRNA_.PBS_name]

    return by_strand

def get_pegRNAs_by_side_of_target(pegRNAs):
    ''' assumes a PAM-in configuration '''

    strand_to_side = {
        '+': 5,
        '-': 3,
    }

    by_side = {}

    for strand, pegRNA_ in get_pegRNAs_by_strand(pegRNAs).items():
        side = strand_to_side[strand]
        by_side[side] = pegRNA_

    return by_side

def get_PBS_names_by_side_of_target(pegRNAs):
    return {side: pegRNA_.PBS_name for side, pegRNA_ in get_pegRNAs_by_side_of_target(pegRNAs).items()}

def get_pegRNA_names_by_side_of_target(pegRNAs):
    return {side: pegRNA_.name for side, pegRNA_ in get_pegRNAs_by_side_of_target(pegRNAs).items()}

class pegRNA_pair:
    ''' features:
            PBS
            RTT
            overlap
    '''
    def __init__(self, pegRNAs):
        if len(pegRNAs) != 2:
            raise ValueError
        else:
            self.pegRNAs = pegRNAs

        target_names = set(pegRNA_.target_name for pegRNA_ in pegRNAs)
        if len(target_names) != 1:
            raise ValueError('pegRNAs in pair must have same target_name')
        else:
            self.target_name = list(target_names)[0]

        target_sequences = set(pegRNA_.target_sequence for pegRNA_ in pegRNAs)
        if len(target_sequences) != 1:
            raise ValueError('pegRNAs in pair must have same target_sequence')
        else:
            self.target_sequence = list(target_sequences)[0]

        self.features = {}
        for pegRNA_ in self.pegRNAs:
            self.features.update(pegRNA_.features)

        self.infer_overlap_features()

    @memoized_property
    def best_RT_extended_sequences_alignment(self):
        ''' Align RT'ed flaps to each other. Include target sequence that precedes flaps
            to allow for flaps to align to opposite R-loop (or past it).
        '''

        # TODO: these parameters are copied from flap-RTT alignment,
        # but this use case is very different.
        # TODO: add tests for this from EXP24002732_intron21_DF OLIP7981_hg38_Y11P11+Z11P11_

        flap_aligner = Bio.Align.PairwiseAligner(
            match_score=2,
            mismatch_score=-3,
            open_gap_score=-200,
            extend_gap_score=-0.1,
            mode='global',
            query_left_open_gap_score=0,
            query_left_extend_gap_score=0,
            target_right_open_gap_score=0,
            target_right_extend_gap_score=0,
        )

        alignments = flap_aligner.align(self.RT_extended_target_sequence['+'],
                                        self.RT_extended_target_sequence['-'],
                                       )
        best_alignment = next(alignments)

        return best_alignment

    @memoized_property
    def pegRNAs_by_strand(self):
        return get_pegRNAs_by_strand(self.pegRNAs)

    @memoized_property
    def target_PBSs(self):
        return get_PBS_features_by_strand(self.pegRNAs)

    @memoized_property
    def target_sequence_up_to_nick(self):
        ''' Target sequence from the start up to the + nick
        and from the - nick to the end.
        '''
        target_sequence_up_to_nick = {
            '+': self.target_sequence[:self.target_PBSs['+'].end + 1],
            '-': self.target_sequence[self.target_PBSs['-'].start:],
        }

        return target_sequence_up_to_nick

    @memoized_property
    def RTed_sequence(self):
        return {
            '+': self.pegRNAs_by_strand['+'].intended_flap_sequence,
            '-': utilities.reverse_complement(self.pegRNAs_by_strand['-'].intended_flap_sequence),
        }

    @memoized_property
    def RT_extended_target_sequence(self):
        return {
            '+': self.target_sequence_up_to_nick['+'] + self.RTed_sequence['+'],
            '-': self.RTed_sequence['-'] + self.target_sequence_up_to_nick['-'],
        }

    @memoized_property
    def complete_integrase_site_ends_in_RT_extended_target_sequence(self):
        features = knock_knock.integrases.identify_split_recognition_sequences(self.RT_extended_target_sequence)

        all_complete_sites = {
            '+': [],
            '-': [],
        }

        for (strand, feature_name), feature in features.items():
            if feature.attribute['component'] == 'complete_site':
                all_complete_sites[strand].append(feature)
                
        ends = {}

        # First get coordinates relative to last nt before the nick, then translate
        # to the start of the PBS.

        for strand, complete_sites in all_complete_sites.items():
            if len(complete_sites) == 0:
                ends[strand] = None

            elif len(complete_sites) == 1:
                complete_site = complete_sites[0]
                label = f'{complete_site.attribute["recombinase"]}_{complete_site.attribute["site"]}_{complete_site.attribute["CD"]}'
                if strand == '+':
                    end = complete_site.end - (len(self.target_sequence_up_to_nick['+']) - 1) + (len(self.pegRNAs_by_strand['+'].components['PBS']) - 1)
                elif strand == '-':
                    end = len(self.RT_extended_target_sequence['-']) - len(self.target_sequence_up_to_nick['-']) - complete_site.start + (len(self.pegRNAs_by_strand['-'].components['PBS']) - 1)

                ends[strand] = (end, label)

            else:
                raise NotImplementedError

        return ends

    @memoized_property
    def RT_and_overlap_extended_target_sequence(self):
        RTed_seq = self.RT_extended_target_sequence
        overlap = self.overlap_interval

        return RTed_seq['+'][:overlap['+'].end + 1] + RTed_seq['-'][overlap['-'].end + 1:]

    @memoized_property
    def complete_integrase_site_ends_in_RT_and_overlap_extended_target_sequence(self):
        full_seq = self.RT_and_overlap_extended_target_sequence
        features = knock_knock.integrases.identify_split_recognition_sequences({'full_seq': full_seq})
        complete_sites = [feature for feature in features.values() if feature.attribute['component'] == 'complete_site']

        if len(complete_sites) == 0:
            ends = {
                '+': None,
                '-': None,
            }
        elif len(complete_sites) == 1:
            complete_site = complete_sites[0]
            label = f'{complete_site.attribute["recombinase"]}_{complete_site.attribute["site"]}_{complete_site.attribute["CD"]}'

            # First get coordinates relative to last nt before the nick, then translate
            # to the start of the PBS.
            ends = {
                '+': (complete_site.end - (len(self.target_sequence_up_to_nick['+']) - 1) + (len(self.pegRNAs_by_strand['+'].components['PBS']) - 1), label),
                '-': ((len(full_seq) - len(self.target_sequence_up_to_nick['-'])) - complete_site.start + (len(self.pegRNAs_by_strand['-'].components['PBS']) - 1), label),
            }
        else:
            raise NotImplementedError

        return ends

    @memoized_property
    def RTed_interval_in_extended_target(self):
        return {
            '+': interval.Interval(len(self.target_sequence_up_to_nick['+']), len(self.RT_extended_target_sequence['+']) - 1),
            '-': interval.Interval(0, len(self.RTed_sequence['-']) - 1),
        }

    @memoized_property
    def RT_extended_target_coords_to_pegRNA_coords(self):
        mapping = {
            '+': {},
            '-': {},
        }

        components = self.pegRNAs_by_strand['+'].components

        extension_start = len(components['protospacer'] + components['scaffold'])
        PBS_and_RTT_length = len(components['PBS'] + components['RTT'])

        for offset in range(PBS_and_RTT_length):
            target_with_RTed_p = len(self.RT_extended_target_sequence['+']) - 1 - offset
            pegRNA_p = extension_start + offset

            mapping['+'][target_with_RTed_p] = pegRNA_p

        components = self.pegRNAs_by_strand['-'].components

        extension_start = len(components['protospacer'] + components['scaffold'])
        PBS_and_RTT_length = len(components['PBS'] + components['RTT'])

        for offset in range(PBS_and_RTT_length):
            target_with_RTed_p = offset
            pegRNA_p = extension_start + offset

            mapping['-'][target_with_RTed_p] = pegRNA_p

        return mapping

    @memoized_property
    def edit_coords_to_RT_extended_target_coords(self):
        pass

    @memoized_property
    def WT_between_nick_coords_to_target_coords(self):
        target_ps = range(self.target_PBSs['+'].end + 1, self.target_PBSs['-'].start)
        return dict(enumerate(target_ps))

    @memoized_property
    def aligned_RT_extended_positions(self):
        mapping = {
            '+ to -': {},
            '- to +': {},
        }

        for plus_subsequence, minus_subsequence in zip(*self.best_RT_extended_sequences_alignment.aligned):
            for plus_p, minus_p in zip(range(*plus_subsequence), range(*minus_subsequence)):
                mapping['+ to -'][plus_p] = minus_p
                mapping['- to +'][minus_p] = plus_p

        return mapping

    @memoized_property
    def WT_sequence_between_nicks(self):
        return ''.join(self.target_sequence[target_p] for between_nicks_p, target_p in sorted(self.WT_between_nick_coords_to_target_coords.items()))

    @memoized_property
    def RT_extended_target_coords_to_edit_coords(self):
        # for +, start after PBS and continue to the end of RTT
        # or to the position that aligns to the last base before the
        # PBS of the -, whichever come first
        
        mapping = {
            '+': {},
            '-': {},
        }

        if self.overlap_interval_in_RTed_for_both['+'].total_length > 0:
            before_overlap_start = self.RTed_interval_in_extended_target['+'].start
            before_overlap_end = self.overlap_interval_in_RTed_for_both['+'].start
            
            for edit_p, RT_extended_p in enumerate(range(before_overlap_start, before_overlap_end)):
                mapping['+'][RT_extended_p] = edit_p

            overlap_start = self.overlap_interval_in_RTed_for_both['+'].start
            overlap_end = self.overlap_interval_in_RTed_for_both['+'].end + 1

            overlap_start_in_edit = max(mapping['+'].values(), default=-1) + 1

            for edit_p, RT_extended_p in enumerate(range(overlap_start, overlap_end), overlap_start_in_edit):
                mapping['+'][RT_extended_p] = edit_p

            overlap_start = self.overlap_interval_in_RTed_for_both['-'].start
            overlap_end = self.overlap_interval_in_RTed_for_both['-'].end + 1

            for edit_p, RT_extended_p in enumerate(range(overlap_start, overlap_end), overlap_start_in_edit):
                mapping['-'][RT_extended_p] = edit_p

            after_overlap_start = self.overlap_interval_in_RTed_for_both['-'].end + 1
            after_overlap_end = self.RTed_interval_in_extended_target['-'].end + 1

            after_overlap_start_in_edit = max(mapping['-'].values()) + 1

            for edit_p, RT_extended_p in enumerate(range(after_overlap_start, after_overlap_end), after_overlap_start_in_edit):
                mapping['-'][RT_extended_p] = edit_p

        return mapping

    @memoized_property
    def edit_coords_to_RT_extended_target_coords(self):
        return {
            strand: utilities.reverse_dictionary(self.RT_extended_target_coords_to_edit_coords[strand])
            for strand in ['+', '-']
        }

    @memoized_property
    def edit_coords_to_pegRNA_coords(self):
        edit_coords_to_pegRNA_coords = {}

        for strand in ['+', '-']:
            mapping = {}
            
            for edit_p, RT_extended_p in self.edit_coords_to_RT_extended_target_coords[strand].items():
                pegRNA_p = self.RT_extended_target_coords_to_pegRNA_coords[strand][RT_extended_p]
                mapping[edit_p] = pegRNA_p
                
            edit_coords_to_pegRNA_coords[strand] = mapping

        return edit_coords_to_pegRNA_coords

    @memoized_property
    def pegRNA_coords_to_edit_coords(self):
        pegRNA_coords_to_edit_coords = {}

        for strand, edit_to_pegRNA in self.edit_coords_to_pegRNA_coords.items():
            pegRNA_name = self.pegRNAs_by_strand[strand].name
            
            pegRNA_coords_to_edit_coords[pegRNA_name] = {}
            for edit_p, pegRNA_p in edit_to_pegRNA.items():
                pegRNA_coords_to_edit_coords[pegRNA_name][pegRNA_p] = edit_p

        return pegRNA_coords_to_edit_coords

    @memoized_property
    def intended_edit_between_nicks(self):
        edit_bs = [[] for _ in range(max(self.edit_coords_to_RT_extended_target_coords['-'], default=-1) + 1)]

        for strand, mapping in self.edit_coords_to_RT_extended_target_coords.items():
            for edit_p, RT_extended_p in mapping.items():
                edit_bs[edit_p].append(self.RT_extended_target_sequence[strand][RT_extended_p])
                
        for i in range(len(edit_bs)):
            if len(set(edit_bs[i])) != 1:
                raise ValueError
            else:
                edit_bs[i] = edit_bs[i][0]
                
        intended_edit_between_nicks = ''.join(edit_bs)
        
        return intended_edit_between_nicks

    @memoized_property
    def overlap_interval(self):
        aligned_ps = self.aligned_RT_extended_positions

        # TODO: this implicitly assumes a single long alignment.

        return {
            '+': interval.Interval(min(aligned_ps['+ to -']), max(aligned_ps['+ to -'])),
            '-': interval.Interval(min(aligned_ps['- to +']), max(aligned_ps['- to +'])),
        }

    @memoized_property
    def overlap_interval_in_RTed(self):
        return {
            strand: self.overlap_interval[strand] & self.RTed_interval_in_extended_target[strand]
            for strand in ['+', '-']
        }

    @memoized_property
    def overlap_interval_in_RTed_for_both(self):
        ps = {
            '+': set(),
            '-': set(),
        }
        
        for plus_p in self.overlap_interval_in_RTed['+']:
            minus_p = self.aligned_RT_extended_positions['+ to -'].get(plus_p, -1)
            if minus_p in self.overlap_interval_in_RTed['-']:
                ps['+'].add(plus_p)
                
        for minus_p in self.overlap_interval_in_RTed['-']:
            plus_p = self.aligned_RT_extended_positions['- to +'].get(minus_p, -1)
            if plus_p in self.overlap_interval_in_RTed['+']:
                ps['-'].add(minus_p)
                
        return {
            strand: interval.Interval(min(ps[strand]), max(ps[strand])) if len(ps[strand]) > 0 else interval.Interval.empty() 
            for strand in ['+', '-']
        }

    @memoized_property
    def overlap_interval_outside_RTed(self):
        return {
            strand: self.overlap_interval[strand] - self.RTed_interval_in_extended_target[strand]
            for strand in ['+', '-']
        }

    @memoized_property
    def is_prime_del(self):
        # prime del if any RT'ed sequence from one flap is aligned to non-RT'ed sequence from the other.
        return any(self.overlap_interval_outside_RTed.values())

    def RT_extended_interval_to_pegRNA_interval(self, RT_extended_interval, strand):
        mapping = self.RT_extended_target_coords_to_pegRNA_coords[strand]
        start, end = sorted([mapping[RT_extended_interval.start], mapping[RT_extended_interval.end]])
        return interval.Interval(start, end)

    def infer_overlap_features(self):
        for strand in ['+', '-']:
            pegRNA_ = self.pegRNAs_by_strand[strand]

            if not self.overlap_interval_in_RTed_for_both[strand].is_empty:
                pegRNA_interval = self.RT_extended_interval_to_pegRNA_interval(self.overlap_interval_in_RTed_for_both[strand], strand)

                opposite_strand = '-' if strand == '+' else '+'

                overlap_feature = gff.Feature.from_fields(seqname=pegRNA_.name,
                                                          feature='overlap',
                                                          start=pegRNA_interval.start,
                                                          end=pegRNA_interval.end,
                                                          strand=opposite_strand,
                                                          attribute_string=gff.make_attribute_string({
                                                              'ID': 'overlap',
                                                              'color': default_feature_colors['overlap'],
                                                              'short_name': 'overlap',
                                                          }),
                                                         )

                self.features[pegRNA_.name, 'overlap'] = overlap_feature

    @memoized_property
    def best_alignment_of_edit_to_WT_between_nicks(self):
        if self.intended_edit_between_nicks == '':
            best_alignment = None
        else:
            edit_aligner = Bio.Align.PairwiseAligner(
                match_score=2,
                mismatch_score=-3,
                open_gap_score=-12,
                extend_gap_score=-0.1,
                mode='global',
            )
            
            alignments = edit_aligner.align(self.intended_edit_between_nicks, self.WT_sequence_between_nicks)

            best_alignment = next(alignments)

            if best_alignment.score <= 0:
                best_alignment = None

        return best_alignment

    @memoized_property
    def programmed_substitution_ps(self):
        self.extract_edits()

        ps = {}

        for pegRNA in self.pegRNAs:
            ps[pegRNA.name] = set()

            for substitution_name, substitution_details in self.substitutions[pegRNA.name].items():
                ps[pegRNA.name].add(substitution_details['position'])

        return ps

    def extract_edits(self):

        self.substitutions = {
            self.target_name: {},
            self.pegRNAs[0].name: {},
            self.pegRNAs[1].name: {},
        }

        self.insertions = []

        if self.best_alignment_of_edit_to_WT_between_nicks is not None:
            edit_subsequences, WT_subsequences = self.best_alignment_of_edit_to_WT_between_nicks.aligned

            # *_subsequences are arrays of length-2 arrays
            # that indicate the start and (exclusive) ends of subsequences of edit
            # and target that are aligned to each other.

            # Identify substitutions.
            for edit_subsequence, WT_subsequence in zip(edit_subsequences, WT_subsequences):
                edit_ps = range(*edit_subsequence)
                WT_ps = range(*WT_subsequence)

                for edit_p, WT_between_nicks_p in zip(edit_ps, WT_ps):
                    edit_b = self.intended_edit_between_nicks[edit_p]
                    target_b = self.WT_sequence_between_nicks[WT_between_nicks_p]

                    target_p = self.WT_between_nick_coords_to_target_coords[WT_between_nicks_p]

                    if edit_b != target_b:
                        substitution_name = f'substitution_{target_p}_{target_b}-{edit_b}'

                        self.substitutions[self.target_name][substitution_name] = {
                            'position': target_p,
                            'strand': '+',
                            'base': target_b,
                            'alternative_base': edit_b,
                        }

                        for pegRNA_strand, mapping in self.edit_coords_to_pegRNA_coords.items():
                            if edit_p in mapping:
                                pegRNA_p = mapping[edit_p]
                                pegRNA_b = self.pegRNAs_by_strand[pegRNA_strand].components['full_sequence'][pegRNA_p]

                                target_b_effective = target_b if pegRNA_strand == '-' else utilities.reverse_complement(target_b)

                                opposite_strand = '-' if pegRNA_strand == '+' else '+'

                                self.substitutions[self.pegRNAs_by_strand[pegRNA_strand].name][substitution_name] = {
                                    'position': pegRNA_p,
                                    'strand': opposite_strand,
                                    'base': pegRNA_b,
                                    'alternative_base': target_b_effective,
                                }

            # Insertions are gaps between consecutive edit subsequences. If the
            # first subsequence doesn't start at 0, the edit begins with an insertion.

            if edit_subsequences[0][0] != 0:
                insertion = {
                    'start_in_edit': 0,
                    'end_in_edit': edit_subsequences[0][0] - 1,
                    'starts_after_in_downstream': -1,
                    'ends_before_in_downstream': 0,
                }
                self.insertions.append(insertion)
                
            for (_, left_edit_end), (right_edit_start, _), (_, left_WT_end), (right_WT_start, _) in zip(edit_subsequences, edit_subsequences[1:], WT_subsequences, WT_subsequences[1:]):
                if left_edit_end != right_edit_start:
                    insertion = {
                        'start_in_edit': left_edit_end,
                        'end_in_edit': right_edit_start - 1,
                        'starts_after_in_WT': left_WT_end - 1,
                        'ends_before_in_WT': right_WT_start,
                    }
                    self.insertions.append(insertion)

            ## Deletions are gaps between consecutive target subsequences. If the
            ## first subsequence doesn't start at 0, sequence immediately after the
            ## nick is deleted.

            #if target_subsequences[0][0] != 0:
            #    deletions.append((0, target_subsequences[0][0] - 1))
            #        
            #for (_, left_target_end), (right_target_start, _) in zip(target_subsequences, target_subsequences[1:]):
            #    if left_target_end != right_target_start:
            #        deletions.append((left_target_end, right_target_start - 1))

class PE3b_spacer:
    def __init__(self, name, components, target_sequence):
        self.name = name
        self.components = components
        self.target_sequence = target_sequence

        self.effector = knock_knock.effector.effectors[components['effector']]

        if self.effector.PAM_side == 3:
            self.protospacer_and_PAM = components['protospacer'] + self.effector.PAM_pattern
            self.PAM_positions = len(components['protospacer']) + np.arange(len(self.effector.PAM_pattern))
            self.cut_afters = set(len(components['protospacer']) + offset for offset in self.effector.cut_after_offset if offset is not None)
        else:
            self.protospacer_and_PAM = self.effector.PAM_pattern + components['protospacer'] 
            self.PAM_positions = np.arange(len(self.effector.PAM_pattern))
            self.cut_afters = set(offset for offset in self.effector.cut_after_offset if offset is not None)

    @memoized_property
    def alignment_to_target(self):
        aligner = Bio.Align.PairwiseAligner(match_score=2,
                                            mismatch_score=-3,
                                            open_gap_score=-4,
                                            extend_gap_score=-0.1,
                                            target_right_open_gap_score=0,
                                            target_right_extend_gap_score=0,
                                            target_left_extend_gap_score=0,
                                            target_left_open_gap_score=0,
                                           )

        alignments = {}

        for possibly_reversed, strand in [
            (self.protospacer_and_PAM, '+'),
            (utilities.reverse_complement(self.protospacer_and_PAM), '-'),
        ]:
            alignments[strand] = aligner.align(possibly_reversed, self.target_sequence)

        best_strand = max(alignments, key=lambda strand: alignments[strand].score)
        best_alignment = alignments[best_strand][0]

        return best_alignment, best_strand

    @memoized_property
    def insertions(self):
        alignment, strand = self.alignment_to_target

        # Insertions are gaps between consecutive spacer subsequences.

        spacer_subsequences, target_subsequences = alignment.aligned

        for (_, left_spacer_end), (right_spacer_start, _), (_, left_target_end), (right_target_start, _) in zip(spacer_subsequences, spacer_subsequences[1:], target_subsequences, target_subsequences[1:]):
            if left_spacer_end != right_spacer_start:
                insertion = {
                    'start_in_spacer': left_spacer_end,
                    'end_in_spacer': right_spacer_start - 1,
                    'starts_after_in_target': left_target_end - 1,
                    'ends_before_in_target': right_target_start,
                }
                print(insertion)

        #for (_, left_target_end), (right_target_start, _) in zip(target_subsequences, target_subsequences[1:]):
        #    if left_target_end != right_target_start:
        #        deletions.append((left_target_end, right_target_start - 1))

def infer_twin_pegRNA_features(pegRNAs,
                               target_name,
                               existing_features,
                               reference_sequences,
                              ):

    target_seq = reference_sequences[target_name]

    PBS_names_by_side = get_PBS_names_by_side_of_target(pegRNAs)
    pegRNA_names_by_side = get_pegRNA_names_by_side_of_target(pegRNAs)

    pegRNA_names = [pegRNA_.name for pegRNA_ in pegRNAs]

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
    substitutions = None

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

            substitutions = {
                target_name: {},
                pegRNA_names[0]: {},
                pegRNA_names[1]: {},
            }

            for offset, (pegRNAs_b, target_b) in enumerate(zip(intended_edit_seq, unedited_seq)):
                if pegRNAs_b != target_b:

                    positions = offset_to_positions[offset]

                    substitution_name = f'substitution_{positions[target_name]}_{target_b}-{pegRNAs_b}'

                    substitutions[target_name][substitution_name] = {
                        'position': positions[target_name],
                        'strand': '+',
                        'base': target_b,
                        'alternative_base': pegRNAs_b,
                    }

                    feature = gff.Feature.from_fields(seqname=target_name,
                                                      start=positions[target_name],
                                                      end=positions[target_name],
                                                      strand='+',
                                                      ID=substitution_name,
                                                     )
                
                    new_features[target_name, substitution_name] = feature

                    if pegRNA_names_by_side[5] in positions:
                        pegRNA_name = pegRNA_names_by_side[5]

                        # Note: convention on pegRNA base strandedness is a constant source
                        # of confusion.
                        pegRNA_base_effective = utilities.reverse_complement(pegRNAs_b)
                        target_base_effective = utilities.reverse_complement(target_b)

                        substitutions[pegRNA_name][substitution_name] = {
                            'position': positions[pegRNA_name],
                            'strand': '-',
                            'base': pegRNA_base_effective,
                            'alternative_base': target_base_effective,
                        }

                        feature = gff.Feature.from_fields(seqname=pegRNA_name,
                                                          start=positions[pegRNA_name],
                                                          end=positions[pegRNA_name],
                                                          strand='-',
                                                          ID=substitution_name,
                                                         )
                    
                        new_features[pegRNA_name, substitution_name] = feature

                    if pegRNA_names_by_side[3] in positions:
                        pegRNA_name = pegRNA_names_by_side[3]

                        pegRNA_base_effective = pegRNAs_b
                        target_base_effective = target_b

                        substitutions[pegRNA_name][substitution_name] = {
                            'position': positions[pegRNA_name],
                            'strand': '+',
                            'base': pegRNA_base_effective,
                            'alternative_base': target_base_effective,
                        }

                        feature = gff.Feature.from_fields(seqname=pegRNA_name,
                                                          start=positions[pegRNA_name],
                                                          end=positions[pegRNA_name],
                                                          strand='+',
                                                          ID=substitution_name,
                                                         )
                    
                        new_features[pegRNA_name, substitution_name] = feature

    results = {
        'deletion': deletion,
        'substitutions': substitutions,
        'new_features': new_features,
        'is_prime_del': is_prime_del,
        'intended_edit_seq': intended_edit_seq,
    }

    return results