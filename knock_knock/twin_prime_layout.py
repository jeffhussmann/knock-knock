from collections import defaultdict

import hits.visualize
from hits import interval, sam
from hits.utilities import memoized_property

import knock_knock.layout
import knock_knock.pegRNAs
import knock_knock.prime_editing_layout
import knock_knock.visualize
from knock_knock.outcome import *

other_side = knock_knock.prime_editing_layout.other_side

class Layout(knock_knock.prime_editing_layout.Layout):
    category_order = [
        ('wild type',
            ('clean',
             'mismatches',
             'short indel far from cut',
            ),
        ),
        ('intended edit',
            ('replacement',
             'partial replacement',
             'deletion',
            ),
        ),
        ('unintended rejoining of RT\'ed sequence',
            ('left RT\'ed, right not seen',
             'left RT\'ed, right not RT\'ed',
             'left RT\'ed, right RT\'ed',
             'left not seen, right RT\'ed',
             'left not RT\'ed, right RT\'ed',
            ),
        ),
        ('unintended rejoining of overlap-extended sequence',
            ('left not seen, right RT\'ed + overlap-extended',
             'left not RT\'ed, right RT\'ed + overlap-extended',
             'left RT\'ed, right RT\'ed + overlap-extended',
             'left RT\'ed + overlap-extended, right not seen',
             'left RT\'ed + overlap-extended, right not RT\'ed',
             'left RT\'ed + overlap-extended, right RT\'ed',
             'left RT\'ed + overlap-extended, right RT\'ed + overlap-extended',
            ),
        ),
        ('flipped pegRNA incorporation',
            ('left pegRNA',
             'right pegRNA',
             'both pegRNAs',
            ),
        ),
        ('deletion',
            ('clean',
             'mismatches',
             'multiple',
            ),
        ),
        ('duplication',
            ('simple',
             'iterated',
             'complex',
            ),
        ),
        ('insertion',
            ('clean',
             'mismatches',
            ),
        ),
        ('edit + indel',
            ('deletion',
             'insertion',
             'duplication',
            ),
        ),
        ('multiple indels',
            ('multiple indels',
            ),
        ),
        ('extension from intended annealing',
            ('n/a',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'hg38',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
            ),
        ),
        ('nonspecific amplification',
            ('hg19',
             'hg38',
             'bosTau7',
             'e_coli',
             'primer dimer',
             'short unknown',
             'plasmid',
            ),
        ),
        ('scaffold chimera',
            ('scaffold chimera',
            ),
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ins_size_to_split_at = 1
        self.categorized = False

    @memoized_property
    def parsimonious_split_target_and_donor_alignments(self):
        return interval.make_parsimonious(self.split_target_and_donor_alignments)

    @memoized_property
    def has_any_pegRNA_extension_al(self):
        chains = self.extension_chains_by_side
        contains_RTed_sequence = any(chains[side]['description'].startswith('RT\'ed') for side in ['left', 'right'])

        return contains_RTed_sequence

    @memoized_property
    def pegRNA_alignments_by_side_of_read(self):
        als = {}
        for side in ['left', 'right']:
            name = self.target_info.pegRNA_names_by_side_of_read[side]
            als[side] = self.pegRNA_alignments[name]

        return als

    def other_pegRNA_name(self, pegRNA_name):
        ti = self.target_info
        side = ti.pegRNA_name_to_side_of_read[pegRNA_name]
        other_name = ti.pegRNA_names_by_side_of_read[other_side[side]]
        return other_name

    def pegRNA_alignment_extends_pegRNA_alignment(self, first_pegRNA_al, second_pegRNA_al):
        if first_pegRNA_al is None or second_pegRNA_al is None:
            return None

        if first_pegRNA_al.reference_name == second_pegRNA_al.reference_name:
            return None

        if self.target_info.pegRNA_name_to_side_of_read[first_pegRNA_al.reference_name] == 'left':
            left_al, right_al = first_pegRNA_al, second_pegRNA_al
            right_al = second_pegRNA_al
        else:
            left_al, right_al = second_pegRNA_al, first_pegRNA_al

        extension_results = self.are_mutually_extending_from_shared_feature(left_al, 'overlap', right_al, 'overlap')
        if extension_results:
            status = 'definite'
        else:
            status = None

        return status

    def generate_extended_pegRNA_overlap_alignment(self, pegRNA_al_to_extend):
        if pegRNA_al_to_extend is None:
            return None

        other_pegRNA_name = self.other_pegRNA_name(pegRNA_al_to_extend.reference_name)
        return self.extend_alignment_from_shared_feature(pegRNA_al_to_extend, 'overlap', other_pegRNA_name, 'overlap')

    def find_pegRNA_alignment_extending_from_overlap(self, pegRNA_al_to_extend):
        if pegRNA_al_to_extend is None:
            return None

        # Prime-del strategies (or non-homologous flaps) don't have a natural overlap.
        if (pegRNA_al_to_extend.reference_name, 'overlap') not in self.target_info.features:
            return None

        other_pegRNA_name = self.other_pegRNA_name(pegRNA_al_to_extend.reference_name)
        candidate_als = self.pegRNA_alignments[other_pegRNA_name] 

        manually_extended_al = self.generate_extended_pegRNA_overlap_alignment(pegRNA_al_to_extend)
        if manually_extended_al is not None:
            candidate_als = candidate_als + [manually_extended_al]
        
        relevant_pegRNA_al = None
        
        for candidate_al in sorted(candidate_als, key=lambda al: al.query_length, reverse=True):
            status = self.pegRNA_alignment_extends_pegRNA_alignment(pegRNA_al_to_extend, candidate_al)
            if status == 'definite':
                relevant_pegRNA_al = candidate_al
                break
                
        return relevant_pegRNA_al

    def characterize_extension_chain_on_side(self, side):
        als = {}
    
        target_edge_al = self.target_edge_alignments[side]
        
        if target_edge_al is not None:
            als['first target'] = target_edge_al
        
            pegRNA_al, cropped_pegRNA_al, cropped_target_al = self.find_pegRNA_alignment_extending_target_edge_al(side, 'PBS')
            
            if pegRNA_al is not None:
                als['first pegRNA'] = pegRNA_al

                # Overwrite the target al so that only the cropped extent is used to determine
                # whether the pegRNA alignment contributed any novel query cover.
                als['first target'] = cropped_target_al
                
                overlap_extended_pegRNA_al = self.find_pegRNA_alignment_extending_from_overlap(pegRNA_al)
                
                if overlap_extended_pegRNA_al is not None:
                    als['second pegRNA'] = overlap_extended_pegRNA_al
                    
                    overlap_extended_target_al, _, _ = self.find_target_alignment_extending_pegRNA_alignment(overlap_extended_pegRNA_al, 'PBS')
                    
                    if overlap_extended_target_al is not None:
                        als['second target'] = overlap_extended_target_al
                        
        al_order = [
            'first target',
            'first pegRNA',
            'second pegRNA',
            'second target',
        ]

        query_covered = interval.get_disjoint_covered([])
        query_covered_incremental = {'none': query_covered}

        for al_order_i in range(len(al_order)):
            al_key = al_order[al_order_i]
            if al_key in als:
                als_up_to = [als[key] for key in al_order[:al_order_i + 1]]
                query_covered = interval.get_disjoint_covered(als_up_to)
                query_covered_incremental[al_key] = query_covered

        results = {
            'query_covered': query_covered,
            'query_covered_incremental': query_covered_incremental,
            'alignments': als,
        }

        return results

    @memoized_property
    def extension_chains_by_side(self):
        chains = {side: self.characterize_extension_chain_on_side(side) for side in ['left', 'right']}

        # Check whether any members of an extension chain on one side are not
        # necessary to make it to the other chain. (Warning: could imagine a
        # scenario in which it would be possible to remove from either the
        # left or right chain.)

        al_order = [
            'none',
            'first target',
            'first pegRNA',
            'second pegRNA',
            'second target',
        ]

        last_al_to_description = {
            'none': 'no target',
            'first target': 'not RT\'ed',
            'first pegRNA': 'RT\'ed',
            'second pegRNA': 'RT\'ed + overlap-extended',
            'second target': 'RT\'ed + overlap-extended',
        }

        possible_covers = set()

        if chains['left']['query_covered'] != chains['right']['query_covered']:
            for left_key in al_order:
                if left_key in chains['left']['alignments']:
                    for right_key in al_order:
                        if right_key in chains['right']['alignments']:
                            covered_left = chains['left']['query_covered_incremental'][left_key]
                            covered_right = chains['right']['query_covered_incremental'][right_key]

                            # Check if left and right overlap or abut each other.
                            if covered_left.end >= covered_right.start - 1:
                                possible_covers.add((left_key, right_key))

        #for left, right in sorted(possible_covers, key=lambda pair: (al_order.index(pair[0]), al_order.index(pair[1]))):
        #    print(left, right, al_order.index(left), al_order.index(right))
                            
        last_parsimonious_key = {}

        if possible_covers:
            last_parsimonious_key['left'], last_parsimonious_key['right'] = min(possible_covers, key=lambda pair: (al_order.index(pair[0]), al_order.index(pair[1])))
        else:
            for side in ['left', 'right']:
                last_parsimonious_key[side] = max(chains[side]['alignments'], key=al_order.index, default='none')

        for side in ['left', 'right']:
            key = last_parsimonious_key[side]

            chains[side]['description'] = last_al_to_description[key]

            last_index = al_order.index(key)
            chains[side]['parsimonious_alignments'] = [al for key, al in chains[side]['alignments'].items() if al_order.index(key) <= last_index]

            chains[side]['query_covered'] = chains[side]['query_covered_incremental'][key]

        # If one chain is absent and the other chain covers the whole read
        # (except possibly 2 nts at either edge), classify the missing side
        # as 'not seen'.

        not_covered_by_primers_minus_edges = self.not_covered_by_primers & self.whole_read_minus_edges(2)

        if not_covered_by_primers_minus_edges in chains['left']['query_covered']:
            if chains['right']['description'] == 'no target':
                chains['right']['description'] = 'not seen'

        if not_covered_by_primers_minus_edges in chains['right']['query_covered']:
            if chains['left']['description'] == 'no target':
                chains['left']['description'] = 'not seen'

        return chains

    @memoized_property
    def extension_chain_junction_microhomology(self):
        last_als = {}

        for side in ['left', 'right']:
            chain = self.extension_chains_by_side[side]
            if chain['description'] in ['not seen', 'no target']:
                last_al = None
            else:
                if chain['description'] == 'RT\'ed + overlap-extended':
                    if 'second target' in chain['alignments']:
                        last_al = chain['alignments']['second target']
                    else:
                        last_al = chain['alignments']['second pegRNA']

                else:
                    if chain['description'] == 'not RT\'ed':
                        last_al = chain['alignments']['first target']

                    elif chain['description'] == 'RT\'ed':
                        last_al = chain['alignments']['first pegRNA']

            last_als[side] = last_al

        return knock_knock.layout.junction_microhomology(self.target_info.reference_sequences, last_als['left'], last_als['right'])

    def get_extension_chain_edge(self, side):
        ''' Get the position of the far edge of an extension chain
        in the relevant coordinate system.
        '''
        ti = self.target_info

        this_side_pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
        other_side_pegRNA_name = self.other_pegRNA_name(this_side_pegRNA_name)

        PBS_end = ti.features[this_side_pegRNA_name, 'PBS'].end

        chain = self.extension_chains_by_side[side]
        if chain['description'] in ['not seen', 'no target']:
            relevant_edge = None
        else:
            if chain['description'] == 'RT\'ed + overlap-extended':
                if 'second target' in chain['alignments']:
                    al = chain['alignments']['second target']
                else:
                    al = chain['alignments']['second pegRNA']

                this_side_overlap = ti.features[this_side_pegRNA_name, 'overlap']
                other_side_overlap = ti.features[other_side_pegRNA_name, 'overlap']

                up_to_overlap_end = PBS_end - this_side_overlap.start + 1

                if al.reference_name == other_side_pegRNA_name:
                    extra_other_pegRNA = (al.reference_end - 1) - (other_side_overlap.end + 1) + 1

                    relevant_edge = up_to_overlap_end + extra_other_pegRNA

                elif al.reference_name == ti.target:
                    opposite_PBS_end = ti.features[other_side_pegRNA_name, 'PBS'].end 
                    up_to_opposite_PBS_end = opposite_PBS_end - (other_side_overlap.end + 1) + 1

                    opposite_target_PBS_name = ti.PBS_names_by_side_of_read[other_side[side]]
                    opposite_target_PBS = ti.features[ti.target, opposite_target_PBS_name]

                    if opposite_target_PBS.strand == '+':
                        extra_genomic = (opposite_target_PBS.start - 1) - al.reference_start + 1
                    else:
                        extra_genomic = (al.reference_end - 1) - (opposite_target_PBS.end + 1) + 1

                    relevant_edge = up_to_overlap_end + up_to_opposite_PBS_end + extra_genomic
                    
            else:
                if chain['description'] == 'not RT\'ed':
                    al = chain['alignments']['first target']

                    target_PBS_name = ti.PBS_names_by_side_of_read[side]
                    target_PBS = ti.features[ti.target, target_PBS_name]

                    # Positive values are towards the opposite nick,
                    # negative values are away from the opposite nick.

                    if target_PBS.strand == '+':
                        relevant_edge = (al.reference_end - 1) - target_PBS.end
                    else:
                        relevant_edge = target_PBS.start - al.reference_start

                elif chain['description'] == 'RT\'ed':
                    al = chain['alignments']['first pegRNA']

                    relevant_edge = PBS_end - al.reference_start
                
        return relevant_edge

    @memoized_property
    def has_intended_pegRNA_overlap(self):
        chains = self.extension_chains_by_side

        return (chains['left']['description'] == 'RT\'ed + overlap-extended' and
                chains['right']['description'] == 'RT\'ed + overlap-extended' and 
                chains['left']['query_covered'] == chains['right']['query_covered']
               )

    @memoized_property
    def intended_SNVs_replaced(self):
        als = self.pegRNA_extension_als
        positions_not_replaced = {side: self.alignment_SNV_summary(als[side])['mismatches'] for side in als}
        positions_replaced = {side: self.alignment_SNV_summary(als[side])['matches'] for side in als}

        any_positions_not_replaced = any(len(ps) > 0 for side, ps in positions_not_replaced.items())
        any_positions_replaced = any(len(ps) > 0 for side, ps in positions_replaced.items())

        if not any_positions_replaced:
            fraction_replaced = 'none'
        else:
            if any_positions_not_replaced:
                fraction_replaced = 'partial replacement'
            else:
                fraction_replaced = 'replacement'

        return fraction_replaced

    @memoized_property
    def is_intended_replacement(self):
        if self.target_info.pegRNA_intended_deletion is not None:
            status = False
        else:
            if not self.has_intended_pegRNA_overlap:
                status = False
            else:
                if self.target_info.pegRNA_SNVs is None:
                    status = 'replacement'
                elif self.intended_SNVs_replaced == 'none':
                    status = False
                else:
                    status = self.intended_SNVs_replaced

        return status

    @memoized_property
    def is_unintended_rejoining(self):
        ''' At least one side has RT'ed sequence, and together the extension
        chains cover the whole read.
        '''
        chains = self.extension_chains_by_side

        contains_RTed_sequence = any(chains[side]['description'].startswith('RT\'ed') for side in ['left', 'right'])

        left_covered = chains['left']['query_covered']
        right_covered = chains['right']['query_covered']

        combined_covered = left_covered | right_covered
        uncovered = self.not_covered_by_primers - combined_covered

        # Allow failure to explain the last few nts of the read.
        uncovered = uncovered & self.whole_read_minus_edges(2)

        return contains_RTed_sequence and uncovered.total_length == 0

    def register_unintended_rejoining(self):
        chains = self.extension_chains_by_side

        if any('overlap' in chains[side]['description'] for side in ['left', 'right']):
            self.category = 'unintended rejoining of overlap-extended sequence'
        else:
            self.category = 'unintended rejoining of RT\'ed sequence'

        self.subcategory = f'left {chains["left"]["description"]}, right {chains["right"]["description"]}'

        left_edge = self.get_extension_chain_edge('left')
        right_edge = self.get_extension_chain_edge('right')

        MH_nts = self.extension_chain_junction_microhomology

        self.outcome = knock_knock.prime_editing_layout.UnintendedRejoiningOutcome(left_edge, right_edge, MH_nts)

        self.relevant_alignments = list(chains['left']['alignments'].values()) + list(chains['right']['alignments'].values())

    @memoized_property
    def has_any_flipped_pegRNA_al(self):
        return {side for side in ['left', 'right'] if len(self.flipped_pegRNA_als[side]) > 0}

    def alignment_SNV_summary(self, al):
        ''' Identifies any positions in al that correspond to sequence differences
        between the target and pegRNAs and separates them based on whether they
        agree with al's reference sequence or not.
        ''' 

        ti = self.target_info
        SNVs = ti.pegRNA_SNVs
        
        positions_seen = {
            'matches': set(),
            'mismatches': set(),
        }

        if SNVs is None or al is None or al.is_unmapped:
            return positions_seen

        ref_seq = ti.reference_sequences[al.reference_name]

        pegRNA_SNP_positions = {SNVs[al.reference_name][name]['position'] for name in SNVs[al.reference_name]}

        for true_read_i, read_b, ref_i, ref_b, qual in sam.aligned_tuples(al, ref_seq):
            # Note: read_b and ref_b are as if the read is the forward strand
            if ref_i in pegRNA_SNP_positions:
                if read_b != ref_b:
                    positions_seen['mismatches'].add(ref_i)
                else:
                    positions_seen['matches'].add(ref_i)

        return positions_seen

    @memoized_property
    def scaffold_chimera(self):
        ''' Identify any alignments to plasmids that cover the entire scaffold and nearby amplicon primer. '''
        
        chimera_als = []
        
        for al in self.extra_alignments:
            primer = self.target_info.features.get((al.reference_name, 'AVA184'))
            scaffold = self.target_info.features.get((al.reference_name, 'scaffold'))
            if primer is not None and scaffold is not None:
                if sam.feature_overlap_length(al, primer) >= 5 and sam.feature_overlap_length(al, scaffold) == len(scaffold):
                    chimera_als.append(al)
                    
        return chimera_als

    def categorize(self):
        self.outcome = None

        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.no_alignments_detected:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'
            self.outcome = None

        elif self.has_any_pegRNA_extension_al:
            if self.is_intended_replacement:
                self.category = 'intended edit'
                self.subcategory = self.is_intended_replacement
                self.outcome = Outcome('n/a')
                self.relevant_alignments = self.target_edge_alignments_list + self.pegRNA_extension_als_list

            elif self.is_intended_deletion:
                self.category = 'intended edit'
                self.subcategory = 'deletion'
                self.outcome = DeletionOutcome(self.target_info.pegRNA_intended_deletion)
                self.relevant_alignments = self.target_edge_alignments_list + self.pegRNA_extension_als_list

            elif self.is_unintended_rejoining:
                self.register_unintended_rejoining()

            else:
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'

                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            if len(interesting_indels) == 0:
                if self.starts_at_expected_location:
                    # Need to check in case the intended replacements only involves minimal changes. 
                    if self.is_intended_replacement:
                        self.category = 'intended edit'
                        self.subcategory = self.is_intended_replacement
                        self.outcome = Outcome('n/a')
                        self.relevant_alignments = self.target_edge_alignments_list + self.pegRNA_extension_als_list

                    else:
                        self.category = 'wild type'

                        if len(self.non_pegRNA_SNVs) == 0 and len(uninteresting_indels) == 0:
                            self.subcategory = 'clean'
                            self.outcome = Outcome('n/a')

                        elif len(uninteresting_indels) == 1:
                            self.subcategory = 'short indel far from cut'

                            indel = uninteresting_indels[0]
                            if indel.kind == 'D':
                                self.outcome = DeletionOutcome(indel)
                            elif indel.kind == 'I':
                                self.outcome = InsertionOutcome(indel)
                            else:
                                raise ValueError(indel.kind)

                        elif len(uninteresting_indels) > 1:
                            self.category = 'uncategorized'
                            self.subcategory = 'uncategorized'
                            self.outcome = Outcome('n/a')

                        else:
                            self.subcategory = 'mismatches'
                            self.outcome = MismatchOutcome(self.non_pegRNA_SNVs)

                        self.relevant_alignments = [target_alignment]

                else:
                    self.category = 'uncategorized'
                    self.subcategory = 'uncategorized'
                    self.outcome = Outcome('n/a')

                    self.relevant_alignments = [target_alignment]

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.non_pegRNA_SNVs) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    if indel == self.target_info.pegRNA_intended_deletion:
                        self.category = 'intended edit'
                        self.subcategory = 'deletion'
                        self.relevant_alignments = [target_alignment] + self.pegRNA_extension_als_list

                    else:
                        self.category = 'deletion'
                        self.relevant_alignments = self.target_edge_alignments_list

                    self.outcome = DeletionOutcome(indel)

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.outcome = InsertionOutcome(indel)
                    self.relevant_alignments = [target_alignment]

            else: # more than one indel
                self.category = 'uncategorized'
                self.subcategory = 'uncategorized'
                self.details = 'n/a'
                self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.duplication_covers_whole_read:
            subcategory, ref_junctions, indels, als_with_donor_SNVs, merged_als = self.duplication
            self.outcome = DuplicationOutcome(ref_junctions)

            self.category = 'duplication'

            self.subcategory = subcategory
            self.relevant_alignments = merged_als

        elif self.scaffold_chimera:
            self.category = 'scaffold chimera'
            self.subcategory = 'scaffold chimera'
            self.details = 'n/a'
            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif len(self.has_any_flipped_pegRNA_al) > 0:
            self.category = 'flipped pegRNA incorporation'
            if len(self.has_any_flipped_pegRNA_al) == 1:
                side = sorted(self.has_any_flipped_pegRNA_al)[0]
                self.subcategory = f'{side} pegRNA'
            elif len(self.has_any_flipped_pegRNA_al) == 2:
                self.subcategory = f'both pegRNAs'
            else:
                raise ValueError(len(self.has_any_flipped_pegRNA_al))

            self.outcome = Outcome('n/a')
            self.relevant_alignments = self.target_edge_alignments_list + self.flipped_pegRNA_als['left'] + self.flipped_pegRNA_als['right']

        elif self.original_target_alignment_has_only_relevant_indels:
            self.register_simple_indels()

        else:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'

            self.relevant_alignments = self.uncategorized_relevant_alignments

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        if self.outcome is not None:
            # Translate positions to be relative to a registered anchor
            # on the target sequence.
            self.details = str(self.outcome.perform_anchor_shift(self.target_info.anchor))

        self.categorized = True

        return self.category, self.subcategory, self.details, self.outcome

    def manual_anchors(self, alignments_to_plot):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        ti = self.target_info

        manual_anchors = {}

        if ti.pegRNA_names is None:
            return manual_anchors

        overlap_feature = ti.features.get((ti.pegRNA_names[0], 'overlap'))
        if overlap_feature is not None:
            overlap_length = len(ti.features[ti.pegRNA_names[0], 'overlap'])

            overlap_offset_to_qs = defaultdict(dict)

            for side, expected_strand in [('left', '-'), ('right', '+')]:
                pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
                
                pegRNA_als = [al for al in alignments_to_plot if al.reference_name == pegRNA_name and sam.get_strand(al) == expected_strand]

                if len(pegRNA_als) == 0:
                    continue

                def priority_key(al):
                    is_extension_al = (al == self.pegRNA_extension_als['left']) or (al == self.pegRNA_extension_als['right'])
                    overlap_length = sam.feature_overlap_length(al, self.target_info.features[pegRNA_name, 'overlap'])
                    return is_extension_al, overlap_length
                
                pegRNA_als = sorted(pegRNA_als, key=priority_key)
                best_overlap_pegRNA_al = max(pegRNA_als, key=priority_key)
                
                overlap_offset_to_qs[side] = self.feature_offset_to_q(best_overlap_pegRNA_al, 'overlap')
                
            present_in_both = sorted(set(overlap_offset_to_qs['left']) & set(overlap_offset_to_qs['right']))
            present_in_either = sorted(set(overlap_offset_to_qs['left']) | set(overlap_offset_to_qs['right']))

            # If there is any offset present in both sides, use it as the anchor.
            # Otherwise, pick any offset present in either side arbitrarily.
            # If there is no such offset, don't make anchors for the pegRNAs.
            if overlap_length > 5 and present_in_either:
                if present_in_both:
                    anchor_offset = present_in_both[0]
                    qs = [overlap_offset_to_qs[side][anchor_offset] for side in ['left', 'right']] 
                    q = int(np.floor(np.mean(qs)))
                elif len(overlap_offset_to_qs['left']) > 0:
                    anchor_offset = sorted(overlap_offset_to_qs['left'])[0]
                    q = overlap_offset_to_qs['left'][anchor_offset]
                elif len(overlap_offset_to_qs['right']) > 0:
                    anchor_offset = sorted(overlap_offset_to_qs['right'])[0]
                    q = overlap_offset_to_qs['right'][anchor_offset]

                for side in ['left', 'right']:
                    pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
                    ref_p = ti.feature_offset_to_ref_p(pegRNA_name, 'overlap')[anchor_offset]
                    manual_anchors[pegRNA_name] = (q, ref_p)
                
        return manual_anchors

    def plot(self, relevant=True, manual_alignments=None, **manual_diagram_kwargs):
        if not self.categorized:
            self.categorize()

        ti = self.target_info

        feature_heights = {}
        label_offsets = {feature_name: 1 for feature_name in ti.PAM_features}
        label_overrides = {name: 'protospacer' for name in ti.protospacer_names}

        flip_target = ti.sequencing_direction == '-'

        color_overrides = {}
        if ti.primer_names is not None:
            for primer_name in ti.primer_names:
                color_overrides[primer_name] = 'lightgrey'

        pegRNA_names = ti.pegRNA_names
        if pegRNA_names is None:
            pegRNA_names = []
        else:
            for pegRNA_name in pegRNA_names:
                color = ti.pegRNA_name_to_color[pegRNA_name]
                light_color = hits.visualize.apply_alpha(color, 0.5)
                color_overrides[pegRNA_name] = color
                color_overrides[pegRNA_name, 'protospacer'] = light_color
                ps_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)
                color_overrides[ps_name] = light_color

                PAM_name = f'{ps_name}_PAM'
                color_overrides[PAM_name] = color

        # Draw protospacer features on the same side as their nick.
        for feature_name, feature in ti.PAM_features.items():
            if (feature.strand == '+' and not flip_target) or (feature.strand == '-' and flip_target):
                feature_heights[feature_name] = -1

        for feature_name, feature in ti.protospacer_features.items():
            if (feature.strand == '+' and not flip_target) or (feature.strand == '-' and flip_target):
                feature_heights[feature_name] = -1

        features_to_show = {*ti.features_to_show}
        features_to_show.update({(ti.target, name) for name in ti.protospacer_names})
        features_to_show.update({(ti.target, name) for name in ti.PAM_features})


        for pegRNA_name in ti.pegRNA_names:
            PBS_name = knock_knock.pegRNAs.PBS_name(pegRNA_name)
            features_to_show.add((ti.target, PBS_name))
            label_overrides[PBS_name] = None
            feature_heights[PBS_name] = 0.5

        for deletion in self.target_info.pegRNA_programmed_deletions:
            label_overrides[deletion.ID] = f'programmed deletion ({len(deletion)} nts)'
            feature_heights[deletion.ID] = -0.5

        if 'features_to_show' in manual_diagram_kwargs:
            features_to_show.update(manual_diagram_kwargs.pop('features_to_show'))

        if 'color_overrides' in manual_diagram_kwargs:
            color_overrides.update(manual_diagram_kwargs.pop('color_overrides'))

        if 'label_overrides' in manual_diagram_kwargs:
            label_overrides.update(manual_diagram_kwargs.pop('label_overrides'))

        if 'label_offsets' in manual_diagram_kwargs:
            label_offsets.update(manual_diagram_kwargs.pop('label_offsets'))

        refs_to_draw= {ti.target, *pegRNA_names}
        if 'refs_to_draw' in manual_diagram_kwargs:
            refs_to_draw.update(manual_diagram_kwargs.pop('refs_to_draw'))

        if manual_alignments is not None:
            als_to_plot = manual_alignments
        elif relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.uncategorized_relevant_alignments

        manual_anchors = manual_diagram_kwargs.get('manual_anchors', self.manual_anchors(als_to_plot))

        diagram_kwargs = dict(
            draw_sequence=True,
            flip_target=flip_target,
            split_at_indels=True,
            label_offsets=label_offsets,
            features_to_show=features_to_show,
            manual_anchors=manual_anchors,
            refs_to_draw=refs_to_draw,
            label_overrides=label_overrides,
            inferred_amplicon_length=self.inferred_amplicon_length,
            center_on_primers=True,
            color_overrides=color_overrides,
            feature_heights=feature_heights,
        )

        diagram_kwargs.update(**manual_diagram_kwargs)

        diagram = knock_knock.visualize.ReadDiagram(als_to_plot,
                                                    ti,
                                                    **diagram_kwargs,
                                                   )

        # Note that diagram.alignments may be different than als_to_plot
        # due to application of parsimony.

        # Draw the pegRNAs.
        if any(al.reference_name in pegRNA_names for al in diagram.alignments):
            ref_ys = {}
            ref_ys['left'] = diagram.max_y + diagram.target_and_donor_y_gap
            ref_ys['right'] = ref_ys['left'] + 7 * diagram.gap_between_als

            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = {}

            left_name = ti.pegRNA_names_by_side_of_read['left']
            left_visible = any(al.reference_name == left_name for al in diagram.alignments)

            right_name = ti.pegRNA_names_by_side_of_read['right']
            right_visible = any(al.reference_name == right_name for al in diagram.alignments)

            ref_p_to_xs['left'] = diagram.draw_reference(left_name, ref_ys['left'],
                                                         flip=True,
                                                         label_features=left_visible and (not right_visible),
                                                         visible=left_visible,
                                                        )

            diagram.max_x = max(old_max_x, ref_p_to_xs['left'](0))

            ref_p_to_xs['right'] = diagram.draw_reference(right_name, ref_ys['right'],
                                                          flip=False,
                                                          label_features=True,
                                                          visible=right_visible,
                                                         )

            diagram.min_x = min(old_min_x, ref_p_to_xs['right'](0))

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            if self.manual_anchors and (left_name, 'overlap') in ti.features:
                offset_to_ref_ps = ti.feature_offset_to_ref_p(left_name, 'overlap')
                overlap_xs = sorted([ref_p_to_xs['left'](offset_to_ref_ps[0]), ref_p_to_xs['left'](offset_to_ref_ps[max(offset_to_ref_ps)])])

                overlap_xs = knock_knock.visualize.adjust_edges(overlap_xs)

                overlap_color = ti.features[left_name, 'overlap'].attribute['color']
                    
                diagram.ax.fill_betweenx([ref_ys['left'], ref_ys['right'] + diagram.ref_line_width + diagram.feature_line_width],
                                         [overlap_xs[0], overlap_xs[0]],
                                         [overlap_xs[1], overlap_xs[1]],
                                         color=overlap_color,
                                         alpha=0.3,
                                         visible=left_visible and right_visible,
                                        )

                text_x = np.mean(overlap_xs)
                text_y = np.mean([ref_ys['left'] + diagram.feature_line_width, ref_ys['right']])
                diagram.ax.annotate('overlap',
                                    xy=(text_x, text_y),
                                    color=overlap_color,
                                    ha='center',
                                    va='center',
                                    size=diagram.font_sizes['feature_label'],
                                    weight='bold',
                                    visible=left_visible and right_visible,
                                   )

            diagram.update_size()

        return diagram
