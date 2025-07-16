from collections import defaultdict

from hits import interval, sam
from hits.utilities import memoized_property

from . import prime_editing
import knock_knock.target_info
import knock_knock.visualize.architecture
from knock_knock.outcome import *

class Architecture(prime_editing.Architecture):
    category_order = [
        (
            'wild type',
            (
                'clean',
                'mismatches',
                'short indel far from cut',
            ),
        ),
        (
            'intended edit',
            (
                'replacement',
                'deletion',
            ),
        ),
        (
            'partial replacement',
            (
                'left pegRNA',
                'right pegRNA',
                'both pegRNAs',
                'single pegRNA (ambiguous)',
            ),
        ),
        (
            'unintended rejoining of RT\'ed sequence',
            (
                'left RT\'ed, right RT\'ed',
                'left RT\'ed, right not RT\'ed',
                'left not RT\'ed, right RT\'ed',
                'left RT\'ed, right not seen',
                'left not seen, right RT\'ed',
            ),
        ),
        (
            'unintended rejoining of overlap-extended sequence',
            (
                'left RT\'ed + overlap-extended, right RT\'ed + overlap-extended',
                'left RT\'ed + overlap-extended, right RT\'ed',
                'left RT\'ed, right RT\'ed + overlap-extended',
                'left RT\'ed + overlap-extended, right not RT\'ed',
                'left not RT\'ed, right RT\'ed + overlap-extended',
                'left RT\'ed + overlap-extended, right not seen',
                'left not seen, right RT\'ed + overlap-extended',
            ),
        ),
        (
            'multistep unintended rejoining of RT\'ed sequence',
            (
                'left RT\'ed, right RT\'ed',
                'left RT\'ed, right indel',
                'left indel, right RT\'ed',
            ),
        ),
        (
            'flipped pegRNA incorporation',
            (
                'left pegRNA',
                'right pegRNA',
                'both pegRNAs',
            ),
        ),
        (
            'deletion',
            (
                'clean',
                'mismatches',
                'multiple',
            ),
        ),
        (
            'duplication',
            (
                'simple',
                'iterated',
                'complex',
            ),
        ),
        (
            'insertion',
            (
                'clean',
                'mismatches',
            ),
        ),
        (
            'edit + indel',
            (
                'deletion',
                'insertion',
                'duplication',
            ),
        ),
        (
            'multiple indels',
            (
                'multiple indels',
            ),
        ),
        (
            'genomic insertion',
            (
                'hg19',
                'hg38',
                'macFas5',
                'mm10',
                'bosTau7',
                'e_coli',
                'phiX',
            ),
        ),
        (
            'inversion',
            (
                'inversion',
            ),
        ),
        (
            'incorporation of extra sequence',
            (
                'has RT\'ed extension',
                'no RT\'ed extension',
            ),
        ),
        (
            'uncategorized',
            (
                'uncategorized',
                'low quality', 
            ),
        ),
        (
            'nonspecific amplification',
            (
                'hg19',
                'hg38',
                'macFas5',
                'T2T-MFA8v1.0',
                'mm10',
                'bosTau7',
                'e_coli',
                'b_subtilis',
                'phiX',
                'primer dimer',
                'short unknown',
                'extra sequence',
            ),
        ),
        (
            'phiX',
            (
                'phiX',
            ),
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ins_size_to_split_at = 1
        self.categorized = False

    @memoized_property
    def pegRNA_alignments_by_side_of_read(self):
        als = {}
        for side in ['left', 'right']:
            name = self.pegRNA_names_by_side_of_read[side]
            als[side] = self.pegRNA_alignments_by_pegRNA_name[name]

        return als

    def other_pegRNA_name(self, pegRNA_name):
        side = self.pegRNA_name_to_side_of_read[pegRNA_name]
        other_name = self.pegRNA_names_by_side_of_read[knock_knock.target_info.other_side[side]]
        return other_name

    def pegRNA_alignment_extends_pegRNA_alignment(self, first_pegRNA_al, second_pegRNA_al):
        if first_pegRNA_al is None or second_pegRNA_al is None:
            return None

        if first_pegRNA_al.reference_name == second_pegRNA_al.reference_name:
            return None

        if self.pegRNA_name_to_side_of_read[first_pegRNA_al.reference_name] == 'left':
            left_al, right_al = first_pegRNA_al, second_pegRNA_al
            right_al = second_pegRNA_al
        else:
            left_al, right_al = second_pegRNA_al, first_pegRNA_al

        extension_results = self.are_mutually_extending_from_shared_feature(left_al, 'overlap', right_al, 'overlap')

        if extension_results['status'] == 'definite':
            status = 'definite'
        elif extension_results['status'] == 'possible':
            status = 'possible'
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
        candidate_als = self.pegRNA_alignments_by_pegRNA_name[other_pegRNA_name] 

        manually_extended_al = self.generate_extended_pegRNA_overlap_alignment(pegRNA_al_to_extend)
        if manually_extended_al is not None:
            candidate_als = candidate_als + [manually_extended_al]
        
        relevant_pegRNA_al = None
        
        for candidate_al in sorted(candidate_als, key=lambda al: al.query_length, reverse=True):
            status = self.pegRNA_alignment_extends_pegRNA_alignment(pegRNA_al_to_extend, candidate_al)
            if status == 'definite' or (status == 'possible' and self.target_info.pegRNA_pair.is_prime_del):
                relevant_pegRNA_al = candidate_al
                break
                
        return relevant_pegRNA_al

    def characterize_extension_chain_on_side(self, side, require_definite=True):
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

                if side == 'left':
                    left_al, right_al = pegRNA_al, overlap_extended_pegRNA_al
                elif side == 'right':
                    left_al, right_al = overlap_extended_pegRNA_al, pegRNA_al
                else:
                    raise ValueError

                cropped = sam.crop_to_best_switch_point(left_al, right_al, self.target_info.reference_sequences)

                if side == 'left':
                    cropped_pegRNA_al, cropped_overlap_extended_pegRNA_al = cropped['left'], cropped['right']
                elif side == 'right':
                    cropped_pegRNA_al, cropped_overlap_extended_pegRNA_al = cropped['right'], cropped['left']
                else:
                    raise ValueError

                contribution_from_overlap_extended = not (interval.get_covered(cropped_overlap_extended_pegRNA_al) - interval.get_covered(cropped_pegRNA_al)).is_empty

                if contribution_from_overlap_extended or (overlap_extended_pegRNA_al is not None and self.target_info.pegRNA_pair.is_prime_del):
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

        for al_order_i, al_key in enumerate(al_order):
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

        # If right covers part of right side not covered by left, or left
        # covers part of left side not covered by right, look for joint covers.

        left_covered = chains['left']['query_covered']
        right_covered = chains['right']['query_covered']

        if not left_covered.is_empty and not right_covered.is_empty:
            if (right_covered.end > left_covered.end) or \
               (left_covered.start < right_covered.start):

                for left_key in al_order:
                    if left_key in chains['left']['alignments']:
                        for right_key in al_order:
                            if right_key in chains['right']['alignments']:
                                covered_left = chains['left']['query_covered_incremental'][left_key]
                                covered_right = chains['right']['query_covered_incremental'][right_key]

                                # Check if left and right overlap or abut each other.
                                if covered_left.end >= covered_right.start - 1:
                                    possible_covers.add((left_key, right_key))

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
            chains[side]['parsimonious_alignments'] = [
                al
                for key, al in chains[side]['alignments'].items()
                if al_order.index(key) <= last_index
            ]

            chains[side]['query_covered'] = chains[side]['query_covered_incremental'][key]

        last_als = {
            side: chains[side]['alignments'][last_parsimonious_key[side]]
            if last_parsimonious_key[side] != 'none' else None
            for side in ['left', 'right']
        }

        cropped_als = sam.crop_to_best_switch_point(last_als['left'], last_als['right'], self.target_info.reference_sequences)

        for side in ['left', 'right']:
            chains[side]['cropped_last_al'] = cropped_als[side]

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

        # If a putatively overlap-extended chain ends in a target alignment that is a short indel away from
        # the target edge alignment from the other chain:

        left_al = None
        right_al = None

        if chains['left']['description'] == 'RT\'ed + overlap-extended' and \
           chains['right']['description'] == 'not RT\'ed':
            
            left_al = chains['left']['alignments'].get('second target')
            right_al = chains['right']['alignments'].get('first target')
                
        elif chains['left']['description'] == 'not RT\'ed' and \
             chains['right']['description'] == 'RT\'ed + overlap-extended':
            
            left_al = chains['left']['alignments'].get('first target')
            right_al = chains['right']['alignments'].get('second target')
            
        if left_al is not None and right_al is not None:
            merged_al = sam.merge_adjacent_alignments(left_al, right_al, self.target_info.reference_sequences, max_deletion_length=2, max_insertion_length=2)

            if merged_al is not None:
                if chains['right']['description'] == 'not RT\'ed':
                    chains['left']['alignments']['second target'] = merged_al
                    chains['right']['alignments']['first target'] = merged_al

                    chains['right']['alignments']['first pegRNA'] = chains['left']['alignments']['second pegRNA']
                    chains['right']['alignments']['second pegRNA'] = chains['left']['alignments']['first pegRNA']
                    chains['right']['alignments']['second target'] = chains['left']['alignments']['first target']

                elif chains['left']['description'] == 'not RT\'ed':
                    chains['left']['alignments']['first target'] = merged_al
                    chains['right']['alignments']['second target'] = merged_al

                    chains['left']['alignments']['first pegRNA'] = chains['right']['alignments']['second pegRNA']
                    chains['left']['alignments']['second pegRNA'] = chains['right']['alignments']['first pegRNA']
                    chains['left']['alignments']['second target'] = chains['right']['alignments']['first target']
                
                else:
                    raise ValueError

                chains['left']['description'] = 'RT\'ed + overlap-extended'
                chains['right']['description'] = 'RT\'ed + overlap-extended'

                query_covered = chains['left']['query_covered'] | chains['right']['query_covered']

                chains['left']['query_covered'] = query_covered
                chains['right']['query_covered'] = query_covered

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

        return knock_knock.architecture.junction_microhomology(self.target_info.reference_sequences, last_als['left'], last_als['right'])

    def get_extension_chain_edge(self, side):
        ''' Get the position of the far edge of an extension chain
        in the relevant coordinate system.
        '''
        ti = self.target_info

        this_side_pegRNA_name = self.pegRNA_names_by_side_of_read[side]
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

                    opposite_target_PBS_name = ti.PBS_names_by_side_of_read[knock_knock.target_info.other_side[side]]
                    opposite_target_PBS = ti.features[ti.target, opposite_target_PBS_name]

                    if opposite_target_PBS.strand == self.sequencing_direction:
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
                    al = chain['cropped_last_al']

                    relevant_edge = PBS_end - al.reference_start
                
        return relevant_edge

    @memoized_property
    def has_intended_pegRNA_overlap(self):
        chains = self.extension_chains_by_side

        return (
            chains['left']['description'] == 'RT\'ed + overlap-extended' and
            chains['right']['description'] == 'RT\'ed + overlap-extended' and 
            chains['left']['query_covered'] == chains['right']['query_covered']
        )

    @memoized_property
    def is_intended_or_partial_replacement(self):
        if self.target_info.pegRNA_programmed_deletion is not None:
            status = False
        else:
            if not self.has_intended_pegRNA_overlap:
                status = False
            else:
                if self.target_info.pegRNA_substitutions is None:
                    status = True
                elif not self.has_pegRNA_substitution:
                    status = False
                else:
                    status = True

        return status

    @memoized_property
    def non_programmed_edit_mismatches(self):
        mismatches_seen = set()

        for al in self.pegRNA_extension_als_from_either_side_list:
            mismatches = knock_knock.architecture.get_mismatch_info(al, self.target_info.reference_sequences)

            programmed_ps = self.target_info.pegRNA_pair.programmed_substitution_ps[al.reference_name]

            for true_read_p, read_b, ref_p, ref_b, q in mismatches:
                
                # ref_p might be outside of edit portion or might be a programmed substitution.
                edit_p = self.target_info.pegRNA_pair.pegRNA_coords_to_edit_coords[al.reference_name].get(ref_p)
                
                if edit_p is not None and ref_p not in programmed_ps:
                    mismatches_seen.add(knock_knock.target_info.Mismatch(edit_p, read_b))
                    
        mismatches = knock_knock.target_info.Mismatches(mismatches_seen)

        return mismatches

    def register_intended_replacement(self):
        if self.pegRNA_substitution_string == self.full_incorporation_pegRNA_substitution_string:
            self.category = 'intended edit'
            self.subcategory = 'replacement'
        else:
            self.category = 'partial replacement'

            if len(self.pegRNAs_that_explain_all_substitutions) == 0:
                self.subcategory = 'both pegRNAs'
            elif len(self.pegRNAs_that_explain_all_substitutions) == 2:
                self.subcategory = 'single pegRNA (ambiguous)'
            elif self.pegRNA_names_by_side_of_read['left'] in self.pegRNAs_that_explain_all_substitutions:
                self.subcategory = 'left pegRNA'
            elif self.pegRNA_names_by_side_of_read['right'] in self.pegRNAs_that_explain_all_substitutions:
                self.subcategory = 'right pegRNA'
            else:
                raise ValueError

        if self.mode == 'nanopore':
            mismatches = []
        else:
            mismatches = self.non_pegRNA_mismatches

        self.Details = Details(programmed_substitution_read_bases=self.pegRNA_substitution_string,
                               mismatches=mismatches,
                               non_programmed_edit_mismatches=self.non_programmed_edit_mismatches,
                               deletions=[],
                               insertions=[],
                               integrase_sites=self.integrase_sites_in_chains,
                              )

        self.relevant_alignments = self.intended_edit_relevant_alignments

    @memoized_property
    def intended_edit_relevant_alignments(self):
        return self.target_edge_alignments_list + self.pegRNA_extension_als_list

    @memoized_property
    def contains_RTed_sequence(self):
        chains = self.extension_chains_by_side

        contains_RTed_sequence = {
            side for side in ['left', 'right']
            if chains[side]['description'].startswith('RT\'ed')
        }

        return contains_RTed_sequence

    @memoized_property
    def uncovered_by_extension_chains(self):
        chains = self.extension_chains_by_side

        left_covered = chains['left']['query_covered']
        right_covered = chains['right']['query_covered']

        combined_covered = left_covered | right_covered
        uncovered = self.not_covered_by_primers - combined_covered

        # Allow failure to explain the last few nts of the read.
        uncovered = uncovered & self.whole_read_minus_edges(2)

        return uncovered

    @memoized_property
    def is_unintended_rejoining(self):
        ''' At least one side has RT'ed sequence, and together the extension
        chains cover the whole read.
        '''
        return self.contains_RTed_sequence and self.uncovered_by_extension_chains.total_length == 0

    @memoized_property
    def integrase_sites_in_chains(self):
        pegRNA_pair = self.target_info.pegRNA_pair
        chains = self.extension_chains_by_side

        edges = {side: self.get_extension_chain_edge(side) for side in ['left', 'right']}

        integrase_sites = []

        chains_are_distinct = chains['left']['query_covered'] != chains['right']['query_covered']

        side_and_strands = [('left', '+')]

        if chains_are_distinct:
            side_and_strands.append(('right', '-'))

        for side, strand in side_and_strands:
            if 'overlap' in chains[side]['description']:
                relevant_threshold = pegRNA_pair.complete_integrase_site_ends_in_RT_and_overlap_extended_target_sequence[strand]
            elif chains[side]['description'].startswith('RT\'ed'):
                relevant_threshold = pegRNA_pair.complete_integrase_site_ends_in_RT_extended_target_sequence[strand]
            else:
                continue

            if relevant_threshold is not None:
                threshold, label = relevant_threshold

                if edges[side] >= threshold:
                    integrase_sites.append(label)

        return integrase_sites

    def register_unintended_rejoining(self):
        chains = self.extension_chains_by_side

        edges = {side: self.get_extension_chain_edge(side) for side in ['left', 'right']}

        if any('overlap' in chains[side]['description'] for side in ['left', 'right']):
            self.category = 'unintended rejoining of overlap-extended sequence'

        else:
            self.category = 'unintended rejoining of RT\'ed sequence'

        if self.flipped:
            possibly_flipped_side = {
                'left': 'right',
                'right': 'left',
            }
        else:
            possibly_flipped_side = {
                'left': 'left',
                'right': 'right',
            }

        self.subcategory = f'left {chains[possibly_flipped_side["left"]]["description"]}, right {chains[possibly_flipped_side["right"]]["description"]}'

        MH_nts = self.extension_chain_junction_microhomology

        details_kwargs = dict(
            junction_microhomology_length=MH_nts,
            integrase_sites=self.integrase_sites_in_chains,
        )

        if edges[possibly_flipped_side['left']] is not None:
            details_kwargs['left_rejoining_edge'] = edges[possibly_flipped_side['left']]

        if edges[possibly_flipped_side['right']] is not None:
            details_kwargs['right_rejoining_edge'] = edges[possibly_flipped_side['right']]

        self.Details = Details(**details_kwargs)

        als_by_ref = defaultdict(list)
        for al in list(chains['left']['alignments'].values()) + list(chains['right']['alignments'].values()):
            als_by_ref[al.reference_name].append(al)

        self.relevant_alignments = []
        for ref_name, als in als_by_ref.items():
            self.relevant_alignments.extend(sam.make_noncontained(als))

    @memoized_property
    def extension_chain_gap_covers(self):
        def covers_all_uncovered(al):
            not_covered_by_al = self.uncovered_by_extension_chains - interval.get_covered(al)
            return not_covered_by_al.total_length == 0

        covers = [al for al in self.target_alignments if covers_all_uncovered(al)]

        return covers

    @memoized_property
    def is_multistep_unintended_rejoining(self):
        return self.contains_RTed_sequence and self.uncovered_by_extension_chains.total_length > 0 and len(self.extension_chain_gap_covers) > 0

    def register_multistep_unintended_rejoining(self):
        self.category = 'multistep unintended rejoining of RT\'ed sequence'

        if 'left' in self.contains_RTed_sequence and 'right' in self.contains_RTed_sequence:
            self.subcategory = 'left RT\'ed, right RT\'ed'
        elif 'left' in self.contains_RTed_sequence:
            self.subcategory = 'left RT\'ed, right indel'
        elif 'right' in self.contains_RTed_sequence:
            self.subcategory = 'left indel, right RT\'ed'
        else:
            raise ValueError

        self.relevant_alignments = self.extension_chains_by_side['left']['parsimonious_alignments'] + \
                                   self.extension_chains_by_side['right']['parsimonious_alignments'] + \
                                   self.extension_chain_gap_covers

    @memoized_property
    def has_any_flipped_pegRNA_al(self):
        return {side for side, als in self.flipped_pegRNA_als.items() if len(als) > 0}

    def convert_target_alignment_edge_to_nick_coordinate(self, al, start_or_end):
        ti = self.target_info
        target_PBS = ti.features[ti.target, ti.PBS_names_by_side_of_read['left']]

        if start_or_end == 'start':
            reference_edge = al.reference_start
        elif start_or_end == 'end':
            reference_edge = al.reference_end - 1
        
        if target_PBS.strand == '+':
            converted_edge = reference_edge - target_PBS.end
        else:
            converted_edge = target_PBS.start - reference_edge

        return converted_edge

    def categorize(self):
        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.aligns_to_phiX:
            self.category = 'phiX'
            self.subcategory = 'phiX'

            self.relevant_alignments = [self.longest_phiX_alignment]

        elif self.no_alignments_detected:
            self.register_uncategorized()

        elif self.is_intended_or_partial_replacement:
            self.register_intended_replacement()

        elif self.is_intended_deletion:
            self.category = 'intended edit'
            self.subcategory = 'deletion'
            self.Details = Details(programmed_substitution_read_bases=self.pegRNA_substitution_string,
                                   mismatches=self.non_pegRNA_mismatches,
                                   non_programmed_edit_mismatches=self.non_programmed_edit_mismatches,
                                   deletions=[self.target_info.pegRNA_programmed_deletion],
                                   insertions=[],
                                  )
            self.relevant_alignments = self.intended_edit_relevant_alignments

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            deletions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'D']
            insertions = [indel for indel in interesting_indels + uninteresting_indels if indel.kind == 'I']

            self.Details = Details(deletions=deletions, insertions=insertions, mismatches=self.non_pegRNA_mismatches)

            if len(interesting_indels) == 0:
                if self.starts_at_expected_location:
                    self.category = 'wild type'

                    if len(self.non_pegRNA_mismatches) == 0 and len(uninteresting_indels) == 0:
                        self.subcategory = 'clean'

                    elif len(uninteresting_indels) == 1:
                        self.subcategory = 'short indel far from cut'

                    elif len(uninteresting_indels) > 1:
                        self.register_uncategorized()

                    else:
                        self.subcategory = 'mismatches'

                    self.relevant_alignments = [target_alignment]

                else:
                    self.register_uncategorized()

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.non_pegRNA_mismatches) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    if indel == self.target_info.pegRNA_programmed_deletion:
                        self.category = 'intended edit'
                        self.subcategory = 'deletion'
                        self.relevant_alignments = [target_alignment] + self.pegRNA_extension_als_list

                    else:
                        self.category = 'deletion'
                        self.relevant_alignments = self.target_edge_alignments_list

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.relevant_alignments = [target_alignment]

            else: # more than one indel
                self.register_uncategorized()

        elif self.duplication_covers_whole_read:
            subcategory, ref_junctions, indels, als_with_donor_substitutions, merged_als = self.duplication
            self.Details = Details(duplication_junctions=ref_junctions)

            self.category = 'duplication'

            self.subcategory = subcategory
            self.relevant_alignments = merged_als

        elif self.inversion:
            self.category = 'inversion'
            self.subcategory = 'inversion'

            self.relevant_alignments = self.target_edge_alignments_list + self.inversion

        elif self.is_multistep_unintended_rejoining:
            self.register_multistep_unintended_rejoining()

        elif len(self.has_any_flipped_pegRNA_al) > 0:
            self.category = 'flipped pegRNA incorporation'

            if len(self.has_any_flipped_pegRNA_al) == 1:
                side = sorted(self.has_any_flipped_pegRNA_al)[0]
                self.subcategory = f'{side} pegRNA'

            elif len(self.has_any_flipped_pegRNA_al) == 2:
                self.subcategory = f'both pegRNAs'

            else:
                raise ValueError(len(self.has_any_flipped_pegRNA_al))

            self.relevant_alignments = self.uncategorized_relevant_alignments

        elif self.original_target_alignment_has_only_relevant_indels:
            self.register_indels_in_original_alignment()

        elif self.nonredundant_extra_alignments:
            self.register_incorporation_of_extra_sequence()

        else:
            self.register_uncategorized()

        self.relevant_alignments = sam.make_nonredundant(self.relevant_alignments)

        self.details = str(self.Details)

        self.categorized = True

        return self.category, self.subcategory, self.details, self.Details

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
                pegRNA_name = self.pegRNA_names_by_side_of_read[side]
                
                pegRNA_als = [al for al in alignments_to_plot if al.reference_name == pegRNA_name and sam.get_strand(al) == expected_strand]

                if len(pegRNA_als) == 0:
                    continue

                def priority_key(al):
                    is_extension_al = (al == self.extension_chains_by_side['left']['alignments'].get('first pegRNA')) or \
                                      (al == self.extension_chains_by_side['right']['alignments'].get('first pegRNA'))

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
                    pegRNA_name = self.pegRNA_names_by_side_of_read[side]
                    ref_p = ti.feature_offset_to_ref_p(pegRNA_name, 'overlap')[anchor_offset]
                    manual_anchors[pegRNA_name] = (q, ref_p)
                
        return manual_anchors

    def plot(self,
             relevant=True,
             manual_alignments=None,
             annotate_overlap=True,
             label_integrase_features=False,
             draw_pegRNAs=True,
             label_pegRNAs=False,
             extra_features_to_show=None,
             extra_label_overrides=None,
             **manual_diagram_kwargs,
            ):

        plot_parameters = self.plot_parameters()

        features_to_show = manual_diagram_kwargs.pop('features_to_show', plot_parameters['features_to_show'])
        label_overrides = manual_diagram_kwargs.pop('label_overrides', plot_parameters['label_overrides'])
        label_offsets = manual_diagram_kwargs.pop('label_offsets', plot_parameters['label_offsets'])
        feature_heights = manual_diagram_kwargs.pop('feature_heights', plot_parameters['feature_heights'])
        color_overrides = manual_diagram_kwargs.pop('color_overrides', plot_parameters['color_overrides'])

        if extra_features_to_show is not None:
            features_to_show.update(extra_features_to_show)

        if extra_label_overrides is not None:
            label_overrides.update(extra_label_overrides)

        flip_target = plot_parameters['flip_target']

        if relevant and not self.categorized:
            self.categorize()

        ti = self.target_info

        if label_integrase_features:
            for ref_name, name in ti.integrase_sites:
                if 'right' in name:
                    label_offsets[name] = 1
                    features_to_show.add((ref_name, name))
                if 'left' in name:
                    label_offsets[name] = 2
                    features_to_show.add((ref_name, name))

        if 'refs_to_draw' in manual_diagram_kwargs:
            refs_to_draw = manual_diagram_kwargs.pop('refs_to_draw')
        else:
            refs_to_draw = set()

            if ti.amplicon_length < 10000:
                refs_to_draw.add(ti.target)

            if draw_pegRNAs:
                refs_to_draw.update(ti.pegRNA_names)

        invisible_references = manual_diagram_kwargs.get('invisible_references', set())

        if manual_alignments is not None:
            als_to_plot = manual_alignments
        elif relevant:
            als_to_plot = self.relevant_alignments
        else:
            als_to_plot = self.uncategorized_relevant_alignments

        if relevant:
            manual_anchors = manual_diagram_kwargs.get('manual_anchors', self.manual_anchors(als_to_plot))
            inferred_amplicon_length = self.inferred_amplicon_length
        else:
            manual_anchors = {}
            inferred_amplicon_length = None

        if 'phiX' in ti.supplemental_indices:
            supplementary_reference_sequences = ti.supplemental_reference_sequences('phiX')
        else:
            supplementary_reference_sequences = {}

        diagram_kwargs = dict(
            draw_sequence=True,
            flip_target=flip_target,
            split_at_indels=False,
            label_offsets=label_offsets,
            features_to_show=features_to_show,
            manual_anchors=manual_anchors,
            refs_to_draw=refs_to_draw,
            label_overrides=label_overrides,
            inferred_amplicon_length=inferred_amplicon_length,
            color_overrides=color_overrides,
            feature_heights=feature_heights,
            supplementary_reference_sequences=supplementary_reference_sequences,
            highlight_programmed_substitutions=True,
            invisible_references=invisible_references,
        )

        diagram_kwargs.update(**manual_diagram_kwargs)

        diagram = knock_knock.visualize.architecture.ReadDiagram(als_to_plot,
                                                                 ti,
                                                                 **diagram_kwargs,
                                                                )

        # Note that diagram.alignments may be different than als_to_plot
        # due to application of parsimony.

        # Draw the pegRNAs.
        if draw_pegRNAs and any(al.reference_name in ti.pegRNA_names for al in diagram.alignments):
            ref_ys = {}
            ref_ys['left'] = diagram.max_y + diagram.target_and_donor_y_gap * 0.75
            ref_ys['right'] = ref_ys['left'] + 7 * diagram.gap_between_als

            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = {}

            left_name = self.pegRNA_names_by_side_of_read['left']
            left_visible = (left_name not in invisible_references) and any(al.reference_name == left_name for al in diagram.alignments)

            right_name = self.pegRNA_names_by_side_of_read['right']
            right_visible = (right_name not in invisible_references) and any(al.reference_name == right_name for al in diagram.alignments)

            ref_p_to_xs['left'] = diagram.draw_reference(left_name, ref_ys['left'],
                                                         flip=True,
                                                         label_features=label_pegRNAs,
                                                         visible=left_visible,
                                                        )

            diagram.max_x = max(old_max_x, ref_p_to_xs['left'](0))

            ref_p_to_xs['right'] = diagram.draw_reference(right_name, ref_ys['right'],
                                                          flip=False,
                                                          label_features=label_pegRNAs,
                                                          visible=right_visible,
                                                         )

            diagram.min_x = min(old_min_x, ref_p_to_xs['right'](0))

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            if annotate_overlap and self.manual_anchors and (left_name, 'overlap') in ti.features:
                offset_to_ref_ps = ti.feature_offset_to_ref_p(left_name, 'overlap')
                overlap_xs = sorted([ref_p_to_xs['left'](offset_to_ref_ps[0]), ref_p_to_xs['left'](offset_to_ref_ps[max(offset_to_ref_ps)])])

                overlap_xs = knock_knock.visualize.architecture.adjust_edges(overlap_xs)

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
