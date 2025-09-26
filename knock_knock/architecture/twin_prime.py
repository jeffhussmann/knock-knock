from collections import defaultdict

import numpy as np

from hits import interval, sam
from hits.utilities import memoized_property, memoized_with_kwargs

import knock_knock.architecture
from . import prime_editing
import knock_knock.editing_strategy
import knock_knock.visualize.architecture

from knock_knock.outcome import Details

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

    @memoized_property
    def pegRNA_alignments_by_side_of_read(self):
        als = {}
        for side in ['left', 'right']:
            name = self.pegRNA_names_by_side_of_read[side]
            als[side] = self.pegRNA_alignments_by_pegRNA_name[name]

        return als

    def other_pegRNA_name(self, pegRNA_name):
        side = self.pegRNA_name_to_side_of_read[pegRNA_name]
        other_name = self.pegRNA_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]
        return other_name

    def generate_extended_pegRNA_overlap_alignment(self, pegRNA_al_to_extend):
        # Not currently used, but probably this means test cases didn't capture when it was needed.
        if pegRNA_al_to_extend is None:
            return None

        other_pegRNA_name = self.other_pegRNA_name(pegRNA_al_to_extend.reference_name)
        return self.extend_alignment_from_shared_feature(pegRNA_al_to_extend, 'overlap', other_pegRNA_name, 'overlap')

    def extension_chain_link_specifications(self):
        left_pegRNA_name = self.editing_strategy.pegRNA_names_by_side_of_read['left']
        right_pegRNA_name = self.editing_strategy.pegRNA_names_by_side_of_read['right']

        links = [
            (self.target_alignments, ('first target', 'second target')),
                knock_knock.pegRNAs.make_HA_PBS_name(left_pegRNA_name),
            (self.pegRNA_alignments_by_pegRNA_name[left_pegRNA_name], ('first pegRNA', 'second pegRNA')),
                'overlap',
            (self.pegRNA_alignments_by_pegRNA_name[right_pegRNA_name], ('second pegRNA', 'first pegRNA')),
                knock_knock.pegRNAs.make_HA_PBS_name(right_pegRNA_name),
            (self.target_alignments, ('second target', 'first target')),
        ]

        last_al_to_description = {
            'none': 'no target',
            'first target': 'not RT\'ed',
            'first pegRNA': 'RT\'ed',
            'second pegRNA': 'RT\'ed + overlap-extended',
            'second target': 'RT\'ed + overlap-extended',
        }

        return links, last_al_to_description

    @property
    def reconcile_function(self):
        return min

    @memoized_property
    def extension_chains_by_side(self):
        return self.reconcile_extension_chains(require_definite=True)

    @memoized_property
    def possible_extension_chains_by_side(self):
        return self.reconcile_extension_chains(require_definite=False)

    @memoized_property
    def extension_chain_junction_microhomology(self):
        last_als = {}

        for side in ['left', 'right']:
            chain = self.extension_chains_by_side[side]

            if chain.description in ['not seen', 'no target']:
                last_al = None

            else:
                if chain.description == 'RT\'ed + overlap-extended':
                    if 'second target' in chain.alignments:
                        last_al = chain.alignments['second target']
                    else:
                        last_al = chain.alignments['second pegRNA']

                else:
                    if chain.description == 'not RT\'ed':
                        last_al = chain.alignments['first target']

                    elif chain.description == 'RT\'ed':
                        last_al = chain.alignments['first pegRNA']

            last_als[side] = last_al

        return knock_knock.architecture.junction_microhomology(self.editing_strategy.reference_sequences, last_als['left'], last_als['right'])

    def get_extension_chain_edge(self, side):
        ''' Get the position of the far edge of an extension chain
        in the relevant coordinate system.
        '''
        strat = self.editing_strategy

        this_side_pegRNA_name = self.pegRNA_names_by_side_of_read[side]
        other_side_pegRNA_name = self.other_pegRNA_name(this_side_pegRNA_name)

        PBS_end = strat.features[this_side_pegRNA_name, 'PBS'].end

        chain = self.extension_chains_by_side[side]

        if chain.description in ['not seen', 'no target']:
            relevant_edge = None

        else:
            if chain.description == 'RT\'ed + overlap-extended':
                if 'second target' in chain.alignments:
                    al = chain.alignments['second target']
                else:
                    al = chain.alignments['second pegRNA']

                this_side_overlap = strat.features[this_side_pegRNA_name, 'overlap']
                other_side_overlap = strat.features[other_side_pegRNA_name, 'overlap']

                up_to_overlap_end = PBS_end - this_side_overlap.start + 1

                if al.reference_name == other_side_pegRNA_name:
                    extra_other_pegRNA = (al.reference_end - 1) - (other_side_overlap.end + 1) + 1

                    relevant_edge = up_to_overlap_end + extra_other_pegRNA

                elif al.reference_name == strat.target:
                    opposite_PBS_end = strat.features[other_side_pegRNA_name, 'PBS'].end 
                    up_to_opposite_PBS_end = opposite_PBS_end - (other_side_overlap.end + 1) + 1

                    opposite_target_PBS_name = strat.PBS_names_by_side_of_read[knock_knock.editing_strategy.other_side[side]]
                    opposite_target_PBS = strat.features[strat.target, opposite_target_PBS_name]

                    if opposite_target_PBS.strand == self.sequencing_direction:
                        extra_genomic = (opposite_target_PBS.start - 1) - al.reference_start + 1
                    else:
                        extra_genomic = (al.reference_end - 1) - (opposite_target_PBS.end + 1) + 1

                    relevant_edge = up_to_overlap_end + up_to_opposite_PBS_end + extra_genomic
                    
            else:
                if chain.description == 'not RT\'ed':
                    al = chain.alignments['first target']

                    target_PBS_name = strat.PBS_names_by_side_of_read[side]
                    target_PBS = strat.features[strat.target, target_PBS_name]

                    # Positive values are towards the opposite nick,
                    # negative values are away from the opposite nick.

                    if target_PBS.strand == '+':
                        relevant_edge = (al.reference_end - 1) - target_PBS.end
                    else:
                        relevant_edge = target_PBS.start - al.reference_start

                elif chain.description == 'RT\'ed':
                    al = chain.cropped_last_al

                    relevant_edge = PBS_end - al.reference_start
                
        return relevant_edge

    @memoized_property
    def has_intended_pegRNA_overlap(self):
        chains = self.extension_chains_by_side

        return (
            chains['left'].description == 'RT\'ed + overlap-extended' and
            chains['right'].description == 'RT\'ed + overlap-extended' and 
            chains['left'].query_covered == chains['right'].query_covered
        )

    @memoized_property
    def has_possible_intended_pegRNA_overlap(self):
        chains = self.possible_extension_chains_by_side

        return (
            chains['left'].description == 'RT\'ed + overlap-extended' and
            chains['right'].description == 'RT\'ed + overlap-extended' and 
            chains['left'].query_covered == chains['right'].query_covered
        )

    @memoized_property
    def is_intended_or_partial_replacement(self):
        if self.editing_strategy.pegRNA_programmed_deletion is not None:
            status = False
        else:
            if not (self.has_intended_pegRNA_overlap or self.has_possible_intended_pegRNA_overlap):
                status = False
            else:
                if self.editing_strategy.pegRNA_substitutions is None:
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
            mismatches = knock_knock.architecture.get_mismatch_info(al, self.editing_strategy.reference_sequences)

            programmed_ps = self.editing_strategy.pegRNA_pair.programmed_substitution_ps[al.reference_name]

            for true_read_p, read_b, ref_p, ref_b, q in mismatches:
                
                # ref_p might be outside of edit portion or might be a programmed substitution.
                edit_p = self.editing_strategy.pegRNA_pair.pegRNA_coords_to_edit_coords[al.reference_name].get(ref_p)
                
                if edit_p is not None and ref_p not in programmed_ps:
                    mismatches_seen.add(knock_knock.outcome.Mismatch(edit_p, read_b))
                    
        mismatches = knock_knock.outcome.Mismatches(mismatches_seen)

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

        if self.platform == 'nanopore':
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
            if chains[side].description.startswith('RT\'ed')
        }

        return contains_RTed_sequence

    @memoized_property
    def is_unintended_rejoining(self):
        ''' At least one side has RT'ed sequence, and together the extension
        chains cover the whole read.
        '''
        return self.contains_RTed_sequence and self.uncovered_by_extension_chains.total_length == 0

    @memoized_property
    def integrase_sites_in_chains(self):
        pegRNA_pair = self.editing_strategy.pegRNA_pair
        chains = self.extension_chains_by_side

        edges = {side: self.get_extension_chain_edge(side) for side in ['left', 'right']}

        integrase_sites = []

        chains_are_distinct = chains['left'].query_covered != chains['right'].query_covered

        side_and_strands = [('left', '+')]

        if chains_are_distinct:
            side_and_strands.append(('right', '-'))

        for side, strand in side_and_strands:
            if 'overlap' in chains[side].description:
                relevant_threshold = pegRNA_pair.complete_integrase_site_ends_in_RT_and_overlap_extended_target_sequence[strand]
            elif chains[side].description.startswith('RT\'ed'):
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

        if any('overlap' in chains[side].description for side in ['left', 'right']):
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

        self.subcategory = f'left {chains[possibly_flipped_side["left"]].description}, right {chains[possibly_flipped_side["right"]].description}'

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
        for al in list(chains['left'].alignments.values()) + list(chains['right'].alignments.values()):
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

        self.relevant_alignments = self.extension_chains_by_side['left'].parsimonious_alignments + \
                                   self.extension_chains_by_side['right'].parsimonious_alignments + \
                                   self.extension_chain_gap_covers

    @memoized_property
    def has_any_flipped_pegRNA_al(self):
        return {side for side, als in self.flipped_pegRNA_als.items() if len(als) > 0}

    def convert_target_alignment_edge_to_nick_coordinate(self, al, start_or_end):
        strat = self.editing_strategy
        target_PBS = strat.features[strat.target, strat.PBS_names_by_side_of_read['left']]

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
                                   deletions=[self.editing_strategy.pegRNA_programmed_deletion],
                                   insertions=[],
                                  )
            self.relevant_alignments = self.intended_edit_relevant_alignments

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.single_read_covering_target_alignment:
            self.register_single_read_covering_target_alignment()

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

        self.categorized = True

        return self.category, self.subcategory, self.details, self.Details

    def manual_anchors(self, alignments_to_plot):
        ''' Anchors for drawing knock-knock ref-centric diagrams with overlap in pegRNA aligned.
        '''
        strat = self.editing_strategy

        manual_anchors = {}

        if strat.pegRNA_names is None:
            return manual_anchors

        overlap_feature = strat.features.get((strat.pegRNA_names[0], 'overlap'))
        if overlap_feature is not None:
            overlap_length = len(strat.features[strat.pegRNA_names[0], 'overlap'])

            overlap_offset_to_qs = defaultdict(dict)

            for side, expected_strand in [('left', '-'), ('right', '+')]:
                pegRNA_name = self.pegRNA_names_by_side_of_read[side]
                
                pegRNA_als = [al for al in alignments_to_plot if al.reference_name == pegRNA_name and sam.get_strand(al) == expected_strand]

                if len(pegRNA_als) == 0:
                    continue

                def priority_key(al):
                    is_extension_al = (al == self.extension_chains_by_side['left'].alignments.get('first pegRNA')) or \
                                      (al == self.extension_chains_by_side['right'].alignments.get('first pegRNA'))

                    overlap_length = sam.feature_overlap_length(al, self.editing_strategy.features[pegRNA_name, 'overlap'])
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
                    ref_p = strat.features[pegRNA_name, 'overlap'].offset_to_ref_p[anchor_offset]
                    manual_anchors[pegRNA_name] = (q, ref_p)
                
        return manual_anchors

    def plot(self,
             relevant=True,
             manual_alignments=None,
             annotate_overlap=True,
             label_integrase_features=False,
             draw_pegRNAs=True,
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
        refs_to_draw = manual_diagram_kwargs.pop('refs_to_draw', plot_parameters['refs_to_draw'])
        refs_to_label = manual_diagram_kwargs.pop('refs_to_label', plot_parameters['refs_to_label'])
        refs_to_flip = manual_diagram_kwargs.pop('refs_to_flip', plot_parameters['refs_to_flip'])

        if extra_features_to_show is not None:
            features_to_show.update(extra_features_to_show)

        if extra_label_overrides is not None:
            label_overrides.update(extra_label_overrides)

        if relevant and not self.categorized:
            self.categorize()

        strat = self.editing_strategy

        if label_integrase_features:
            for ref_name, name in strat.integrase_sites:
                if 'right' in name:
                    label_offsets[name] = 1
                    features_to_show.add((ref_name, name))
                if 'left' in name:
                    label_offsets[name] = 2
                    features_to_show.add((ref_name, name))

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

        if 'phiX' in strat.supplemental_indices:
            supplementary_reference_sequences = strat.supplemental_reference_sequences('phiX')
        else:
            supplementary_reference_sequences = {}

        diagram_kwargs = dict(
            draw_sequence=True,
            split_at_indels=False,
            label_offsets=label_offsets,
            features_to_show=features_to_show,
            manual_anchors=manual_anchors,
            refs_to_draw=refs_to_draw,
            refs_to_flip=refs_to_flip,
            refs_to_label=refs_to_label,
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
                                                                 strat,
                                                                 architecture=self,
                                                                 **diagram_kwargs,
                                                                )

        # Note that diagram.alignments may be different than als_to_plot
        # due to application of parsimony.

        # Draw the overlap.

        if all(pegRNA_name in diagram.ref_ys for pegRNA_name in strat.pegRNA_names):
            # To ensure that features on pegRNAs that extend far to the right of
            # the read are plotted, temporarily make the x range very wide.
            old_min_x, old_max_x = diagram.min_x, diagram.max_x

            diagram.min_x = -1000
            diagram.max_x = 1000

            ref_p_to_xs = {}
            ref_ys = {}

            left_name = self.pegRNA_names_by_side_of_read['left']

            right_name = self.pegRNA_names_by_side_of_read['right']

            ref_ys['left'] = diagram.ref_ys[left_name]
            ref_p_to_xs['left'] = diagram.ref_p_to_xs[left_name]

            diagram.max_x = max(old_max_x, ref_p_to_xs['left'](0))

            ref_ys['right'] = diagram.ref_ys[right_name]
            ref_p_to_xs['right'] = diagram.ref_p_to_xs[right_name]

            diagram.min_x = min(old_min_x, ref_p_to_xs['right'](0))

            diagram.ax.set_xlim(diagram.min_x, diagram.max_x)

            if annotate_overlap and self.manual_anchors and (left_name, 'overlap') in strat.features:
                offset_to_ref_ps = strat.features[left_name, 'overlap'].offset_to_ref_p
                overlap_xs = sorted([ref_p_to_xs['left'](offset_to_ref_ps[0]), ref_p_to_xs['left'](offset_to_ref_ps[max(offset_to_ref_ps)])])

                overlap_xs = knock_knock.visualize.architecture.adjust_edges(overlap_xs)

                overlap_color = strat.features[left_name, 'overlap'].attribute['color']
                    
                diagram.ax.fill_betweenx([ref_ys['left'], ref_ys['right'] + diagram.ref_line_width + diagram.feature_line_width],
                                         [overlap_xs[0], overlap_xs[0]],
                                         [overlap_xs[1], overlap_xs[1]],
                                         color=overlap_color,
                                         alpha=0.4,
                                        )

                text_x = np.mean(overlap_xs)
                text_y = ref_ys['right'] - 2 * diagram.ref_line_width

                diagram.ax.annotate('overlap',
                                    xy=(text_x, text_y),
                                    color=overlap_color,
                                    ha='center',
                                    va='top',
                                    size=diagram.font_sizes['feature_label'],
                                    weight='bold',
                                   )

            diagram.update_size()

        return diagram
