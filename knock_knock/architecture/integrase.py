import copy
import itertools
from collections import defaultdict

import hits.interval
import hits.utilities

from hits import sam, interval

from . import twin_prime
from knock_knock.outcome import *

memoized_property = hits.utilities.memoized_property

additional_categories = [
    (
        'intended integration',
        (
            'single',
            'multiple',
        ),
    ),
    (
        'unintended integration',
        (
            'messy',
            'internal priming',
            'contains genomic sequence',
        ),
    ),
    (
        'malformed',
        (
            'missing primer(s)',
            'likely concatamer',
        ),
    ),
]

original_order =  twin_prime.Architecture.category_order

class Architecture(twin_prime.Architecture):
    category_order = original_order[:3] + additional_categories + original_order[3:]

    non_relevant_categories = [
        'phiX',
        'nonspecific amplification',
        'minimal alignment to intended target',
        'malformed',
    ]

    @property
    def inferred_amplicon_length(self):
        if self.is_malformed:
            inferred_length = len(self.seq)

        else:
            
            offset_to_q = {
                strand: self.feature_offset_to_q(self.longest_primer_alignments[strand], self.target_info.primers_by_strand[strand].ID)
                for strand in ['+', '-']
            }

            inner_edge_offsets = {strand: max(d) for strand, d in offset_to_q.items()}
            inner_edge_qs = {strand: offset_to_q[strand][offset] for strand, offset in inner_edge_offsets.items()}
            
            # inner_edge_qs are last positions in the primer. + 1 to get length of interval that includes them, then - 2 to exclude them 
            length_seen_between_primers = abs(inner_edge_qs['-'] - inner_edge_qs['+']) + 1 - 2

            inferred_length = length_seen_between_primers + len(self.target_info.primers_by_strand['+']) + len(self.target_info.primers_by_strand['-'])

        return inferred_length

    @property
    def old_inferred_amplicon_length(self):
        if self.primer_alignments['left'] is not None:
            left = interval.get_covered(self.primer_alignments['left']).start
        else:
            left = 0

        if self.primer_alignments['right'] is not None:
            right = interval.get_covered(self.primer_alignments['right']).end
        else:
            right = self.whole_read.end

        inferred_amplicon_length = right - left + 1

        if inferred_amplicon_length < 0:
            inferred_amplicon_length = len(self.seq)

        return inferred_amplicon_length
            
    @memoized_property
    def is_low_quality(self):
        return False

    def overlaps_primer(self, al, strand):
        primer = self.target_info.primers_by_strand[strand]
        overlaps = sam.overlaps_feature(al, primer, require_same_strand=False)

        return overlaps

    @memoized_property
    def long_primer_alignments(self):
        by_strand = {
            strand: [al for al in self.target_alignments if self.overlaps_primer(al, strand) and al.query_alignment_length > 50]
            for strand in ['+', '-']
        }
        
        return by_strand

    @memoized_property
    def longest_primer_alignments(self):
        by_strand = {
            strand: max(als, key=lambda al: al.query_alignment_length, default=None)
            for strand, als in self.long_primer_alignments
        }
        
        return by_strand

    @memoized_property
    def longest_primer_alignments(self):
        by_strand = {
            strand: max([al for al in self.target_alignments if self.overlaps_primer(al, strand)], key=lambda al: al.query_alignment_length, default=None)
            for strand in ['+', '-']
        }
        
        return by_strand

    @memoized_property
    def sequencing_direction(self):
        strands = set(sam.get_strand(al) for al in self.longest_primer_alignments.values() if al is not None)
        
        if len(strands) == 1:    
            strand = list(strands)[0]
        else:
            strand = None
            
        return strand

    @property
    def flipped(self):
        return self.sequencing_direction == '-'

    @memoized_property
    def target_edge_alignments(self):
        if self.sequencing_direction == '-':
            side_to_strand = {
                'left': '-',
                'right': '+',
            }
        else:
            side_to_strand = {
                'left': '+',
                'right': '-',
            }
            
        target_edge_als = {
            side: self.longest_primer_alignments[side_to_strand[side]]
            for side in ['left', 'right']
        }

        return target_edge_als

    @memoized_property
    def between_primers(self):
        if self.primer_alignments['left'] is not None:
            start = interval.get_covered(self.primer_alignments['left']).start
        else:
            start = self.whole_read.start

        if self.primer_alignments['right'] is not None:
            end = interval.get_covered(self.primer_alignments['right']).end
        else:
            end = self.whole_read.end

        return interval.Interval(start, end)

    def overlaps_outside_primers(self, al):
        return (hits.interval.get_covered(al) - self.between_primers).total_length > 0

    @memoized_property
    def starts_at_expected_location(self):
        edge_al = self.target_edge_alignments['left']
        return edge_al is not None

    @memoized_property
    def integrase_sites(self):
        all_sites = self.target_info.integrase_sites

        by_ref = defaultdict(lambda: defaultdict(list))

        for (ref_name, feature_name), feature in all_sites.items():
            component = feature.attribute['component']
            by_ref[ref_name][component].append(feature)

        return by_ref

    @memoized_property
    def compatible_pairs(self):

        def is_compatible_pair(pair):
            first, second = pair

            distinct = (first != second)
            same_CD = (first.attribute['CD'] == second.attribute['CD'])
            opposite_sites = ({f.parent.attribute['site'] for f in [first, second]} == {'attP', 'attB'})

            return distinct and same_CD and opposite_sites

        compatible_pairs = defaultdict(list)

        for first_name, second_name in itertools.product(self.integrase_sites, repeat=2):
            pairs = itertools.product(self.integrase_sites[first_name]['CD'], self.integrase_sites[second_name]['CD'])
            compatible_pairs[first_name, second_name] = [pair for pair in pairs if is_compatible_pair(pair)]

        return compatible_pairs

    @memoized_property
    def donor_DISC_CD_pairs(self):
        pairs = []

        for first, second in itertools.combinations(self.integrase_sites[self.target_info.donor]['CD'], 2):
            if first.attribute['CD'] == second.attribute['CD']:
                if {f.parent.attribute['site'] for f in [first, second]} == {'attP', 'attB'}:
                    pairs.append((first, second))

        return pairs

    @memoized_property
    def target_donor_CD_pairs(self):
        pairs = []

        for target_CD in self.integrase_sites[self.target_info.target]['CD']:
            for donor_CD in self.integrase_sites[self.target_info.donor]['CD']:
                if target_CD.attribute['CD'] == donor_CD.attribute['CD']:
                    if {f.parent.attribute['site'] for f in [target_CD, donor_CD]} == {'attP', 'attB'}:
                        pairs.append((target_CD, donor_CD))

        return pairs

    def find_alignment_extending_from_CD(self, first_al, second_al_source, from_side):
        if second_al_source == 'target':
            possible_second_als = {self.target_info.target: self.target_alignments}
        elif second_al_source == 'donor':
            possible_second_als = {self.target_info.donor: self.donor_alignments}
        elif second_al_source == 'pegRNAs':
            possible_second_als = self.pegRNA_alignments_by_pegRNA_name

        extension_als = []

        for second_name, second_als in possible_second_als.items():
            for second_al in second_als:
                for first_feature, second_feature in self.compatible_pairs[first_al.reference_name, second_name]:
                    if from_side == 'left':
                        left_al = first_al
                        right_al = second_al

                        left_feature = first_feature
                        right_feature = second_feature

                    elif from_side == 'right':
                        left_al = second_al
                        right_al = first_al

                        left_feature = second_feature
                        right_feature = first_feature
                    
                    else:
                        raise ValueError

                    extension_results = self.are_mutually_extending_from_shared_feature(left_al,
                                                                                        left_feature.ID,
                                                                                        right_al,
                                                                                        right_feature.ID,
                                                                                       )

                    if extension_results['status'] == 'definite':
                        extension_als.append(second_al)
                    
        return max(extension_als, key=lambda al: al.query_alignment_length, default=None)

    @memoized_property
    def gap_covering_alignments(self):
        return []

    @memoized_property
    def partial_gap_perfect_alignments(self):
        return []

    def realign_edges_to_primers(self, read_side):
        return None

    @memoized_property
    def perfect_right_edge_alignment(self):
        return None

    def manual_anchors(self, alignments_to_plot):
        return {}

    def interesting_and_uninteresting_indels(self, als):
        # For now, ignore most indels.

        interesting = []
        uninteresting = []

        indels = self.extract_indels_from_alignments(als)

        for indel, near_cut in indels:
            if indel.kind == 'D' and indel.length > 20:
                interesting.append(indel)

        return interesting, uninteresting

    @memoized_property
    def donor_extension_chains_by_side(self):
        if len(self.target_info.pegRNA_names) == 0:
            return self.landing_pad_donor_extension_chains_by_side
        else:
            return self.non_landing_pad_donor_extension_chains_by_side
    
    @memoized_property
    def landing_pad_donor_extension_chains_by_side(self):

        chains = {}

        for side in ['left', 'right']:

            chain = {
                'alignments': {},
                'description': 'no target',
            }
    
            target_edge_al = self.target_edge_alignments[side]

            
            if target_edge_al is not None:
                nts_past_primer = (interval.get_covered(target_edge_al) - interval.get_covered(self.primer_alignments[side])).total_length
                
                if nts_past_primer > 10:
                    chain['description'] = 'no integration'

                chain['alignments']['first target'] = target_edge_al
            
                first_donor_al = self.find_alignment_extending_from_CD(target_edge_al, 'donor', side)
                
                if first_donor_al is not None:
                    chain['description'] = 'one-sided integration'
                    chain['alignments']['first donor'] = first_donor_al

                    second_donor_al = self.find_alignment_extending_from_CD(first_donor_al, 'donor', side)

                    if second_donor_al is not None:
                        chain['alignments']['second donor'] = second_donor_al
                        last_donor_al = second_donor_al
                    else:
                        last_donor_al = first_donor_al

                    second_target_al = self.find_alignment_extending_from_CD(last_donor_al, 'target', side)

                    if second_target_al is not None:
                        chain['description'] = 'integration'
                        chain['alignments']['second target'] = second_target_al

                            
            al_order = [
                'first target',
                'first donor',
                'second donor',
                'second target',
            ]

            query_covered = interval.get_disjoint_covered([])
            query_covered_incremental = {'none': query_covered}

            for al_order_i, al_key in enumerate(al_order):
                if al_key in chain['alignments']:
                    als_up_to = [chain['alignments'][key] for key in al_order[:al_order_i + 1] if key in chain['alignments']]
                    query_covered = interval.get_disjoint_covered(als_up_to)
                    query_covered_incremental[al_key] = query_covered

            chain.update({
                'query_covered': query_covered,
                'query_covered_incremental': query_covered_incremental,
            })

            chains[side] = chain

        return chains

    @memoized_property
    def non_landing_pad_donor_extension_chains_by_side(self):
        chains = copy.deepcopy(self.extension_chains_by_side)

        for side in ['left', 'right']:
            if chains[side]['description'] == "RT'ed":
                first_pegRNA_al = chains[side]['alignments']['first pegRNA']
                first_donor_al = self.find_alignment_extending_from_CD(first_pegRNA_al, 'donor', side)

                if first_donor_al is not None:
                    chains[side]['description'] = 'one-sided integration'

                    chains[side]['alignments']['first donor'] = first_donor_al

                    second_donor_al = self.find_alignment_extending_from_CD(first_donor_al, 'donor', side)

                    if second_donor_al is not None:
                        chains[side]['alignments']['second donor'] = second_donor_al
                        last_donor_al = second_donor_al
                    else:
                        last_donor_al = first_donor_al

                    second_pegRNA_al = self.find_alignment_extending_from_CD(last_donor_al, 'pegRNAs', side)

                    if second_pegRNA_al is not None:
                        chains[side]['description'] = 'integration'
                        chains[side]['alignments']['second pegRNA'] = second_pegRNA_al

                        overlap_extended_target_al, _, _ = self.find_target_alignment_extending_pegRNA_alignment(second_pegRNA_al, 'PBS')
                    
                        if overlap_extended_target_al is not None:
                            chains[side]['alignments']['second target'] = overlap_extended_target_al

            if 'first donor' in chains[side]['alignments']:
                al_order = [
                    'first target',
                    'first pegRNA',
                    'first donor',
                    'second donor',
                    'second pegRNA',
                    'second target',
                ]

                query_covered = interval.get_disjoint_covered([])
                query_covered_incremental = {'none': query_covered}

                for al_order_i, al_key in enumerate(al_order):
                    if al_key in chains[side]['alignments']:
                        als_up_to = [chains[side]['alignments'][key] for key in al_order[:al_order_i + 1] if key in chains[side]['alignments']]
                        query_covered = interval.get_disjoint_covered(als_up_to)
                        query_covered_incremental[al_key] = query_covered

                chains[side].update({
                    'query_covered': query_covered,
                    'query_covered_incremental': query_covered_incremental,
                })

        return chains

    @memoized_property
    def is_intended_integration(self):
        chains = self.donor_extension_chains_by_side
        # This needs more.
        return chains['left']['description'] == 'integration' and chains['right']['description'] == 'integration'

    @memoized_property
    def nonredundant_donor_alignments(self):
        parsimonious_alignments = interval.make_parsimonious(self.alignments)
        return [al for al in parsimonious_alignments if al.reference_name == self.target_info.donor]

    @memoized_property
    def is_unintended_integration(self):
        return (not self.is_intended_integration) and (self.nonredundant_donor_alignments or self.nonredundant_supplemental_alignments)

    @memoized_property
    def rejoining_edges(self):
        gap = self.not_covered_by_primers - hits.interval.get_disjoint_covered(self.target_edge_alignments_list)

        overlap_gap = [al for al in self.nonredundant_donor_alignments + self.nonredundant_supplemental_alignments if (hits.interval.get_covered(al) & gap).total_length  > 0]

        farthest = {
            'left': min(overlap_gap, key=lambda al: hits.interval.get_covered(al).start, default=None),
            'right': max(overlap_gap, key=lambda al: hits.interval.get_covered(al).end, default=None),
        }
        
        rejoining_edges = {}
        
        if self.sequencing_direction == '+':
            side_to_side = {'left': 'left', 'right': 'right'} 
            strand_to_sign = {'+': 1, '-': -1}
        else:
            side_to_side = {'left': 'right', 'right': 'left'} 
            strand_to_sign = {'+': -1, '-': 1}
            
        side_to_key = {'left': 5, 'right': 3}
        
        for read_side, al in farthest.items():
            if al is None or al.reference_name != self.target_info.donor:
                if self.target_info.donor_sequence is None:
                    edge = 0
                else:
                    edge = len(self.target_info.donor_sequence)

                sign = 1
                
            else:
                if read_side == 'left':
                    left_al, right_al = self.target_edge_alignments['left'], al
                else:
                    left_al, right_al = al, self.target_edge_alignments['right']

                cropped_als = hits.sam.crop_to_best_switch_point(left_al, right_al, self.target_info.reference_sequences)

                if read_side == 'left':
                    relevant_al = cropped_als['right']
                else:
                    relevant_al = cropped_als['left']

                edge = hits.sam.reference_edges(relevant_al)[side_to_key[read_side]]
                
                sign = strand_to_sign[hits.sam.get_strand(al)]

            rejoining_edges[side_to_side[read_side]] = edge * sign
            
        return rejoining_edges

    def register_intended_integration(self):
        chains = self.donor_extension_chains_by_side

        self.category = 'intended integration'

        if chains['left']['query_covered'] == chains['right']['query_covered']:
            self.subcategory = 'single'
        else:
            self.subcategory = 'multiple'

        if len(self.target_info.pegRNA_names) > 0:
            als = [al for side, chain in self.donor_extension_chains_by_side.items() for al in chain['alignments'].values()]
        else:
            als = interval.make_parsimonious(self.target_alignments + self.donor_alignments)

        self.relevant_alignments = sam.make_nonredundant(als)

        self.Details = Details(
            left_rejoining_edge=self.rejoining_edges['left'],
            right_rejoining_edge=self.rejoining_edges['right'],
        )

    def register_unintended_integration(self):
        chains = self.donor_extension_chains_by_side

        self.category = 'unintended integration'

        if 'no target' in {chains['left']['description'], chains['right']['description']}:
            self.subcategory = 'internal priming'
        elif self.nonredundant_supplemental_alignments:
            self.subcategory = 'contains genomic sequence'
        else:
            self.subcategory = 'messy'

        if len(self.target_info.pegRNA_names) > 0:
            als = [al for side, chain in self.donor_extension_chains_by_side.items() for al in chain['alignments'].values()]
        else:
            als = interval.make_parsimonious(self.target_alignments + self.donor_alignments + self.supplemental_alignments)

        self.relevant_alignments = sam.make_nonredundant(als)

        self.Details = Details(
            left_rejoining_edge=self.rejoining_edges['left'],
            right_rejoining_edge=self.rejoining_edges['right'],
        )

    def plot_parameters(self):
        plot_parameters = super().plot_parameters()

        to_remove = {
            (ref_name, feature_name)
            for ref_name, feature_name in plot_parameters['features_to_show']
            if ref_name in self.target_info.pegRNA_names
        }

        for k in to_remove:
            plot_parameters['features_to_show'].remove(k)

        colors = {
            'attB': 'tab:cyan',
            'attP': 'tab:olive',
        }

        for (ref_name, feature_name), feature in self.target_info.integrase_sites.items():
            if feature.attribute['component'] == 'complete_site':
                plot_parameters['features_to_show'].add((ref_name, feature_name))
                plot_parameters['color_overrides'][ref_name, feature_name] = hits.visualize.apply_alpha(feature.attribute['color'], 0.7)
                plot_parameters['label_overrides'][feature_name] = f'{feature.attribute["site"]}-{feature.attribute["CD"]}'

            if feature.attribute['component'] == 'CD':
                plot_parameters['features_to_show'].add((ref_name, feature_name))
                plot_parameters['color_overrides'][ref_name, feature_name] = colors[feature.parent.attribute['site']]
                plot_parameters['label_overrides'][feature_name] = None

        for (ref_name, feature_name), feature in self.target_info.features.items():
            if '5\'-ITR' in feature_name:
                plot_parameters['features_to_show'].add((ref_name, feature_name))
                plot_parameters['label_overrides'][feature_name] = '5\' ITR'

            if '3\'-ITR' in feature_name:
                plot_parameters['features_to_show'].add((ref_name, feature_name))
                plot_parameters['label_overrides'][feature_name] = '3\' ITR'

        for feature_name, new_name in zip(sorted(self.target_info.primers), ['forward primer', 'reverse primer']):
            plot_parameters['label_overrides'][feature_name] = new_name

        return plot_parameters

    @memoized_property
    def uncategorized_relevant_alignments(self):
        relevant_alignments = self.target_edge_alignments_list + [
            al for al in self.target_alignments + sam.make_noncontained(self.non_protospacer_pegRNA_alignments) + sam.make_noncontained(self.donor_alignments)
            if not self.overlaps_outside_primers(al)
        ] + self.extra_alignments + self.nonredundant_supplemental_alignments

        relevant_alignments = sam.make_nonredundant(relevant_alignments)

        return relevant_alignments

    @memoized_property
    def is_malformed(self):
        if any(len(als) == 0 for side, als in self.long_primer_alignments.items()):
            malformed = 'missing primer(s)'

        elif any(len(als) > 1 for side, als in self.long_primer_alignments.items()):
            malformed = 'likely concatamer'

        elif (self.whole_read - self.between_primers).total_length > 500:
            malformed = 'likely concatamer'

        else:
            malformed = False

        return malformed

    def categorize(self):
        if self.is_malformed:
            self.category = 'malformed'
            self.subcategory = self.is_malformed

        elif self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.aligns_to_phiX:
            self.category = 'phiX'
            self.subcategory = 'phiX'

            self.relevant_alignments = [self.longest_phiX_alignment]

        elif self.no_alignments_detected:
            self.register_uncategorized()

        elif self.is_intended_or_partial_replacement:
            self.register_intended_replacement()

        elif self.is_intended_integration:
            self.register_intended_integration()

        elif self.is_unintended_integration:
            self.register_unintended_integration()

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()

        elif self.single_read_covering_target_alignment:
            self.register_single_read_covering_target_alignment()

        elif self.inversion:
            self.category = 'inversion'
            self.subcategory = 'inversion'
            self.details = 'n/a'

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

        else:
            self.register_uncategorized()

        self.details = str(self.Details)

        self.categorized = True

        return self.category, self.subcategory, self.details, self.Details

    def plot(self, **kwargs):
        diagram = super().plot(alignment_registration='centered on primers',
                               split_at_indels=False,
                               draw_sequence=False,
                               high_resolution_parallelograms=False,
                               annotate_overlap=False,
                               label_pegRNAs=True,
                               draw_pegRNAs=False,
                               label_features_on_alignments=False,
                               architecture_mode='nanopore',
                               flip_donor=(self.sequencing_direction == '-'),
                               refs_to_draw={
                                   self.target_info.target,
                                   #self.target_info.donor,
                                },
                               **kwargs,
                              )

        return diagram
