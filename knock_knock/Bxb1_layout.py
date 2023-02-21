import hits.utilities

from hits import sam, interval

import knock_knock.layout
import knock_knock.twin_prime_layout
from knock_knock.outcome import *

memoized_property = hits.utilities.memoized_property

class Layout(knock_knock.twin_prime_layout.Layout):
    category_order = [
        ('wild type',
            ('clean',
             'mismatches',
             'short indel far from cut',
            ),
        ),
        ('intended prime edit',
            ('replacement',
             'partial replacement',
            ),
        ),
        ('intended integration',
            ('n/a',
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
        ('integration at unintended prime edit',
            ('n/a',
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
        ('extension from intended annealing',
            ('n/a',
            ),
        ),
        ('uncategorized',
            ('uncategorized',
            ),
        ),
        ('genomic insertion',
            ('hg19',
             'hg38',
             'bosTau7',
             'e_coli',
            ),
        ),
        ('unintended donor sequence',
            ('simple',
             'complex',
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
    ]

    @memoized_property
    def donor_alignments(self):
        ''' Donor meaning integrase donor '''
        if self.target_info.donor is not None:
            valid_names = [self.target_info.donor]
        else:
            valid_names = []

        d_als = [
            al for al in self.alignments
            if al.reference_name in valid_names
        ]

        split_d_als = []
        for al in d_als:
            split_d_als.extend(knock_knock.layout.split_at_edit_clusters(al, self.target_info.reference_sequences))
        
        return split_d_als

    @memoized_property
    def pegRNA_integrase_sites(self):
        all_sites = self.target_info.integrase_sites
        by_pegRNA = {}
        for pegRNA_name in self.target_info.pegRNA_names:
            by_pegRNA[pegRNA_name] = {}

            for (ref_name, feature_name), feature in all_sites.items():
                if ref_name == pegRNA_name:
                    by_pegRNA[pegRNA_name][feature_name.split('_')[-1]] = feature

        return by_pegRNA

    @memoized_property
    def donor_integrase_sites(self):
        all_sites = self.target_info.integrase_sites
        return {feature_name.split('_')[-1]: feature for (ref_name, feature_name), feature in all_sites.items()
                if ref_name == self.target_info.donor}

    @memoized_property
    def pegRNA_integrase_sites_by_side(self):
        by_side = {
            side: self.pegRNA_integrase_sites[name]
            for side, name in self.target_info.pegRNA_names_by_side_of_read.items()
        }
        return by_side

    @memoized_property
    def intended_integrations(self):
        # Idea: look for a pegRNA al that cleanly hands off to a donor alignment.
        
        integrations = []

        pegRNA_als = sam.make_noncontained(self.pegRNA_alignments_by_side_of_read['left'] + self.pegRNA_alignments_by_side_of_read['right'])
        
        for pegRNA_al in pegRNA_als:
            side = self.target_info.pegRNA_name_to_side_of_read[pegRNA_al.reference_name]

            for donor_al in self.donor_alignments:
                shares_CD = self.are_mutually_extending_from_shared_feature(pegRNA_al,
                                                                            self.pegRNA_integrase_sites_by_side[side]['CD'].ID,
                                                                            donor_al,
                                                                            self.donor_integrase_sites['CD'].ID,
                                                                           )

                
                if shares_CD:
                    details = {
                        'pegRNA_al': pegRNA_al,
                        'donor_al': donor_al,
                    }
                    integrations.append(details)
                    
        return integrations

    @memoized_property
    def nonspecific_amplification_of_donor(self):
        covering_donor_als = []

        if self.primer_alignments['left'] is not None and self.donor_alignments:
            covered = interval.get_disjoint_covered(self.donor_alignments)
            if (self.not_covered_by_primers - covered).total_length == 0:
                covering_donor_als = self.donor_alignments

        return covering_donor_als

    @memoized_property
    def is_intended_integration(self):
        return len(self.intended_integrations) > 0

    def register_intended_integration(self):
        # For now, hard-code in the assumption that pegRNA extension al
        # is on the left side and that this will include the central dinucleotide.

        if len(self.intended_integrations) > 1:
            self.category = 'integration at unintended prime edit'
        else:
            details = self.intended_integrations[0]
            reaches_read_end = self.not_covered_by_primers.end in interval.get_covered(details['donor_al'])
            from_left_edge_al = sam.fingerprint(details['pegRNA_al']) == sam.fingerprint(self.pegRNA_extension_als['left'])

            if reaches_read_end and from_left_edge_al:
                self.category = 'intended integration'
            else:
                self.category = 'integration at unintended prime edit'

        if self.category == 'intended integration':
            self.relevant_alignments = (
                self.target_edge_alignments_list + \
                self.pegRNA_extension_als_list + \
                [self.intended_integrations[0]['donor_al']]
            )
        else:
            #self.relevant_alignments = self.uncategorized_relevant_alignments
            self.relevant_alignments = (
                self.target_edge_alignments_list + \
                [self.intended_integrations[0]['donor_al']]
            )
            pegRNA_als = []
            for pegRNA_name, als in self.pegRNA_alignments.items():
                for al in als:
                    if not self.is_pegRNA_protospacer_alignment(al):
                        pegRNA_als.append(al)

            pegRNA_als = sam.make_noncontained(pegRNA_als)
            self.relevant_alignments.extend(pegRNA_als)

        self.subcategory = 'n/a'
        self.outcome = Outcome('n/a')

    @memoized_property
    def alignments_to_search_for_gaps(self):
        return self.split_target_and_pegRNA_alignments + self.donor_alignments

    def categorize(self):
        self.outcome = None

        if self.nonspecific_amplification:
            self.register_nonspecific_amplification()

        elif self.no_alignments_detected:
            self.category = 'uncategorized'
            self.subcategory = 'uncategorized'
            self.details = 'n/a'
            self.outcome = None

        elif self.single_read_covering_target_alignment:
            target_alignment = self.single_read_covering_target_alignment
            interesting_indels, uninteresting_indels = self.interesting_and_uninteresting_indels([target_alignment])

            if len(interesting_indels) == 0:
                if self.starts_at_expected_location:
                    # Need to check in case the intended replacements only involves minimal changes. 
                    if self.is_intended_replacement:
                        self.category = 'intended prime edit'
                        self.subcategory = self.is_intended_replacement
                        self.outcome = Outcome('n/a')
                        self.relevant_alignments = self.target_edge_alignments_list + self.possible_pegRNA_extension_als_list

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

                    self.details = str(self.outcome)
                    self.relevant_alignments = [target_alignment]

            elif len(interesting_indels) == 1:
                indel = interesting_indels[0]

                if len(self.non_pegRNA_SNVs) > 0:
                    self.subcategory = 'mismatches'
                else:
                    self.subcategory = 'clean'

                if indel.kind == 'D':
                    if indel == self.target_info.pegRNA_intended_deletion:
                        self.category = 'intended prime edit'
                        self.subcategory = 'deletion'
                        self.relevant_alignments = [target_alignment] + self.possible_pegRNA_extension_als_list

                    else:
                        self.category = 'deletion'
                        self.relevant_alignments = [target_alignment]

                    self.outcome = DeletionOutcome(indel)
                    self.details = str(self.outcome)

                elif indel.kind == 'I':
                    self.category = 'insertion'
                    self.outcome = InsertionOutcome(indel)
                    self.details = str(self.outcome)
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
            self.details = str(self.outcome)
            self.relevant_alignments = merged_als

        elif self.is_intended_replacement:
            self.category = 'intended prime edit'
            self.subcategory = self.is_intended_replacement
            self.outcome = Outcome('n/a')
            self.relevant_alignments = self.parsimonious_target_alignments + self.pegRNA_extension_als_list

        elif self.is_intended_integration:
            self.register_intended_integration()

        elif self.is_unintended_rejoining:
            self.register_unintended_rejoining()
            
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

        elif self.nonspecific_amplification_of_donor:
            donor_als = self.nonspecific_amplification_of_donor
            self.category = 'nonspecific amplification'
            self.subcategory = 'plasmid'
            self.details = 'n/a'

            self.relevant_alignments = self.target_edge_alignments_list + donor_als + self.extra_alignments

        elif self.genomic_insertion:
            self.register_genomic_insertion()

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

    def plot(self, **kwargs):
        ti = self.target_info

        features_to_show = {
            (ti.donor, 'reverse_primer'),
            (ti.donor, 'AmpR'),
            (ti.donor, 'ori'),
        }

        label_overrides = {}
        label_offsets = {}

        if ti.integrase_sites:
            suffixes = [
                'attP_left',
                'attP_right',
                'attB_left',
                'attB_right',
            ]

            for _, name in ti.integrase_sites:
                for suffix in suffixes:
                    if name.endswith(suffix):
                        label_overrides[name] = '\n'.join(suffix.split('_'))
            
            label_offsets['RTT'] = 2

            for ref_name, name in ti.integrase_sites:
                if 'left' in name or 'right' in name:
                    features_to_show.add((ref_name, name))

        refs_to_draw= {ti.target, ti.donor, *ti.pegRNA_names}

        if 'features_to_show' in kwargs:
            features_to_show.update(kwargs.pop('features_to_show'))

        diagram = super().plot(features_to_show=features_to_show,
                               label_overrides=label_overrides,
                               label_offsets=label_offsets,
                               refs_to_draw=refs_to_draw,
                               donor_below=True,
                               **kwargs,
                              )

        return diagram
