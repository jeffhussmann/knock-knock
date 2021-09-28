import numpy as np

import hits.sam as sam
from hits.utilities import memoized_property
from knock_knock.pacbio_experiment import PacbioExperiment
from knock_knock import target_info, layout

class LengthBiasExperiment(PacbioExperiment):
    @memoized_property    
    def specialized_target_infos(self):
        tis = {}
        for seq_name in self.target_info.reference_sequences:
            ti = target_info.TargetInfo(self.base_dir,
                                        self.target_name,
                                        donor=self.donor,
                                        nonhomologous_donor=self.nonhomologous_donor,
                                        sgRNA=self.sgRNA,
                                        primer_names=self.primer_names,
                                        sequencing_start_feature_name=self.sequencing_start_feature_name,
                                        supplemental_indices=self.supplemental_indices,
                                        infer_homology_arms=self.infer_homology_arms,
                                        target=seq_name,
                                    )
            tis[seq_name] = ti

        return tis

    @memoized_property
    def max_relevant_length(self):
        auto_length = 0
        for seq_name, ti in self.specialized_target_infos.items():
            auto_length = max(auto_length, int((ti.amplicon_length * 1.2 // 1000 + 1)) * 1000)

        return self.description.get('max_relevant_length', auto_length)

    @memoized_property
    def expected_lengths(self):
        expected_lengths = {
            seq_name: ti.amplicon_length
            for seq_name, ti in self.specialized_target_infos.items()
        }

        return expected_lengths

    def process(self, stage):
        try:
            if stage == 'preprocess':
                pass
            elif stage == 'align':
                for read_type in self.read_types:
                    self.generate_alignments(read_type=read_type)
                    self.generate_supplemental_alignments_with_minimap2(read_type=read_type)
                    self.combine_alignments(read_type=read_type)

            elif stage == 'categorize':
                self.categorize_outcomes(read_type='CCS')

                self.generate_outcome_counts()
                self.generate_read_lengths()

                self.record_sanitized_category_names()

            elif stage == 'visualize':
                pass

        except:
            print(self.group, self.sample_name)
            raise

    @memoized_property
    def categorizer(exp_self):
        class MultipleAmpliconLayout(layout.Categorizer):
            category_order = [
                ('WT',
                    tuple(sorted(exp_self.specialized_target_infos)),
                ),
                ('unknown',
                    ('unknown',
                    ),
                ),
            ]

            def __init__(self, alignments, target_info, mode='pacbio'):
                self.mode = mode

                self.target_info = target_info
                self.alignments = alignments

                # TODO: these should all be done in parent class constructor.
                alignment = alignments[0]
                self.name = alignment.query_name
                self.query_name = self.name
                self.seq = sam.get_original_seq(alignment)
                self.qual = np.array(sam.get_original_qual(alignment))

                if self.seq is None:
                    length = 0
                else:
                    length = len(self.seq)

                self.inferred_amplicon_length = length

                self.specialized_categorizer = super(PacbioExperiment, exp_self).categorizer

            def categorize(self):
                recognized_amplicons = set()

                for seq_name, ti in exp_self.specialized_target_infos.items():
                    layout = self.specialized_categorizer(self.alignments, ti, mode='pacbio')
                    layout.categorize()
                    if layout.category == 'WT':
                        recognized_amplicons.add(seq_name)

                if len(recognized_amplicons) == 1:
                    self.category = 'WT'
                    self.subcategory = recognized_amplicons.pop()
                    self.details = 'n/a'
                else:
                    self.category = 'unknown'
                    self.subcategory = 'unknown'
                    self.details = 'n/a'

                return self.category, self.subcategory, self.details

        return MultipleAmpliconLayout