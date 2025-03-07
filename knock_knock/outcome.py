import numpy as np
import pandas as pd

import knock_knock.target_info
from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, Mismatches

class Detail:
    def perform_anchor_shift(self, anchor):
        return self

def simple_Detail_factory(convertor):
    class SimpleDetail(Detail):
        def __init__(self, value):
            self.value = value

        @classmethod
        def from_string(cls, details_string):
            value = convertor(details_string)
            return cls(value)

        def __str__(self):
            return str(self.value)

    return SimpleDetail

def DetailList_factory(SpecificDetail):
    INTERMEDIATE_DELIMITER = ';'

    class DetailList:
        def __init__(self, fields):
            self.fields = fields

        @classmethod
        def from_string(cls, details_string):
            fields_strings = details_string.split(INTERMEDIATE_DELIMITER)
            fields = [SpecificDetail.from_string(s) for s in fields_strings]
            return cls(fields)

        def __str__(self):
            return INTERMEDIATE_DELIMITER.join(str(field) for field in self.fields)

    return DetailList

def Details_factory(class_name, field_names_and_types):
    TOP_LEVEL_DELIMITER = '__'

    class Details:
        def __init__(self, *fields):
            self.fields = fields
            for (name, _), field in zip(field_names_and_types, fields):
                setattr(self, name, field)
                
        @classmethod
        def from_string(cls, details_string):
            fields_strings = details_string.split(TOP_LEVEL_DELIMITER)
            fields = [Field.from_string(s) for (_, Field), s in zip(field_names_and_types, fields_strings)]
            return cls(*fields)
        
        def __str__(self):
            return TOP_LEVEL_DELIMITER.join(str(field) for field in self.fields)

        def perform_anchor_shift(self, anchor):
            shifted_fields = [field.perform_anchor_shift(anchor) for field in self.fields]
            return type(self)(*shifted_fields)
            
        def undo_anchor_shift(self, anchor):
            return self.perform_anchor_shift(-anchor)

    Details.__name__ = class_name
    Details.__qualname__ = f'Details_factory.{class_name}'

    return Details

Deletions = DetailList_factory(DegenerateDeletion)
Insertions = DetailList_factory(DegenerateInsertion)

IntDetail = simple_Detail_factory(int)

class AnchoredIntDetail(IntDetail):
    def perform_anchor_shift(self, anchor):
        return type(self)(self.value - anchor)

StrDetail = simple_Detail_factory(str)

StrList = DetailList_factory(StrDetail)

DeletionPlusMismatches = Details_factory('DeletionPlusMismatches', [
    ('deletion', DegenerateDeletion),
    ('mismatches', Mismatches),
]) 

InsertionPlusMismatches = Details_factory('InsertionPlusMismatches', [
    ('insertion', DegenerateInsertion),
    ('mismatches', Mismatches),
]) 

UnintendedRejoining = Details_factory('UnintendedRejoining', [
    ('left_edge', IntDetail),
    ('right_edge', IntDetail),
    ('junction_microhomology_length', IntDetail),
    ('integrase_sites', StrList),
])

ProgrammedEdit = Details_factory('ProgrammedEdit', [
    ('programmed_substitution_read_bases', StrDetail),
    ('non_programmed_target_mismatches_outcome', Mismatches),
    ('non_programmed_edit_mismatches_outcome', Mismatches),
    ('insertions', Insertions),
    ('deletions', Deletions),
])

class Outcome:
    pass

class ProgrammedEditOutcome(Outcome):
    def __init__(self,
                 SNV_read_bases,
                 non_programmed_target_mismatches_outcome,
                 non_programmed_edit_mismatches_outcome,
                 indels,
                ):

        self.SNV_read_bases = SNV_read_bases

        self.non_programmed_target_mismatches_outcome = non_programmed_target_mismatches_outcome
        self.non_programmed_edit_mismatches_outcome = non_programmed_edit_mismatches_outcome

        self.indels = indels
        self.deletions = [indel for indel in self.indels if indel.kind == 'D']
        self.insertions = [indel for indel in self.indels if indel.kind == 'I']

    @classmethod
    def from_string(cls, details_string):
        SNV_string, non_programmed_target_mismatches_string, non_programmed_edit_mismatches_string, indels_string = details_string.split(';', 3)

        non_programmed_target_mismatches_outcome = MismatchOutcome.from_string(non_programmed_target_mismatches_string)
        non_programmed_edit_mismatches_outcome = MismatchOutcome.from_string(non_programmed_edit_mismatches_string)

        if indels_string == '':
            indels = []
        else:
            indels = [knock_knock.target_info.degenerate_indel_from_string(s) for s in indels_string.split(';')]

        return cls(SNV_string,
                   non_programmed_target_mismatches_outcome,
                   non_programmed_edit_mismatches_outcome,
                   indels,
                  )

    def __str__(self):
        indels_string = ';'.join(str(indel) for indel in self.indels)
        return f'{self.SNV_read_bases};{self.non_programmed_target_mismatches_outcome};{self.non_programmed_edit_mismatches_outcome};{indels_string}'

    def perform_anchor_shift(self, anchor):
        # Note: doesn't touch programmed SNVs or edit mismatches.
        shifted_non_programmed_target_mismatches = self.non_programmed_target_mismatches_outcome.perform_anchor_shift(anchor)
        shifted_deletions = [DeletionOutcome(d).perform_anchor_shift(anchor).deletion for d in self.deletions]
        shifted_insertions = [InsertionOutcome(i).perform_anchor_shift(anchor).insertion for i in self.insertions]
        return type(self)(self.SNV_read_bases,
                          shifted_non_programmed_target_mismatches,
                          self.non_programmed_edit_mismatches_outcome,
                          shifted_deletions + shifted_insertions,
                         )

class HDROutcome(Outcome):
    def __init__(self, donor_SNV_read_bases, donor_deletions):
        self.donor_SNV_read_bases = donor_SNV_read_bases
        self.donor_deletions = donor_deletions

    @classmethod
    def from_string(cls, details_string):
        SNV_string, donor_deletions_string = details_string.split(';', 1)
        if donor_deletions_string == '':
            deletions = []
        else:
            deletions = [DegenerateDeletion.from_string(s) for s in donor_deletions_string.split(';')]
        return cls(SNV_string, deletions)

    def __str__(self):
        donor_deletions_string = ';'.join(str(d) for d in self.donor_deletions)
        return f'{self.donor_SNV_read_bases};{donor_deletions_string}'

    #TODO: there is no anchor shifting of donor deletions.

class HDRPlusDeletionOutcome(Outcome):
    def __init__(self, HDR_outcome, deletion_outcome):
        self.HDR_outcome = HDR_outcome
        self.deletion_outcome = deletion_outcome
    
    @classmethod
    def from_string(cls, details_string):
        deletion_string, HDR_string = details_string.split(';', 1)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)
        HDR_outcome = HDROutcome.from_string(HDR_string)

        return cls(HDR_outcome, deletion_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.HDR_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_deletion = self.deletion_outcome.perform_anchor_shift(anchor)
        return type(self)(self.HDR_outcome, shifted_deletion)
        
class DeletionPlusDuplicationOutcome(Outcome):
    def __init__(self, deletion_outcome, duplication_outcome):
        self.deletion_outcome = deletion_outcome
        self.duplication_outcome = duplication_outcome
    
    @classmethod
    def from_string(cls, details_string):
        deletion_string, duplication_string = details_string.split(';', 1)
        duplication_outcome = DuplicationOutcome.from_string(duplication_string)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)

        return cls(deletion_outcome, duplication_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.duplication_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_deletion = self.deletion_outcome.perform_anchor_shift(anchor)
        shifted_duplication = self.deletion_outcome.perform_anchor_shift(anchor)
        return type(self)(shifted_deletion, shifted_duplication)

class MultipleIndelOutcome(Outcome):
    def __init__(self, deletion_outcomes, insertion_outcomes):
        self.deletion_outcomes = deletion_outcomes
        self.insertion_outcomes = insertion_outcomes
    
    @classmethod
    def from_string(cls, details_string):
        indel_strings = details_string.split(';', 1)
        deletion_outcomes = [DeletionOutcome.from_string(s) for s in indel_strings if s.startswith('D')]
        insertion_outcomes = [InsertionOutcome.from_string(s) for s in indel_strings if s.startswith('I')]

        return cls(deletion_outcomes, insertion_outcomes)

    def __str__(self):
        return ';'.join(map(str, self.deletion_outcomes + self.insertion_outcomes))

    def perform_anchor_shift(self, anchor):
        shifted_deletions = [d.perform_anchor_shift(anchor) for d in self.deletion_outcomes]
        shifted_insertions = [i.perform_anchor_shift(anchor) for i in self.insertion_outcomes]
        return type(self)(shifted_deletions, shifted_insertions)

class HDRPlusInsertionOutcome(Outcome):
    def __init__(self, HDR_outcome, insertion_outcome):
        self.HDR_outcome = HDR_outcome
        self.insertion_outcome = insertion_outcome
    
    @classmethod
    def from_string(cls, details_string):
        insertion_string, HDR_string = details_string.split(';', 1)
        insertion_outcome = InsertionOutcome.from_string(insertion_string)
        HDR_outcome = HDROutcome.from_string(HDR_string)

        return cls(HDR_outcome, insertion_outcome)

    def __str__(self):
        return f'{self.insertion_outcome};{self.HDR_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_insertion = self.insertion_outcome.perform_anchor_shift(anchor)
        return type(self)(self.HDR_outcome, shifted_insertion)

class InsertionPlusMismatchOutcome(Outcome):
    def __init__(self, insertion_outcome, mismatch_outcome):
        self.insertion_outcome = insertion_outcome
        self.mismatch_outcome = mismatch_outcome

    @classmethod
    def from_string(cls, details_string):
        insertion_string, mismatch_string = details_string.split(';', 1)
        insertion_outcome = InsertionOutcome.from_string(insertion_string)
        mismatch_outcome = MismatchOutcome.from_string(mismatch_string)
        return cls(insertion_outcome, mismatch_outcome)

    def __str__(self):
        return f'{self.insertion_outcome};{self.mismatch_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_insertion = self.insertion_outcome.perform_anchor_shift(anchor)
        shifted_mismatch = self.mismatch_outcome.perform_anchor_shift(anchor)
        return type(self)(shifted_insertion, shifted_mismatch)

NAN_INT = np.iinfo(np.int64).min

def int_or_nan_from_string(s):
    if s == 'None':
        return NAN_INT
    else:
        return int(s)

class DuplicationOutcome(Outcome):
    def __init__(self, ref_junctions):
        self.ref_junctions = ref_junctions

    @classmethod
    def from_string(cls, details_string):
        fields = details_string.split(';')
        ref_junctions = []
        for field in fields:
            lefts, rights = field.split(',')
            lefts = [int(l) for l in lefts.strip('{}').split('|')]
            rights = [int(r) for r in rights.strip('{}').split('|')]
            ref_junctions.append((lefts, rights))
        return DuplicationOutcome(ref_junctions)

    def __str__(self):
        fields = []
        for lefts, rights in self.ref_junctions:
            lefts_string = '|'.join(map(str, lefts))
            rights_string = '|'.join(map(str, rights))
            field = '{' + lefts_string + '},{' + rights_string + '}'
            fields.append(field)

        return ';'.join(fields)

    def perform_anchor_shift(self, anchor):
        shifted_ref_junctions = [([l - anchor for l in lefts], [r - anchor for r in rights]) for lefts, rights in self.ref_junctions]
        return type(self)(shifted_ref_junctions)

def add_directionalities_to_deletions(outcomes, target_info):
    combined_categories = []

    for category, subcategory, details in outcomes:
        if category == 'deletion':
            deletion = knock_knock.outcome.DeletionOutcome.from_string(details).undo_anchor_shift(target_info.anchor)
            directionality = deletion.classify_directionality(target_info)
            combined_category = f'{category}, {directionality}'
        else:
            combined_category = category
            
        combined_categories.append(combined_category)

    return combined_categories

def extract_deletion_boundaries(target_info,
                                outcome_fractions,
                                include_simple_deletions=True,
                                include_edit_plus_deletions=False,
                               ):

    if isinstance(outcome_fractions, pd.Series):
        outcome_fractions = outcome_fractions.to_frame()

    deletions = [
        (c, s, d) for c, s, d in outcome_fractions.index
        if (include_simple_deletions and c == 'deletion')
        or (include_edit_plus_deletions and (c, s) == ('edit + indel', 'deletion'))
    ]

    deletion_fractions = outcome_fractions.loc[deletions]
    index = np.arange(len(target_info.target_sequence))
    columns = deletion_fractions.columns

    fraction_removed = np.zeros((len(index), len(columns)))
    starts = np.zeros_like(fraction_removed)
    stops = np.zeros_like(fraction_removed)

    for (c, s, d), row in deletion_fractions.iterrows():
        # Undo anchor shift to make coordinates relative to full target sequence.
        if c == 'deletion':
            deletion = DeletionOutcome.from_string(d).undo_anchor_shift(target_info.anchor).deletion
        elif c == 'edit + indel':
            deletions = ProgrammedEditOutcome.from_string(d).undo_anchor_shift(target_info.anchor).deletions
            if len(deletions) != 1:
                raise NotImplementedError
            else:
                deletion = deletions[0]
        else:
            raise ValueError
        
        per_possible_start = row.values / len(deletion.starts_ats)
        
        for start, stop in zip(deletion.starts_ats, deletion.ends_ats):
            deletion_slice = slice(start, stop + 1)

            fraction_removed[deletion_slice] += per_possible_start
            starts[start] += per_possible_start
            stops[stop] += per_possible_start

    fraction_removed = pd.DataFrame(fraction_removed, index=index, columns=columns)
    starts = pd.DataFrame(starts, index=index, columns=columns)
    stops = pd.DataFrame(stops, index=index, columns=columns)

    deletion_boundaries = {
        'fraction_removed': fraction_removed,
        'starts': starts,
        'stops': stops,
    }

    return deletion_boundaries