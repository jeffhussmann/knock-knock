import functools

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

        __repr__ = __str__

    return SimpleDetail

def DetailList_factory(SpecificDetail):
    INTERMEDIATE_DELIMITER = '__'

    class DetailList(Detail):
        def __init__(self, fields):
            self.fields = []

            for field in fields:
                if not isinstance(field, SpecificDetail):
                    field = SpecificDetail(field)

                self.fields.append(field)

        @classmethod
        def from_string(cls, details_string):
            if details_string == '':
                fields = []
            else:
                fields_strings = details_string.split(INTERMEDIATE_DELIMITER)
                fields = [SpecificDetail.from_string(s) for s in fields_strings]

            return cls(fields)

        def __str__(self):
            return INTERMEDIATE_DELIMITER.join(str(field) for field in self.fields)

        def perform_anchor_shift(self, anchor):
            shifted_fields = [field.perform_anchor_shift(anchor) for field in self.fields]
            return type(self)(shifted_fields)
            
    return DetailList

def Details_factory(class_name, field_names_and_types, delimiter=';'):
    class Details(Detail):
        def __init__(self, *fields):
            if len(fields) != len(field_names_and_types):
                raise ValueError

            self.fields = []

            for (name, Field), field in zip(field_names_and_types, fields):
                if not isinstance(field, Detail):
                    field = Field(field)

                self.fields.append(field)

                setattr(self, name, field)
                
        @classmethod
        def from_string(cls, details_string):
            fields_strings = details_string.split(delimiter)

            if len(fields_strings) != len(field_names_and_types):
                raise ValueError

            fields = [Field.from_string(s) for (_, Field), s in zip(field_names_and_types, fields_strings)]

            return cls(*fields)
        
        def __str__(self):
            return delimiter.join(str(field) for field in self.fields)

        def perform_anchor_shift(self, anchor):
            shifted_fields = [field.perform_anchor_shift(anchor) for field in self.fields]
            return type(self)(*shifted_fields)
            
        def undo_anchor_shift(self, anchor):
            return self.perform_anchor_shift(-anchor)

    Details.__name__ = class_name
    Details.__qualname__ = f'Details_factory.{class_name}'

    return Details

class DuplicationJunction(Detail):
    def __init__(self, lefts, rights):
        self.lefts = tuple(sorted(lefts))
        self.rights = tuple(sorted(rights))

    @classmethod
    def from_string(cls, details_string):
        lefts, rights = details_string.split(',')
        lefts = [int(l) for l in lefts.strip('{}').split('|')]
        rights = [int(r) for r in rights.strip('{}').split('|')]
        return cls(lefts, rights)
    
    def __str__(self):
        lefts_string = '|'.join(map(str, self.lefts))
        rights_string = '|'.join(map(str, self.rights))
        return f'{{{lefts_string}}},{{{rights_string}}}'

    def perform_anchor_shift(self, anchor):
        shifted_lefts = [v - anchor for v in self.lefts]
        shifted_rights = [v - anchor for v in self.rights]
        return type(self)(shifted_lefts, shifted_rights)

    def __eq__(self, other):
        return self.lefts == other.lefts and self.rights == other.rights

    def __hash__(self):
        return hash((self.lefts, self.rights))

Deletions = DetailList_factory(DegenerateDeletion)
Insertions = DetailList_factory(DegenerateInsertion)
DuplicationJunctions = DetailList_factory(DuplicationJunction)

DeletionPlusDuplicationJunctions = Details_factory(
    'DeletionPlusDuplicationJunctions',
    [
        ('deletions', Deletions),
        ('duplication_junctions', DuplicationJunctions),
    ],
)

IntDetail = simple_Detail_factory(int)

class AnchoredIntDetail(IntDetail):
    def perform_anchor_shift(self, anchor):
        return type(self)(self.value - anchor)

FloatDetail = simple_Detail_factory(float)

class FormattedFloatDetail(FloatDetail):
    def __str__(self):
        return f'{self.value:0.3f}'

StrDetail = simple_Detail_factory(str)

StrList = DetailList_factory(StrDetail)

DeletionsInsertionsMismatches = Details_factory(
    'DeletionsInsertionsMismatches',
    [
        ('deletions', Deletions),
        ('insertions', Insertions),
        ('mismatches', Mismatches),
    ],
)

UnintendedRejoining = Details_factory(
    'UnintendedRejoining',
    [
        ('left_edge', IntDetail),
        ('right_edge', IntDetail),
        ('junction_microhomology_length', IntDetail),
        ('integrase_sites', StrList),
    ],
)

ProgrammedEdit = Details_factory(
    'ProgrammedEdit', [
        ('programmed_substitution_read_bases', StrDetail),
        ('non_programmed_target_mismatches_outcome', Mismatches),
        ('non_programmed_edit_mismatches_outcome', Mismatches),
        ('deletions', Deletions),
        ('insertions', Insertions),
    ],
)

def Outcome_factory(categorizer, class_name, field_names_and_types):
    delimiter = '\t'

    Outcome = Details_factory(class_name, field_names_and_types, delimiter=delimiter)

    category_to_Details = categorizer.build_category_to_Details()

    @classmethod
    def from_string(cls, line):
        fields_strings = line.strip('\n').split(delimiter)

        if len(fields_strings) != len(field_names_and_types):
            raise ValueError

        fields = []

        for (name, Field), s in zip(field_names_and_types, fields_strings):
            if name == 'category':
                category = s
            elif name == 'details':
                Field = category_to_Details[category]

            field = Field.from_string(s)
            fields.append(field)

        return cls(*fields)
    
    @classmethod
    def from_layout(cls, layout, **overrides):
        ''' Use case for overrides is providing a specific read name
        to a layout looked up from a common sequences dictionary.
        '''
        fields = []
        for name, _ in field_names_and_types:

            if name in overrides:
                field = overrides[name]
            else:
                if name == 'details':
                    name = 'Details'

                field = getattr(layout, name)

            fields.append(field)

        return cls(*fields)

    Outcome.from_string = from_string
    Outcome.from_layout = from_layout

    return Outcome

Outcome_binder = functools.partial(
    Outcome_factory,
    class_name='Outcome',
    field_names_and_types=[
        ('query_name', StrDetail),
        ('inferred_amplicon_length', IntDetail),
        ('Q30_fraction', FormattedFloatDetail),
        ('mean_Q', FormattedFloatDetail),
        ('UMI_seq', StrDetail),
        ('UMI_qual', StrDetail),
        ('category', StrDetail),
        ('subcategory', StrDetail),
        ('details', None),
    ],
)

CommonSequenceOutcome_binder = functools.partial(
    Outcome_factory,
    class_name='CommonSequenceOutcome',
    field_names_and_types=[
        ('query_name', StrDetail),
        ('inferred_amplicon_length', IntDetail),
        ('category', StrDetail),
        ('subcategory', StrDetail),
        ('details', None),
        ('seq', StrDetail),
    ],
)

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