import inspect
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, Mismatches

class Detail:
    def perform_anchor_shift(self, anchor):
        return self

    def undo_anchor_shift(self, anchor):
        return self.perform_anchor_shift(-anchor)

    @classmethod
    def from_string(cls, string):
        return cls(string)

def DetailList_factory(SpecificDetail):
    delimiter = ';'

    class DetailList(Detail):
        def __init__(self, fields=None):
            if fields is None:
                fields = []

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
                fields_strings = details_string.split(delimiter)
                fields = [SpecificDetail.from_string(s) for s in fields_strings]

            return cls(fields)

        def __str__(self):
            return delimiter.join(str(field) for field in self.fields)

        def perform_anchor_shift(self, anchor):
            shifted_fields = [field.perform_anchor_shift(anchor) for field in self.fields]
            return type(self)(shifted_fields)

        def __len__(self):
            return len(self.fields)

        def __getitem__(self, index):
            return self.fields[index]

    return DetailList

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

class Int(int, Detail):
    pass

class AnchoredInt(Int):
    def perform_anchor_shift(self, anchor):
        return type(self)(self - anchor)

class FormattedFloat(float):
    def __str__(self):
        return f'{self:0.3f}'

class Str(str, Detail):
    pass

Strs = DetailList_factory(Str)

tag_to_Detail = OrderedDict({
    'left_rejoining_edge': Int,
    'right_rejoining_edge': Int,
    'junction_microhomology_length': Int,
    'integrase_sites': Strs,
    'programmed_substitution_read_bases': Str,
    'non_programmed_target_mismatches': Mismatches,
    'non_programmed_edit_mismatches': Mismatches,
    'duplication_junctions': DuplicationJunctions,
    'target_edge': AnchoredInt,
    'pegRNA_edge': Int,
    'deletions': Deletions,
    'insertions': Insertions,
    'mismatches': Mismatches,
})

tag_order = {tag: i for i, tag in enumerate(tag_to_Detail)}.__getitem__

_safe_chars = r'#\{}()+-:|,'

class Details(Detail):
    def __init__(self, **details):
        self._tags = sorted(details, key=tag_order)

        for tag, value in details.items():
            _Detail = tag_to_Detail[tag]

            if not isinstance(value, _Detail):
                value = _Detail(value)

            setattr(self, tag, value)
    
    @classmethod
    def from_string(cls, string):
        if string == '':
            fields = []
        else:
            fields = string.split(';')

        pairs = [field.split('=') for field in fields]

        parsed = {tag: tag_to_Detail[tag].from_string(urllib.parse.unquote(value)) for tag, value in pairs}

        return cls(**parsed)
    
    def __str__(self):
        fields = []
        for tag in self._tags:
            value = str(getattr(self, tag))
            if value != '':
                field = f'{tag}={urllib.parse.quote(value, safe=_safe_chars)}' 
                fields.append(field)

        return ';'.join(fields)

    def perform_anchor_shift(self, anchor):
        return type(self)(**{tag: getattr(self, tag).perform_anchor_shift(anchor) for tag in self._tags})

    def __getitem__(self, tag):
        return getattr(self, tag, tag_to_Detail[tag]())

    __repr__ = __str__

@dataclass
class CategorizationRecord:
    query_name: str
    inferred_amplicon_length: int
    Q30_fraction: FormattedFloat
    mean_Q: FormattedFloat
    category: str
    subcategory: str
    details: Details.from_string
    UMI_seq: str = ''
    UMI_qual: str = ''

    delimiter = '\t'

    @classmethod
    def from_string(cls, line):
        fields = line.strip('\n').split(cls.delimiter)
        
        if len(fields) != len(cls.parameters):
            raise ValueError(f'expected {len(cls.parameters)} parameters, got {len(fields)}')

        args = []
        for field, (name, parameter) in zip(fields, cls.parameters):
            args.append(parameter.annotation(field))

        return cls(*args)

    @classmethod
    def from_layout(cls, layout, **overrides):
        args = [overrides[name] if name in overrides else getattr(layout, name) for name, _ in cls.parameters]
        return cls(*args)

    def __str__(self):
        fields = []

        for name, parameter in self.parameters:
            field = getattr(self, name)
            if type(parameter.annotation) == type and not isinstance(field, parameter.annotation):
                field = parameter.annotation(field)

            fields.append(field)

        return '\t'.join(str(field) for field in fields)

signature = inspect.signature(CategorizationRecord.__init__)
CategorizationRecord.parameters = list(signature.parameters.items())[1:]

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
        details = Details.from_string(d)

        deletions = details['deletions']

        for deletion in deletions:
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