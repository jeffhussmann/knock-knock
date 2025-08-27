import inspect
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

import hits.interval

class Detail:
    def perform_anchor_shift(self, anchor):
        return self

    def undo_anchor_shift(self, anchor):
        return self.perform_anchor_shift(-anchor)

    @classmethod
    def from_string(cls, string):
        return cls(string)

class DegenerateDeletion(Detail):
    def __init__(self, starts_ats, length):
        self.kind = 'D'
        self.starts_ats = tuple(sorted(starts_ats))
        self.num_MH_nts = len(self.starts_ats) - 1
        self.length = length
        self.ends_ats = [s + self.length - 1 for s in self.starts_ats]

    @classmethod
    def from_string(cls, details_string):
        kind, rest = details_string.split(':', 1)
        starts_string, length_string = rest.split(',')

        starts_ats = [int(s) for s in starts_string.strip('{}').split('|')]
        length = int(length_string)

        return cls(starts_ats, length)

    @classmethod
    def collapse(cls, degenerate_deletions):
        lengths = {d.length for d in degenerate_deletions}
        if len(lengths) > 1:
            for d in degenerate_deletions:
                print(d)
            raise ValueError
        length = lengths.pop()

        starts_ats = set()
        for d in degenerate_deletions:
            starts_ats.update(d.starts_ats)

        starts_ats = sorted(starts_ats)

        return cls(starts_ats, length)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_ats))
        if len(self.starts_ats) > 1:
            starts_string = '{' + starts_string + '}'

        full_string = f'D:{starts_string},{self.length}'

        return full_string

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.starts_ats == other.starts_ats and self.length == other.length

    def __lt__(self, other):
        return (self.starts_ats[0] < other.starts_ats[0]) or (self.starts_ats[0] == other.starts_ats[0] and self.length < other.length)

    def __hash__(self):
        return hash((self.starts_ats, self.length))

    def singletons(self):
        return (type(self)([starts_at], self.length) for starts_at in self.starts_ats)

    @property
    def possibly_involved_interval(self):
        return hits.interval.Interval(min(self.starts_ats), max(self.ends_ats))

    def perform_anchor_shift(self, anchor):
        shifted_starts_ats = [starts_at - anchor for starts_at in self.starts_ats]
        return type(self)(shifted_starts_ats, self.length)

    def get_min_removed(self, ti):
        min_removed = {
            5: max(0, ti.cut_after - max(self.deletion.starts_ats) + 1),
            3: max(0, min(self.deletion.ends_ats) - ti.cut_after),
        }
        for target_side, PAM_side in ti.target_side_to_PAM_side.items():
            min_removed[PAM_side] = min_removed[target_side]

        return min_removed

    def classify_directionality(self, ti):
        if ti.effector.name == 'SpCas9':
            min_removed = self.get_min_removed(ti)

            if min_removed['PAM-proximal'] > 0 and min_removed['PAM-distal'] > 0:
                directionality = 'bidirectional'
            elif min_removed['PAM-proximal'] > 0 and min_removed['PAM-distal'] <= 0:
                directionality = 'PAM-proximal'
            elif min_removed['PAM-proximal'] <= 0 and min_removed['PAM-distal'] > 0:
                directionality = 'PAM-distal'
            else:
                directionality = 'ambiguous'
        
        elif ti.effector.name in ['Cpf1', 'AsCas12a']:
            start = min(self.deletion.starts_ats)
            end = max(self.deletion.ends_ats)
            first_nick, second_nick = sorted(ti.cut_afters.values())
            includes_first_nick = start - 0.5 <= first_nick + 0.5 <= end + 0.5
            includes_second_nick = start - 0.5 <= second_nick + 0.5 <= end + 0.5

            if not includes_first_nick and not includes_second_nick:
                includes = 'spans neither'
            elif includes_first_nick and not includes_second_nick:
                includes = 'spans PAM-distal nick'
            elif not includes_first_nick and includes_second_nick:
                includes = 'spans PAM-proximal nick'
            else:
                includes = 'spans both nicks'

            directionality = includes

        else:
            raise NotImplementedError(ti.effector.name)

        return directionality

class DegenerateInsertion(Detail):
    def __init__(self, starts_afters, seqs):
        self.kind = 'I'
        order = np.argsort(starts_afters)
        starts_afters = [starts_afters[i] for i in order]
        seqs = [seqs[i] for i in order]
        self.starts_afters = tuple(starts_afters)
        self.seqs = tuple(seqs)
        
        lengths = set(len(seq) for seq in self.seqs)
        if len(lengths) > 1:
            raise ValueError
        self.length = lengths.pop()
    
        self.pairs = list(zip(self.starts_afters, self.seqs))

    @classmethod
    def from_string(cls, details_string):
        kind, rest = details_string.split(':', 1)
        starts_string, seqs_string = rest.split(',')
        starts_afters = [int(s) for s in starts_string.strip('{}').split('|')]
        seqs = [seq for seq in seqs_string.strip('{}').split('|')]

        return DegenerateInsertion(starts_afters, seqs)
    
    @classmethod
    def from_pairs(cls, pairs):
        starts_afters, seqs = zip(*pairs)
        return DegenerateInsertion(starts_afters, seqs)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_afters))
        seqs_string = '|'.join(self.seqs)

        if len(self.starts_afters) > 1:
            starts_string = '{' + starts_string + '}'
            seqs_string = '{' + seqs_string + '}'

        full_string = f'I:{starts_string},{seqs_string}'

        return full_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.starts_afters == other.starts_afters and self.seqs == other.seqs
    
    def __hash__(self):
        return hash((self.starts_afters, self.seqs))

    def singletons(self):
        return (DegenerateInsertion([starts_after], [seq]) for starts_after, seq in self.pairs)
    
    @classmethod
    def collapse(cls, degenerate_insertions):
        all_pairs = []

        for d in degenerate_insertions:
            all_pairs.extend(d.pairs)

        all_pairs = sorted(all_pairs)

        return DegenerateInsertion.from_pairs(all_pairs)

    @property
    def possibly_involved_interval(self):
        return hits.interval.Interval(min(self.starts_afters), max(self.starts_afters) + 1)

    def perform_anchor_shift(self, anchor):
        shifted_starts_afters = [starts_after - anchor for starts_after in self.starts_afters]
        return type(self)(shifted_starts_afters, self.seqs)

def degenerate_indel_from_string(details_string):
    if details_string is None:
        return None
    else:
        kind, rest = details_string.split(':')

        if kind == 'D':
            DegenerateIndel = DegenerateDeletion
        elif kind == 'I':
            DegenerateIndel = DegenerateInsertion
        else:
            raise ValueError(kind)

        return DegenerateIndel.from_string(details_string)

class Mismatch:
    def __init__(self, position, basecall):
        self.position = position
        self.basecall = basecall

    @classmethod
    def from_string(cls, details_string):
        position = int(details_string[:-1])
        basecall = details_string[-1]

        return cls(position, basecall)

    def perform_anchor_shift(self, anchor):
        return type(self)(self.position - anchor, self.basecall)

    def __str__(self):
        return f'{self.position}{self.basecall}'

class Mismatches:
    def __init__(self, mismatches=None):
        if mismatches is None:
            mismatches = []
        self.mismatches = sorted(mismatches, key=lambda mismatch: mismatch.position)
    
    def __str__(self):
        return ','.join(str(mismatch) for mismatch in self.mismatches)

    @classmethod
    def from_string(cls, details_string):
        if details_string == '' or details_string == 'collapsed':
            mismatches = []
        else:
            mismatches = [Mismatch.from_string(s) for s in details_string.split(',')]

        return cls(mismatches)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.mismatches)

    def __iter__(self):
        return iter(self.mismatches)

    @property
    def positions(self):
        return [mismatch.position for mismatch in self.mismatches]
    
    @property
    def basecalls(self):
        return [mismatch.basecall for mismatch in self.mismatches]

    def __lt__(self, other):
        if max(self.positions) != max(other.positions):
            return max(self.positions) < max(other.positions)
        else:
            if len(self) < len(other):
                return True
            elif len(self) == len(other):
                if self.positions != other.positions:
                    return self.positions < other.positions
                else: 
                    return self.basecalls < other.basecalls
            else:
                return False

    def perform_anchor_shift(self, anchor):
        return type(self)([mismatch.perform_anchor_shift(anchor) for mismatch in self.mismatches])

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

    def __eq__(self, other):
        return isinstance(other, Details) and str(self) == str(other)

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
    def from_architecture(cls, architecture, **overrides):
        args = [overrides[name] if name in overrides else getattr(architecture, name) for name, _ in cls.parameters]
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

def add_directionalities_to_deletions(outcomes, editing_strategy):
    combined_categories = []

    for category, subcategory, details in outcomes:
        if category == 'deletion':
            deletion = knock_knock.outcome.DeletionOutcome.from_string(details).undo_anchor_shift(editing_strategy.anchor)
            directionality = deletion.classify_directionality(editing_strategy)
            combined_category = f'{category}, {directionality}'
        else:
            combined_category = category
            
        combined_categories.append(combined_category)

    return combined_categories

def extract_deletion_boundaries(editing_strategy,
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
    index = np.arange(len(editing_strategy.target_sequence))
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

def outcomes_containing_pegRNA_programmed_edits(editing_strategy,
                                                outcome_fractions,
                                               ):
    outcomes_containing_pegRNA_programmed_edits = {}

    if editing_strategy.pegRNA_substitutions is not None:
        subs = editing_strategy.pegRNA_substitutions[editing_strategy.target]
        # Note: sorting subs is critical here to match order set elsewhere
        sub_order = sorted(subs)

        for sub_name in subs:
            outcomes_containing_pegRNA_programmed_edits[sub_name] = []

    else:
        subs = None
    
    if editing_strategy.pegRNA_programmed_insertion is not None:
        insertion = editing_strategy.pegRNA_programmed_insertion

        outcomes_containing_pegRNA_programmed_edits[str(insertion)] = []

    else:
        insertion = None

    if editing_strategy.pegRNA_programmed_deletion is not None:
        deletion = editing_strategy.pegRNA_programmed_deletion

        outcomes_containing_pegRNA_programmed_edits[str(deletion)] = []

    else:
        deletion = None

    if outcome_fractions is not None:
        for c, s, d  in outcome_fractions.index:
            if c in {'intended edit', 'partial replacement', 'partial edit'}:
                details = Details.from_string(d)

                if subs is not None:
                    for sub_name, read_base in zip(sub_order, details.programmed_substitution_read_bases):
                        sub = subs[sub_name]
                        if read_base == sub['alternative_base']:
                            outcomes_containing_pegRNA_programmed_edits[sub_name].append((c, s, d))

                if insertion is not None and insertion in details['insertions']:
                    outcomes_containing_pegRNA_programmed_edits[str(insertion)].append((c, s, d))

                if deletion is not None and deletion in details['deletions']:
                    outcomes_containing_pegRNA_programmed_edits[str(deletion)].append((c, s, d))

    return outcomes_containing_pegRNA_programmed_edits