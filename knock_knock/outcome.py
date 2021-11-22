import numpy as np

import knock_knock.outcome
from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs

class Outcome:
    def __init__(self, description):
        self.description = description

    def __str__(self):
        return self.description

    def perform_anchor_shift(self, anchor):
        return self

    def undo_anchor_shift(self, anchor):
        return self.perform_anchor_shift(-anchor)

class DeletionOutcome(Outcome):
    def __init__(self, deletion):
        self.deletion = deletion

    @classmethod
    def from_string(cls, details_string):
        deletion = DegenerateDeletion.from_string(details_string)
        return cls(deletion)

    def __str__(self):
        return str(self.deletion)

    def perform_anchor_shift(self, anchor):
        shifted_starts_ats = [starts_at - anchor for starts_at in self.deletion.starts_ats]
        shifted_deletion = DegenerateDeletion(shifted_starts_ats, self.deletion.length)
        return type(self)(shifted_deletion)

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

class InsertionOutcome(Outcome):
    def __init__(self, insertion):
        self.insertion = insertion

    @classmethod
    def from_string(cls, details_string):
        insertion = DegenerateInsertion.from_string(details_string)
        return cls(insertion)

    def __str__(self):
        return str(self.insertion)

    def perform_anchor_shift(self, anchor):
        shifted_starts_afters = [starts_after - anchor for starts_after in self.insertion.starts_afters]
        shifted_insertion = DegenerateInsertion(shifted_starts_afters, self.insertion.seqs)
        return type(self)(shifted_insertion)

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

class MultipleDeletionOutcome(Outcome):
    def __init__(self, deletion_outcomes):
        self.deletion_outcomes = deletion_outcomes
    
    @classmethod
    def from_string(cls, details_string):
        deletion_strings = details_string.split(';', 1)
        deletion_outcomes = [DeletionOutcome.from_string(s) for s in deletion_strings]

        return cls(deletion_outcomes)

    def __str__(self):
        return ';'.join(map(str, self.deletion_outcomes))

    def perform_anchor_shift(self, anchor):
        shifted_deletions = [d.perform_anchor_shift(anchor) for d in self.deletion_outcomes]
        return type(self)(shifted_deletions)

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

class MismatchOutcome(Outcome):
    def __init__(self, snvs):
        self.snvs = snvs

    @classmethod
    def from_string(cls, details_string):
        snvs = SNVs.from_string(details_string)
        return cls(snvs)

    def __str__(self):
        return str(self.snvs)

    def perform_anchor_shift(self, anchor):
        shifted_snvs = SNVs([SNV(s.position - anchor, s.basecall, s.quality) for s in self.snvs])
        return type(self)(shifted_snvs)

class TruncationOutcome(Outcome):
    def __init__(self, edge):
        self.edge = edge

    @classmethod
    def from_string(cls, details_string):
        return cls(int(details_string))

    def __str__(self):
        return str(self.edge)

    def perform_anchor_shift(self, anchor):
        return TruncationOutcome(self.edge - anchor)

class DeletionPlusMismatchOutcome(Outcome):
    def __init__(self, deletion_outcome, mismatch_outcome):
        self.deletion_outcome = deletion_outcome
        self.mismatch_outcome = mismatch_outcome

    @classmethod
    def from_string(cls, details_string):
        deletion_string, mismatch_string = details_string.split(';', 1)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)
        mismatch_outcome = MismatchOutcome.from_string(mismatch_string)
        return cls(deletion_outcome, mismatch_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.mismatch_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_deletion = self.deletion_outcome.perform_anchor_shift(anchor)
        shifted_mismatch = self.mismatch_outcome.perform_anchor_shift(anchor)
        return type(self)(shifted_deletion, shifted_mismatch)

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

class InsertionWithDeletionOutcome(Outcome):
    ''' Deletion with at the same place. '''
    def __init__(self, insertion_outcome, deletion_outcome):
        self.insertion_outcome = insertion_outcome
        self.deletion_outcome = deletion_outcome

    @classmethod
    def from_string(cls, details_string):
        insertion_string, deletion_string = details_string.split(';', 1)
        insertion_outcome = InsertionOutcome.from_string(insertion_string)
        deletion_outcome = DeletionOutcome.from_string(deletion_string)
        return cls(insertion_outcome, deletion_outcome)

    def __str__(self):
        return f'{self.insertion_outcome};{self.deletion_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_insertion = self.insertion_outcome.perform_anchor_shift(anchor)
        shifted_deletion = self.deletion_outcome.perform_anchor_shift(anchor)
        return type(self)(shifted_insertion, shifted_deletion)

NAN_INT = np.iinfo(np.int64).min

def int_or_nan_from_string(s):
    if s == 'None':
        return NAN_INT
    else:
        return int(s)

class LongTemplatedInsertionOutcome(Outcome):
    field_names = [
        'source',
        'ref_name',
        'strand',
        'left_insertion_ref_bound',
        'right_insertion_ref_bound',
        'left_insertion_query_bound',
        'right_insertion_query_bound',
        'left_target_ref_bound',
        'right_target_ref_bound',
        'left_target_query_bound',
        'right_target_query_bound',
        'left_MH_length',
        'right_MH_length',
        'donor_SNV_summary_string',
    ]

    int_fields = {
        'left_insertion_ref_bound',
        'right_insertion_ref_bound',
        'left_insertion_query_bound',
        'right_insertion_query_bound',
        'left_target_ref_bound',
        'right_target_ref_bound',
        'left_target_query_bound',
        'right_target_query_bound',
        'left_MH_length',
        'right_MH_length',
    }

    field_index_to_converter = {}
    for i, c in enumerate(field_names):
        if c in int_fields:
            field_index_to_converter[i] = int_or_nan_from_string

    def __init__(self, *args):
        for name, arg in zip(self.__class__.field_names, args):
            setattr(self, name, arg)

    @property
    def left_gap(self):
        return self.left_insertion_query_bound - self.left_target_query_bound - 1

    @property
    def right_gap(self):
        return self.right_target_query_bound - self.right_insertion_query_bound - 1

    def insertion_length(self, single_end_read_length=None):
        if single_end_read_length is not None and self.right_insertion_query_bound == single_end_read_length - 1:
            length = single_end_read_length
        else:
            length = self.right_insertion_query_bound - self.left_insertion_query_bound + 1
        
        return length

    @classmethod
    def from_string(cls, details_string):
        fields = details_string.split(',')

        for i, converter in cls.field_index_to_converter.items():
            fields[i] = converter(fields[i])

        return cls(*fields)

    def __str__(self):
        row = [str(getattr(self, k)) for k in self.__class__.field_names]
        return ','.join(row)

    def pprint(self):
        for field_name in self.__class__.field_names:
            print(f'{field_name}: {getattr(self, field_name)}') 

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