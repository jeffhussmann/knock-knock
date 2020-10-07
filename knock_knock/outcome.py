import numpy as np

from knock_knock.target_info import DegenerateDeletion, DegenerateInsertion, SNV, SNVs

class Outcome:
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
        return HDROutcome(SNV_string, deletions)

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

        return HDRPlusDeletionOutcome(HDR_outcome, deletion_outcome)

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

        return DeletionPlusDuplicationOutcome(deletion_outcome, duplication_outcome)

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

        return MultipleDeletionOutcome(deletion_outcomes)

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

        return HDRPlusInsertionOutcome(HDR_outcome, insertion_outcome)

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
        return MismatchOutcome(snvs)

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
        return TruncationOutcome(int(details_string))

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
        return DeletionPlusMismatchOutcome(deletion_outcome, mismatch_outcome)

    def __str__(self):
        return f'{self.deletion_outcome};{self.mismatch_outcome}'

    def perform_anchor_shift(self, anchor):
        shifted_deletion = self.deletion_outcome.perform_anchor_shift(anchor)
        shifted_mismatch = self.mismatch_outcome.perform_anchor_shift(anchor)
        return type(self)(shifted_deletion, shifted_mismatch)

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
            l, r = field.split(',')
            ref_junctions.append((int(l), int(r)))
        return DuplicationOutcome(ref_junctions)

    def __str__(self):
        return ';'.join(f'{left},{right}' for left, right in self.ref_junctions)

    def perform_anchor_shift(self, anchor):
        shifted_ref_junctions = [(left - anchor, right - anchor) for left, right in self.ref_junctions]
        return type(self)(shifted_ref_junctions)