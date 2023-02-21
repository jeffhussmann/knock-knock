def OutcomeRecord_factory(columns_arg, converters_arg):
    field_index_to_converter = {}
    for i, c in enumerate(columns_arg):
        if c in converters_arg:
            field_index_to_converter[i] = converters_arg[c]
    
    class OutcomeRecord():
        columns = columns_arg

        def __init__(self, *args):
            for name, arg in zip(columns_arg, args):
                setattr(self, name, arg)

        @classmethod
        def from_line(cls, line):
            fields = line.strip('\n').split('\t')
            for i, converter in field_index_to_converter.items():
                fields[i] = converter(fields[i])

            return cls(*fields)

        @classmethod
        def from_layout(cls, layout, **overrides):
            ''' Use case for overrides is providing a specific read name
            to a layout looked up from a common sequences dictionary.
            '''
            args = [overrides[k] if k in overrides else getattr(layout, k) for k in columns_arg]
            return cls(*args)

        @property
        def outcome(self):
            return (self.category, self.subcategory, self.details)        

        def __str__(self):
            row = [str(getattr(self, k)) for k in columns_arg]
            return '\t'.join(row)

        def __repr__(self):
            return str(self)
    
    return OutcomeRecord

OutcomeRecord = OutcomeRecord_factory(
    columns_arg=[
        'query_name',
        'inferred_amplicon_length',
        'category',
        'subcategory',
        'details',
    ],
    converters_arg={'inferred_amplicon_length': int},
)

CommonSequenceOutcomeRecord = OutcomeRecord_factory(
    columns_arg=[
        'query_name',
        'inferred_amplicon_length',
        'category',
        'subcategory',
        'details',
        'seq',
    ],
    converters_arg={'inferred_amplicon_length': int},
)

class Integration:
    def __init__(self, target_edge_before, target_edge_after, donor_strand, donor_start, donor_end, mh_length_5, mh_length_3):
        self.target_edge_before = target_edge_before
        self.target_edge_after = target_edge_after
        self.donor_strand = donor_strand
        self.donor_start = donor_start
        self.donor_end = donor_end
        self.mh_length_5 = mh_length_5
        self.mh_length_3 = mh_length_3

    @classmethod
    def from_string(cls, details_string):
        fields = [int(f) if (f != '+' and f != '-') else f for f in details_string.split(',')]
        return Integration(*fields)

    def __str__(self):
        return ','.join(map(str, [self.target_edge_before, self.target_edge_after, self.donor_strand, self.donor_start, self.donor_end, self.mh_length_5, self.mh_length_3]))