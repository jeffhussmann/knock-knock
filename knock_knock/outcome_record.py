def OutcomeRecord_factory(columns_arg, converters_arg):
    # Converters go from text to usable form,
    # formatters go from usable form to text.

    field_index_to_converter = {}
    field_index_to_formatter = {}

    for i, c in enumerate(columns_arg):
        if c in converters_arg:
            if isinstance(converters_arg[c], tuple):
                converter, formatter = converters_arg[c]
            else:
                converter = converters_arg[c]
                formatter = str

            field_index_to_converter[i] = converter
            field_index_to_formatter[i] = formatter
    
    class OutcomeRecord:
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
            fields = [getattr(self, k) for k in columns_arg]
            for i, formatter in field_index_to_formatter.items():
                fields[i] = formatter(fields[i])

            return '\t'.join(fields)

        def __repr__(self):
            return str(self)
    
    return OutcomeRecord

OutcomeRecord = OutcomeRecord_factory(
    columns_arg=[
        'query_name',
        'inferred_amplicon_length',
        'Q30_fraction',
        'mean_Q',
        'UMI_seq',
        'UMI_qual',
        'category',
        'subcategory',
        'details',
    ],
    converters_arg={
        'inferred_amplicon_length': int,
        'Q30_fraction': (float, '{:0.3f}'.format),
        'mean_Q': (float, '{:0.3f}'.format),
    },
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
    converters_arg={
        'inferred_amplicon_length': int,
    },
)