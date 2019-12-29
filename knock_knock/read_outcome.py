def Outcome_factory(columns_arg, converters_arg):
    field_index_to_converter = {}
    for i, c in enumerate(columns_arg):
        if c in converters_arg:
            field_index_to_converter[i] = converters_arg[c]
    
    class Outcome():
        columns = columns_arg

        def __init__(self, *args):
            for name, arg in zip(columns_arg, args):
                setattr(self, name, arg)

        @classmethod
        def from_line(cls, line):
            fields = line.strip().split('\t')
            for i, converter in field_index_to_converter.items():
                fields[i] = converter(fields[i])

            return cls(*fields)

        @property
        def outcome(self):
            return (self.category, self.subcategory, self.details)        

        def __str__(self):
            row = [str(getattr(self, k)) for k in columns_arg]
            return '\t'.join(row)

        def __repr__(self):
            return str(self)
    
    return Outcome

Outcome = Outcome_factory(
    columns_arg=[
        'query_name',
        'length',
        'category',
        'subcategory',
        'details',
    ],
    converters_arg={'length': int},
)