def Outcome_factory(columns, converters):
    field_index_to_converter = {}
    for i, c in enumerate(columns):
        if c in converters:
            field_index_to_converter[i] = converters[c]
    
    class Outcome():
        def __init__(self, *args):
            for name, arg in zip(columns, args):
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
            row = [str(getattr(self, k)) for k in columns]
            return '\t'.join(row)
    
    return Outcome

Outcome = Outcome_factory(['query_name',
                           'length',
                           'category',
                           'subcategory',
                           'details',
                           ],
                          {'length': int},
                         )
