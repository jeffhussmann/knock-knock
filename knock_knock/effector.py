import Bio.SeqUtils

from hits import utilities, gff

class Effector:
    def __init__(self, name, PAM_pattern, PAM_side, cut_after_offset):
        self.name = name
        self.PAM_pattern = PAM_pattern
        self.PAM_side = PAM_side
        # cut_after_offset is relative to the 5'-most nt of the PAM
        self.cut_after_offset = cut_after_offset

    def __repr__(self):
        return f"{type(self).__name__}('{self.name}', '{self.PAM_pattern}', {self.PAM_side}, {self.cut_after_offset})"

    def PAM_slice(self, protospacer_feature):
        before_slice = slice(protospacer_feature.start - len(self.PAM_pattern), protospacer_feature.start)
        after_slice = slice(protospacer_feature.end + 1, protospacer_feature.end + 1 + len(self.PAM_pattern))

        if (protospacer_feature.strand == '+' and self.PAM_side == 5) or (protospacer_feature.strand == '-' and self.PAM_side == 3):
            PAM_slice = before_slice
        else:
            PAM_slice = after_slice

        return PAM_slice

    def PAM_matches_pattern(self, protospacer_feature, target_sequence):
        PAM_seq = target_sequence[self.PAM_slice(protospacer_feature)].upper()
        if protospacer_feature.strand == '-':
            PAM_seq = utilities.reverse_complement(PAM_seq)

        pattern, *matches = Bio.SeqUtils.nt_search(PAM_seq, self.PAM_pattern) 

        return 0 in matches

    def protospacer_feature(self, protospacer_sequence, protospacer_start, strand):
        protospacer_end = protospacer_start + len(protospacer_sequence) - 1

        feature = gff.Feature.from_fields(start=protospacer_start,
                                          end=protospacer_end,
                                          strand=strand,
                                          feature='sgRNA', 
                                          attribute_string=gff.make_attribute_string({
                                              'color': 'lightgrey',
                                              'effector': self.name,
                                          }),
                                         )

        return feature

    def protospacer_has_PAM(self, protospacer_sequence, protospacer_start, strand, target_sequence):
        feature = self.protospacer_feature(protospacer_sequence, protospacer_start, strand)
        
        return self.PAM_matches_pattern(feature, target_sequence)

    def cut_afters(self, protospacer_feature):
        ''' Returns a dictionary of {strand: position after which nick is made} '''

        if protospacer_feature.strand == '+':
            offset_strand_order = '+-'
        else:
            offset_strand_order = '-+'

        if len(set(self.cut_after_offset)) == 1:
            # Blunt DSB
            offsets = list(set(self.cut_after_offset))
            strands = ['both']
        else:
            offsets = [offset for offset in self.cut_after_offset if offset is not None]
            strands = [strand for strand, offset in zip(offset_strand_order, self.cut_after_offset) if offset is not None]

        cut_afters = {}
        PAM_slice = self.PAM_slice(protospacer_feature)

        for offset, strand in zip(offsets, strands):
            if protospacer_feature.strand == '+':
                PAM_5 = PAM_slice.start
                cut_after = PAM_5 + offset
            else:
                PAM_5 = PAM_slice.stop - 1
                # -1 extra because cut_after is on the other side of the cut
                cut_after = PAM_5 - offset - 1
            
            cut_afters[strand] = cut_after

        return cut_afters

    def identify_protospacer_in_target(self, target_sequence, protospacer):
        ''' Find an occurence of protospacer on either strand of target_sequence
        that has a PAM for effector positioned appropriately. If there is more
        than one such occurence, raise a ValueError. 
        Because the first nt of protospacer might be a non-matching G, first
        look for the whole protospacer. If no match is found, look for the
        protospacer with the first nt removed.
        '''
        def find(protospacer_suffix):
            valid_features = []

            for strand, ps_seq in [('+', protospacer_suffix),
                                   ('-', utilities.reverse_complement(protospacer_suffix)),
                                  ]:

                protospacer_starts = utilities.find_all_substring_starts(target_sequence, ps_seq)
                
                for ps_start in protospacer_starts:
                    target_protospacer_feature = self.protospacer_feature(ps_seq, ps_start, strand)
                    
                    if self.PAM_matches_pattern(target_protospacer_feature, target_sequence):
                        valid_features.append(target_protospacer_feature)

            return valid_features
                    
        valid_features = find(protospacer)

        if len(valid_features) == 0:
            valid_features = find(protospacer[1:])

        if len(valid_features) != 1:
            raise ValueError(f'{len(valid_features)} valid locations for protospacer {protospacer} in target {target_sequence if len(target_sequence) < 1000 else ">1kb long"}')

        valid_feature = valid_features[0]

        return valid_feature

# tuples are (PAM_pattern, PAM side, cut_after_offset)
effector_details = {
    'SpCas9': ('NGG', 3, (-4, -4)), 
    'SpCas9H840A': ('NGG', 3, (-4, None)),

    'SpCas9_VRQR': ('NGA', 3, (-4, -4)),
    'SpCas9H840A_VRQR': ('NGA', 3, (-4, None)),

    'SaCas9': ('NNGRRT', 3, (-4, -4)),
    'SaCas9H840A': ('NNGRRT', 3, (-4, None)),

    'AsCas12a': ('TTTN', 5, (20, 25)),
}

effectors = {
    name: Effector(name, PAM_pattern, PAM_side, cut_after_offset) 
    for name, (PAM_pattern, PAM_side, cut_after_offset) in effector_details.items()
}

effector_aliases = {
    'AsCas12a': {'Cpf1', 'enAsCas12a', 'enCas12a'},
}

for main_name, aliases in effector_aliases.items():
    for alias in aliases:
        effectors[alias] = effectors[main_name]
