import bokeh.palettes
import Bio.SeqUtils

import hits.utilities
import hits.gff

recognition_sequences = {
    'Bxb1': {
        'attP': 'GGTTTGTCTGGTCAACCACCGCGNNCTCAGTGGTGTACGGTACAAACC',
        'attB': 'GGCTTGTCGACGACGGCGNNCTCCGTCGTCAGGATCAT',
    }
}

split_recognition_sequences = {}
for source, site_seqs in recognition_sequences.items():
    split_recognition_sequences[source] = {}
    for site_name, site_seq in site_seqs.items():
        split_seq = dict(zip(('left', 'right'), site_seq.split('NN')))
        split_recognition_sequences[source][site_name] = split_seq

colors = {
    ('attP', 'left'): bokeh.palettes.Category20b_20[10],
    ('attP', 'right'): bokeh.palettes.Category20b_20[11],
    ('attB', 'left'): bokeh.palettes.Category20b_20[18],
    ('attB', 'right'): bokeh.palettes.Category20b_20[19],
}

def identify_split_recognition_sequences(ref_seqs):
    features = {}

    for ref_name, ref_seq in ref_seqs.items():
        for source, site_seqs in split_recognition_sequences.items():
            for site_name, split_seq in site_seqs.items():
                for side, seq in split_seq.items(): 
                    pattern, *matches = Bio.SeqUtils.nt_search(ref_seq, seq)

                    seq_rc = hits.utilities.reverse_complement(seq)
                    pattern, *rc_matches = Bio.SeqUtils.nt_search(ref_seq, seq_rc)

                    if len(matches + rc_matches) > 1:
                        # TODO: annotate multiple.
                        return features

                    if len(matches) == 1:
                        strand = '+'
                    elif len(rc_matches) == 1:
                        strand = '-'
                    else:
                        continue

                    match_start = (matches + rc_matches)[0]
                    match_end = match_start + len(seq) - 1
                    
                    full_name = f'{source}_{site_name}_{side}'
                    feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                           start=match_start,
                                                           end=match_end,
                                                           ID=full_name,
                                                           strand=strand,
                                                          )
                    feature.attribute['color'] = colors.get((site_name, side))

                    features[ref_name, full_name] = feature

                    # Annotate the central dinucleotide next to the left half.
                    if side == 'left':
                        if strand == '+':
                            CD_start = match_end + 1
                        else:
                            CD_start = match_start - 2
                    elif side == 'right':
                        if strand == '+':
                            CD_start = match_start - 2
                        else:
                            CD_start = match_end + 1
                    else:
                        raise ValueError(side)

                    CD_end = CD_start + 1

                    full_name = f'{source}_{site_name}_CD'
                    feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                           start=CD_start,
                                                           end=CD_end,
                                                           ID=full_name,
                                                           strand=strand,
                                                          )

                    features[ref_name, full_name] = feature

    return features

def identify_recognition_sequences(ref_seqs):
    features = {}

    for ref_name, ref_seq in ref_seqs.items():
        for source, site_seqs in recognition_sequences.items():
            for site_name, site_seq in site_seqs.items():
                pattern, *matches = Bio.SeqUtils.nt_search(ref_seq, site_seq)

                site_seq_rc = hits.utilities.reverse_complement(site_seq)
                pattern, *rc_matches = Bio.SeqUtils.nt_search(ref_seq, site_seq_rc)

                if len(matches + rc_matches) > 1:
                    raise NotImplementedError

                central_dinucleotide_offset = site_seq.index('N')

                feature_fields = []

                for match in matches:
                    feature_fields.extend([
                        ('left',
                         match,
                         match + central_dinucleotide_offset - 1,
                         '+',
                        ),
                        ('CD',
                         match + central_dinucleotide_offset,
                         match + central_dinucleotide_offset + 1,
                        '+',
                        ),
                        ('right',
                         match + central_dinucleotide_offset + 2, 
                         match + len(site_seq) - 1,
                         '+',
                        ),
                    ])

                for rc_match in rc_matches:
                    overall_end = rc_match + len(site_seq) - 1

                    feature_fields.extend([
                        ('left',
                         overall_end - (central_dinucleotide_offset - 1),
                         overall_end,
                         '-',
                        ),
                        ('CD',
                         overall_end - central_dinucleotide_offset - 1,
                         overall_end - central_dinucleotide_offset,
                         '-',
                        ),
                        ('right',
                         rc_match,
                         overall_end - central_dinucleotide_offset - 2,
                         '-',
                        ),
                    ])

                for name, start, end, strand in feature_fields:
                    full_name = f'{source}_{site_name}_{name}'
                    feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                           start=start,
                                                           end=end,
                                                           ID=full_name,
                                                           strand=strand,
                                                          )
                    feature.attribute['color'] = colors.get((site_name, name))

                    features[ref_name, full_name] = feature

    return features