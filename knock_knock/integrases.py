import itertools
from collections import defaultdict

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
    all_features = {}

    for ref_name, ref_seq in ref_seqs.items():
        for source, site_seqs in split_recognition_sequences.items():
            for site_name, split_seq in site_seqs.items():
                component_features = defaultdict(list)

                for side, seq in split_seq.items(): 
                    all_matches = []

                    pattern, *matches = Bio.SeqUtils.nt_search(ref_seq, seq)
                    all_matches.extend([(match, '+') for match in matches])

                    seq_rc = hits.utilities.reverse_complement(seq)
                    pattern, *rc_matches = Bio.SeqUtils.nt_search(ref_seq, seq_rc)
                    all_matches.extend([(match, '-') for match in rc_matches])

                    for match_start, strand in all_matches:
                        match_end = match_start + len(seq) - 1
                        
                        side_feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                                    start=match_start,
                                                                    end=match_end,
                                                                    strand=strand,
                                                                   )
                        side_feature.attribute['color'] = colors.get((site_name, side))
                        side_feature.attribute['component'] = side

                        # Annotate the central dinucleotide after the left half.
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

                        if CD_start >= 0 and CD_end < len(ref_seq):
                            CD_feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                                      start=CD_start,
                                                                      end=CD_end,
                                                                      strand=strand,
                                                                     )
                            
                            CD_feature.attribute['component'] = 'CD'
                            CD = CD_feature.sequence(ref_seqs)
                            CD_feature.attribute['CD'] = CD
                            CD_full_name = f'{source}_{site_name}_{CD}_CD_{CD_start}'
                            CD_feature.attribute['ID'] = CD_full_name

                            component_features['CD'].append(CD_feature)
                            all_features[ref_name, CD_full_name] = CD_feature

                            side_full_name = f'{source}_{site_name}_{CD}_{side}_{match_start}'
                            side_feature.attribute['ID'] = side_full_name
                            component_features[side].append(side_feature)
                            all_features[ref_name, side_full_name] = side_feature

                for left, CD, right in itertools.product(component_features['left'], component_features['CD'], component_features['right']):
                    full_name = f'{source}_{site_name}_{left.start}'

                    if left.strand == '+' and CD.strand == '+' and right.strand == '+':
                        if left.end + 1 == CD.start and CD.end + 1 == right.start:
                            strand = '+'
                            start = left.start
                            end = right.end
                        else:
                            continue

                    elif left.strand == '-' and CD.strand == '-' and right.strand == '-':
                        if left.start == CD.end + 1 and CD.start == right.end + 1:
                            strand = '-'
                            start = right.start
                            end = left.end
                        else:
                            continue

                    else:
                       continue

                    feature = hits.gff.Feature.from_fields(seqname=ref_name,
                                                           start=start,
                                                           end=end,
                                                           ID=full_name,
                                                           strand=strand,
                                                          )

                    for child in [left, CD, right]:
                        feature.children.add(child)
                        child.parent = feature

                    feature.attribute['CD'] = CD.sequence(ref_seqs)
                    feature.attribute['component'] = 'complete_site'
                    feature.attribute['recombinase'] = source
                    feature.attribute['site'] = site_name

                    all_features[ref_name, full_name] = feature

    return all_features

def components(full_feature):
    return {child.attribute['component']: child for child in full_feature.children}

def recombine(ref_seqs, target, donor):
    features = identify_split_recognition_sequences(ref_seqs)

    sites = defaultdict(dict)

    opposite_site = {
        'attB': 'attP',
        'attP': 'attB',
    }

    for (ref_name, feature_name), feature in features.items():
        if feature.attribute['component'] == 'complete_site':
            source, site_name, *rest = feature_name.split('_')
            sites[ref_name][source, site_name, feature.attribute['CD']] = feature

    pairs = []

    for source, which, CD in sites[target]:
        if (source, opposite_site[which], CD) in sites[donor]:
            pairs.append((sites[target][source, which, CD], sites[donor][source, opposite_site[which], CD]))
            
    if len(pairs) == 0:
        raise ValueError

    elif len(pairs) == 1:
        target_site, donor_site = pairs[0]

        donor_components = components(donor_site)
        donor_seq = ref_seqs[donor]

        if donor_site.strand == '+':
            first_CD_p = donor_components['CD'].start
            integrated_donor = donor_seq[first_CD_p + 1:] + donor_seq[:first_CD_p + 1]
        else:
            raise NotImplementedError
            
        target_components = components(target_site)
        target_seq = ref_seqs[target]

        if target_site.strand == '+':
            first_CD_p = target_components['CD'].start
            target_before = target_seq[:first_CD_p + 1]
            target_after = target_seq[first_CD_p + 1:]
        else:
            raise NotImplementedError
            
        recombined_sequence = target_before + integrated_donor + target_after

    else:
        raise NotImplementedError

    return recombined_sequence