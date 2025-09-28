from pathlib import Path

import knock_knock.editing_strategy

base_dir = Path(__file__).resolve().parent

def summarize_features(features):
    summary = {}

    for key, feature in features.items():
        summary[key] = (feature.strand, int(feature.start), int(feature.end))
    
    return summary

def test_same_orientation():
    strat = knock_knock.editing_strategy.EditingStrategy(base_dir, 'HDR_pacbio_R_PCR')

    expected_features = {
        ('RAB11A_PAC', 'HA_5'): ('+', 1083, 1230),
        ('RAB11A_PAC', 'HA_3'): ('+', 1231, 1380),
        ('RAB11A-150HA_PCR_donor', 'HA_5'): ('+', 0, 147),
        ('RAB11A-150HA_PCR_donor', 'payload'): ('+', 148, 876),
        ('RAB11A-150HA_PCR_donor', 'HA_3'): ('+', 877, 1026),
    }
    
    assert summarize_features(strat.inferred_HA_features) == expected_features

def test_flipped_orientation():
    strat = knock_knock.editing_strategy.EditingStrategy(base_dir, 'HDR_pacbio_R_PCR')

    records = {record.name: record for record in strat.gb_records}

    flipped_donor = records[strat.donor].reverse_complement()

    flipped_donor.name = f'{records[strat.donor].name}_flipped'

    strat_with_flipped_donor = knock_knock.editing_strategy.EditingStrategy(base_dir,
                                                                            'HDR_pacbio_R_PCR',
                                                                            gb_records=[
                                                                                records[strat.target],
                                                                                flipped_donor,
                                                                            ],
                                                                            donor=flipped_donor.name,
                                                                           )


    expected_features = {
        ('RAB11A_PAC', 'HA_5'): ('-', 1228, 1380),
        ('RAB11A_PAC', 'HA_3'): ('-', 1083, 1227),
        ('RAB11A-150HA_PCR_donor_flipped', 'HA_5'): ('+', 0, 152),
        ('RAB11A-150HA_PCR_donor_flipped', 'payload'): ('+', 153, 881),
        ('RAB11A-150HA_PCR_donor_flipped', 'HA_3'): ('+', 882, 1026),
    }

    assert summarize_features(strat_with_flipped_donor.inferred_HA_features) == expected_features