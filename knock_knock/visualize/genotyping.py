import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import knock_knock.visualize.stacked

def plot(group):
    conditions_with_enough_reads = group.total_reads()[group.total_reads() > 1000].index

    window_size = 200
    window_interval = group.target_info.around_cuts(window_size)
    marginalized_fs = knock_knock.visualize.stacked.marginalize_over_mismatches_outside_window(group.outcome_fractions(), window_interval)

    half_window = window_size // 2

    grids = {}

    for condition in conditions_with_enough_reads:
        sample_name = group.full_condition_to_sample_name[condition]
        color = group.condition_colors(unique=True)[condition]
        
        condition_fs = marginalized_fs[condition][marginalized_fs[condition] > 1e-2].sort_values(ascending=False)
        grid = knock_knock.visualize.stacked.DiagramGrid(condition_fs.index[:10],
                                                         group.target_info,
                                                         draw_wild_type_on_top=True,
                                                         window=(-half_window, half_window),
                                                         title=condition[1],
                                                         title_color=color,
                                                         block_alpha=0.2,
                                                         draw_insertion_degeneracy=False,
                                                        )
        grid.add_ax('ps')
        grid.plot_on_ax('ps', condition_fs, transform='percentage', color='black')
        grid.set_xlim('ps', (0, 100))
        grid.style_frequency_ax('ps')
    
        fig, axs = plt.subplots(1, 2,
                                figsize=(16, 3),
                                gridspec_kw=dict(hspace=0.5, width_ratios=[4, 1],),
                               )
        
        exp = group.sample_name_to_experiment(sample_name)
        expected_length = len(exp.target_info.wild_type_amplicon_sequence)
        
        ls = exp.outcome_stratified_lengths.lengths_for_relevant_reads
        fs = ls / ls.sum() * 100
        fs = pd.Series(fs)
        fs.index = fs.index - expected_length
        
        expected_length = len(exp.target_info.wild_type_amplicon_sequence)
        
        ax = axs[0]
        
        in_window = fs.loc[-half_window:half_window]
        
        ax.plot(in_window, 'o-', markersize=3, clip_on=False, color=color)

        ax.set_xlim(-half_window, half_window)
        ax.set_ylim(0, 100)
        
        ax.axhline(50, alpha=0.5, color='black')
        
        ax.axvline(0, color='black', alpha=0.5)
        
        ax.set_xlabel('Change in length from WT')
        ax.set_ylabel('% of reads')
        
        ax.set_title(f'{group.group_name}\n{sample_name}', color=color)
        
        frames = np.zeros(3)

        for x, f in fs.items():
            frames[x % 3] += f
            
        ax = axs[1]
        
        ax.plot(frames, 'o-', markersize=4, clip_on=False, color=color)
        
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 1, 2])
        
        ax.axhline(50, alpha=0.5, color='black')
        
        ax.set_xlabel('Frameshift')

        grids[condition] = grid

    return grids
        