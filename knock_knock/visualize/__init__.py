import bokeh.palettes
import copy

import numpy as np
import matplotlib.pyplot as plt

fold_changes_cmap = copy.copy(plt.get_cmap('RdBu_r'))

class ColorGroupCycler:
    def __init__(self):
        starts_20c = np.arange(4) * 4
        starts_20b = np.array([3, 1, 0, 2, 4]) * 4

        groups_20c = [bokeh.palettes.Category20c_20[start:start + 3] for start in starts_20c]
        groups_20b = [bokeh.palettes.Category20b_20[start:start + 3] for start in starts_20b]

        self.all_groups = (groups_20c + groups_20b)
        
    def __getitem__(self, key):
        group_num, replicate = key
        group = self.all_groups[group_num % len(self.all_groups)]
        color = group[replicate % len(group)]
        return color
    
color_groups = ColorGroupCycler()

def extract_color(description):
    color = description.get('color')

    if color is None or color == 0:
        color = 'grey'
    else:
        num = int(color) - 1
        replicate = int(description.get('replicate', 1)) - 1
        color = color_groups[num, replicate]

    return color
        