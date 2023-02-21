import gzip
from collections import defaultdict, Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np

import hits.visualize
from hits import utilities
import knock_knock.illumina_experiment
import knock_knock.layout
import knock_knock.outcome
import knock_knock.pegRNAs
import knock_knock.prime_editing_layout
import knock_knock.twin_prime_layout
import knock_knock.Bxb1_layout

from hits.utilities import memoized_property

class PrimeEditingExperiment(knock_knock.illumina_experiment.IlluminaExperiment):
    def __init__(self, base_dir, group, sample_name, **kwargs):
        super().__init__(base_dir, group, sample_name, **kwargs)

    @property
    def read_types_to_align(self):
        return ['nonredundant']
        #return ['trimmed_by_name']

    @memoized_property
    def categorizer(self):
        return knock_knock.prime_editing_layout.Layout

    @memoized_property
    def max_relevant_length(self):
        outcomes = self.outcome_iter()
        longest_seen = max((outcome.inferred_amplicon_length for outcome in outcomes), default=0)
        return max(min(longest_seen, 600), 100)
    
    def make_nonredundant_sequence_fastq(self):
        # This is overloaded by ArrayedExperiment.
        fn = self.fns_by_read_type['fastq']['nonredundant']
        with gzip.open(fn, 'wt', compresslevel=1) as fh:
            for read in self.reads_by_type(self.preprocessed_read_type):
                fh.write(str(read))

    def generate_figures(self):
        lengths_fig = self.length_distribution_figure()
        lengths_fig.savefig(self.fns['lengths_figure'], bbox_inches='tight')
        self.generate_all_outcome_length_range_figures()
        self.generate_outcome_browser()
        self.generate_all_outcome_example_figures(num_examples=5)

    def alignment_groups_to_diagrams(self, alignment_groups, num_examples, **diagram_kwargs):
        subsample = utilities.reservoir_sample(alignment_groups, num_examples)

        for qname, als in subsample:
            layout = self.categorizer(als, self.target_info, mode=self.layout_mode)

            layout.categorize()
            
            try:
                diagram = layout.plot(title='', **diagram_kwargs)
            except:
                print(self.sample_name, qname)
                raise
                
            yield diagram

    def extract_templated_insertion_info(self):
        fields = knock_knock.prime_editing_layout.LongTemplatedInsertionOutcome.int_fields
        
        lists = defaultdict(list)

        for outcome in self.outcome_iter():
            if outcome.category == 'unintended donor integration':
                insertion_outcome = knock_knock.prime_editing_layout.LongTemplatedInsertionOutcome.from_string(outcome.details)
                
                for field in fields: 
                    value = getattr(insertion_outcome, field)
                    key = f'{outcome.category}/{outcome.subcategory}/{field}'
                    lists[key].append(value)
                            
        with h5py.File(self.fns['templated_insertion_details'], 'w') as hdf5_file:
            cat_and_subcats = {key.rsplit('/', 1)[0] for key in lists}
            read_length = 258
            for cat_and_subcat in cat_and_subcats:
                left_key = f'{cat_and_subcat}/left_insertion_query_bound'
                right_key = f'{cat_and_subcat}/right_insertion_query_bound'

                lengths = []

                for left, right in zip(lists[left_key], lists[right_key]):
                    if right == read_length - 1:
                        length = read_length
                    else:
                        length = right - left + 1

                    lengths.append(length)

                lengths_key = f'{cat_and_subcat}/insertion_length'

                lists[lengths_key] = lengths

            for key, value_list in lists.items():
                hdf5_file.create_dataset(f'{key}/list', data=np.array(value_list))

                counts = Counter(value_list)

                if len(counts) == 0:
                    values = np.array([], dtype=int)
                    counts = np.array([], dtype=int)
                else:
                    values = np.array(sorted(counts))
                    counts = np.array([counts[v] for v in values])

                hdf5_file.create_dataset(f'{key}/values', data=values)
                hdf5_file.create_dataset(f'{key}/counts', data=counts)

    def templated_insertion_details(self, category, subcategories, field):
        counts = Counter()

        if isinstance(subcategories, str):
            subcategories = [subcategories]

        with h5py.File(self.fns[f'templated_insertion_details']) as f:
            for subcategory in subcategories:
                group = f'{category}/{subcategory}/{field}'
                if group in f:
                    counts.update(dict(zip(f[group]['values'], f[group]['counts'])))

        if knock_knock.layout.NAN_INT in counts:
            counts.pop(knock_knock.layout.NAN_INT)

        if len(counts) == 0:
            xs = np.array([])
        else:
            xs = np.arange(min(counts), max(counts) + 1)

        ys = np.array([counts[x] for x in xs])

        return xs, ys

class TwinPrimeExperiment(PrimeEditingExperiment):
    @memoized_property
    def categorizer(self):
        return knock_knock.twin_prime_layout.Layout

class Bxb1TwinPrimeExperiment(TwinPrimeExperiment):
    @memoized_property
    def categorizer(self):
        return knock_knock.Bxb1_layout.Layout

class BoundaryProperties:
    def __init__(self, specific_subcategories=None):
        ''' edge_distributions: dict of Counters, outer dict keyed by (side, description), Counter keyed by position
            joint_distribution: Counter keyed by (subcategory, left position, right position)
            MH_nts_distribution: Counter keyed by number of nts of perfect microhomology at junction
            total_outcomes: number of eligibile outcomes counted to get these counts
        '''
        self.edge_distributions = defaultdict(Counter)
        self.joint_distribution = Counter()
        self.MH_nts_distribution = Counter()
        self.total_outcomes = 0
        self.specific_subcategories = specific_subcategories

    def make_joint_array(self):
        counts = Counter({(left, right): c for (key, left, right), c in self.joint_distribution.most_common()
                         if key == "left RT'ed, right RT'ed"
                        })

        self.joint_array = hits.utilities.counts_to_array(counts, dim=2) / self.total_outcomes

    def count_single_flap_boundaries(self, exp):
        ti = exp.target_info

        for outcome in exp.outcome_iter():
            if outcome.category != 'nonspecific amplification':
                self.total_outcomes += 1

            if outcome.category.startswith('unintended rejoining'):
                if self.specific_subcategories is not None:
                    if outcome.subcategory not in self.specific_subcategories:
                        continue

                ur_outcome = knock_knock.prime_editing_layout.UnintendedRejoiningOutcome.from_string(outcome.details)

                if ur_outcome.edges['left'] is not None:
                    self.edge_distributions['pegRNA', 'RT\'ed'][ur_outcome.edges['left']] += 1
                
                if ur_outcome.edges['right'] is not None:
                    self.edge_distributions['target', 'RT\'ed'][ur_outcome.edges['right']] += 1

                self.MH_nts_distribution[ur_outcome.MH_nts] += 1

                joint_key = (outcome.subcategory, ur_outcome.edges['left'], ur_outcome.edges['right'])
                self.joint_distribution[joint_key] += 1

        return self

    def count_dual_flap_boundaries(self, exp):
        ti = exp.target_info

        for outcome in exp.outcome_iter():
            if outcome.category != 'nonspecific amplification':
                self.total_outcomes += 1

            if outcome.category.startswith('unintended rejoining'):
                if self.specific_subcategories is not None:
                    if outcome.subcategory not in self.specific_subcategories:
                        continue

                ur_outcome = prime_editing_layout.UnintendedRejoiningOutcome.from_string(outcome.details)

                for side_description in outcome.subcategory.split(', '):
                    side, description = side_description.split(' ', 1)
                    self.edge_distributions[side, description][ur_outcome.edges[side]] += 1

                self.MH_nts_distribution[ur_outcome.MH_nts] += 1

                joint_key = (outcome.subcategory, ur_outcome.edges['left'], ur_outcome.edges['right'])
                self.joint_distribution[joint_key] += 1

            elif outcome.category == 'deletion':
                deletion_outcome = knock_knock.outcome.DeletionOutcome.from_string(outcome.details).undo_anchor_shift(ti.anchor)

                if ti.sequencing_direction == '+':
                    left = max(deletion_outcome.deletion.starts_ats)
                    right = min(deletion_outcome.deletion.ends_ats)
                else:
                    right = max(deletion_outcome.deletion.starts_ats)
                    left = min(deletion_outcome.deletion.ends_ats)

                relevant_edges = {}

                for side, ref_edge in [('left', left), ('right', right)]:
                    
                    target_PBS_name = ti.PBS_names_by_side_of_read[side]
                    target_PBS = ti.features[ti.target, target_PBS_name]

                    # Positive values are towards the opposite nick,
                    # negative values are away from the opposite nick.

                    if target_PBS.strand == '+':
                        relevant_edges[side] = ref_edge - target_PBS.end
                    else:
                        relevant_edges[side] = target_PBS.start - ref_edge

                    self.edge_distributions[side, 'deletion'][relevant_edges[side]] += 1

                joint_key = ('deletion',  relevant_edges['left'], relevant_edges['right'])
                self.joint_distribution[joint_key] += 1

        return self

    def get_xs_and_ys(self, side, subcategory_key, cumulative=False, normalize=False):
        counts = self.edge_distributions[side, subcategory_key]

        if len(counts) == 0:
            return None, None

        xs = np.arange(min(counts), max(counts) + 1)
        ys = np.array([counts[x] for x in xs]) / self.total_outcomes * 100

        if normalize:
            ys = ys / ys.sum()

        if cumulative:
            if subcategory_key in ("not RT'ed", 'deletion'):
                ys = np.cumsum(ys[::-1])[::-1]
            elif subcategory_key == "RT\'ed":
                ys = np.cumsum(ys)
            else:
                ys = ys

        return xs, ys

def plot_single_flap_extension_chain_edges(ti,
                                           guide_sets,
                                           normalize=False,
                                          ):
    features = ti.features

    # Common parameters.
    ref_bar_height = 0.02
    feature_height = 0.03

    marker_size = 2

    figsize = (16, 6)

    # just RT'ed
    subcategory_key = "RT\'ed"

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    xs = {}
    ys = {}
    for ax_col, side in zip(axs.T, ['pegRNA', 'target']):
        for ax, cumulative in zip(ax_col, [True, False]):
            for set_name, set_details in guide_sets.items():
                color = set_details['color']

                xs[side], ys[side] = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

                if xs[side] is None:
                    continue

                ax.plot(xs[side], ys[side], 'o-', label=set_name, markersize=marker_size, color=color, alpha=0.5)

    ax = axs[1, 0]
    pegRNA_name = ti.pegRNA_names[0]

    # By definition, the end of the PBS on this side's pegRNA 
    # is zero in the coordinate system.
    PBS_end = features[pegRNA_name, 'PBS'].end

    y_start = -0.2

    for feature_name in ['PBS', 'RTT', 'scaffold', 'protospacer']:
        feature = features[pegRNA_name, feature_name]

        # Moving back from the PBS end is moving
        # forward in the coordinate system.
        start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

        ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                   facecolor=feature.attribute['color'],
                   alpha=0.75,
                   clip_on=False,
                  )

        ax.annotate(feature_name,
                    xy=(np.mean([start, end]), y_start),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -5),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color=feature.attribute['color'],
                    annotation_clip=False,
                    weight='bold',
                   )

    pegRNA_length = len(ti.reference_sequences[pegRNA_name])

    start = PBS_end - pegRNA_length - 0.5
    end = PBS_end + 0.5
    ax.axvspan(start, end, y_start, y_start + ref_bar_height,
               facecolor='C1',
               clip_on=False,
              )

    for ax in axs[:, 0]:
        ax.set_xlim(start - 10, end + 10)
        ax.set_ylim(0)

    axs[0, 0].set_title('pegRNA', color='C1')

    axs[0, 1].set_xticklabels([])

    ax = axs[1, 1]

    colors = {
    }

    for primer_name in ti.primer_names:
        colors[primer_name] = 'grey'

    PBS = features[ti.target, knock_knock.pegRNAs.PBS_name(pegRNA_name)]

    # By definition, the edge of the PBS adjacent to the nick in the target
    # for the pegRNA is zero in the coordinate system.

    feature_names = ti.protospacer_names + list(ti.PAM_features) + ti.primer_names

    if PBS.strand == '+':
        x_min = ti.amplicon_interval.start - PBS.end - 0.5
        x_max = ti.amplicon_interval.end - PBS.end + 0.5
    else:
        x_min = PBS.start - ti.amplicon_interval.end - 0.5
        x_max = PBS.start - ti.amplicon_interval.start + 0.5

    for feature_name in feature_names:
        feature = features[ti.target, feature_name]
        
        if PBS.strand == '+':
            start, end = feature.start - PBS.end - 0.5, feature.end - PBS.end + 0.5
        else:
            start, end = PBS.start - feature.end - 0.5, PBS.start - feature.start + 0.5

        if 'PBS' in feature_name:
            height = 0.015
        else:
            height = 0.03
        
        color = colors.get(feature_name, feature.attribute['color'])

        ax.axvspan(start, end,
                   y_start, y_start - height,
                   facecolor=color,
                   clip_on=False,
                  )

        if 'PAM' in feature_name:
            label = None
        elif 'protospacer' in feature_name:
            if feature_name.startswith(pegRNA_name):
                label = 'pegRNA\nprotospacer'
            else:
                label = 'nicking\nprotospacer'
        else:
            label = feature_name

        ax.annotate(label,
                    xy=(np.mean([start, end]), y_start),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color=color,
                    annotation_clip=False,
                    weight='bold',
                   )


    ax.axvspan(x_min, x_max, y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

    for cut_after_name, cut_after in ti.cut_afters.items():
        if PBS.strand == '+':
            x = cut_after - PBS.end
        else:
            x = PBS.start - cut_after

        name, strand = cut_after_name.rsplit('_', 1)

        ref_y = y_start + 0.5 * ref_bar_height
        cut_y_bottom = ref_y - feature_height
        cut_y_middle = ref_y
        cut_y_top = ref_y + feature_height

        if (strand == '+' and ti.sequencing_direction == '+') or (strand == '-' and ti.sequencing_direction == '-'):
            ys = [cut_y_middle, cut_y_top]
        elif (strand == '-' and ti.sequencing_direction == '+') or (strand == '+' and ti.sequencing_direction == '-'):
            ys = [cut_y_bottom, cut_y_middle]
        else:
            raise ValueError

        ax.plot([x, x],
                ys,
                '-',
                linewidth=1,
                color='black',
                solid_capstyle='butt',
                zorder=10,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
               )

    for ax in axs[:, 1]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0)

    axs[0, 1].set_title('genome', color='C0')

    axs[0, 0].set_xticklabels([])
    axs[0, 0].legend()

    axs[0, 0].set_ylabel('Cumulative\npercentage of reads', size=12)
    axs[1, 0].set_ylabel('Percentage of reads', size=12)

    return fig

def plot_dual_flap_extension_chain_edges(ti,
                               guide_sets,
                               cumulative=False,
                               normalize=False,
                               x_lims=(-100, 150),
                              ):
    features = ti.features

    # Common parameters.
    ref_bar_height = 0.02
    feature_height = 0.03
    gap_between_refs = 0.01

    if cumulative:
        marker_size = 2
    else:
        marker_size = 3

    figsize = (16, 4)

    # not RT'ed and deletion
    for subcategory_key in [
        "deletion",
        "not RT'ed",
    ]:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for ax, side in zip(axs, ['left', 'right']):
            for set_name, set_details in guide_sets.items():
                color = set_details['color']

                xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

                if xs is None:
                    continue

                ax.plot(xs, ys, 'o-', label=set_name, markersize=marker_size, color=color, alpha=0.5)

            pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
            protospacer_name = knock_knock.pegRNAs.protospacer_name(pegRNA_name)

            PBS_name = ti.PBS_names_by_side_of_read[side]
            PBS = features[ti.target, PBS_name]

            other_pegRNA_name = ti.pegRNA_names_by_side_of_read[twin_prime_layout.other_side[side]]
            other_PBS_name = ti.PBS_names_by_side_of_read[twin_prime_layout.other_side[side]]
            other_protospacer_name = knock_knock.pegRNAs.protospacer_name(other_pegRNA_name)
            PAM_name = f'{protospacer_name}_PAM'
            other_PAM_name = f'{other_protospacer_name}_PAM'

            colors = {
                protospacer_name: hits.visualize.apply_alpha(ti.pegRNA_name_to_color[pegRNA_name], 0.5),
                other_protospacer_name: hits.visualize.apply_alpha(ti.pegRNA_name_to_color[other_pegRNA_name], 0.5),
                PAM_name: ti.pegRNA_name_to_color[pegRNA_name],
                other_PAM_name: ti.pegRNA_name_to_color[other_pegRNA_name],
            }

            for primer_name in ti.primer_names:
                colors[primer_name] = 'lightgrey'

            # By definition, the edge of the PBS adjacent to the nick in the target
            # for this side's pegRNA is zero in the coordinate system.

            y_start = -0.1

            feature_names = [
                protospacer_name,
                other_protospacer_name,
                PBS_name, other_PBS_name,
                PAM_name,
                other_PAM_name
            ] + ti.primer_names

            for feature_name in feature_names:
                feature = features[ti.target, feature_name]
                
                # Moving towards the other nicks is moving
                # forward in the coordinate system.
                if PBS.strand == '+':
                    start, end = feature.start - PBS.end - 0.5, feature.end - PBS.end + 0.5
                else:
                    start, end = PBS.start - feature.end - 0.5, PBS.start - feature.start + 0.5

                if 'PBS' in feature_name:
                    height = 0.015
                else:
                    height = 0.03
                
                ax.axvspan(start, end,
                           y_start, y_start - height,
                           facecolor=colors.get(feature_name, feature.attribute['color']),
                           clip_on=False,
                          )

            ax.axvspan(x_lims[0], x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

            for cut_after_name, cut_after in ti.cut_afters.items():
                if PBS.strand == '+':
                    x = cut_after - PBS.end
                else:
                    x = PBS.start - cut_after

                name, strand = cut_after_name.rsplit('_', 1)

                ref_y = y_start + 0.5 * ref_bar_height
                cut_y_bottom = ref_y - feature_height
                cut_y_middle = ref_y
                cut_y_top = ref_y + feature_height

                if (strand == '+' and ti.sequencing_direction == '+') or (strand == '-' and ti.sequencing_direction == '-'):
                    ys = [cut_y_middle, cut_y_top]
                elif (strand == '-' and ti.sequencing_direction == '+') or (strand == '+' and ti.sequencing_direction == '-'):
                    ys = [cut_y_bottom, cut_y_middle]
                else:
                    raise ValueError

                ax.plot([x, x],
                        ys,
                        '-',
                        linewidth=1,
                        color='black',
                        solid_capstyle='butt',
                        zorder=10,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False,
                       )

            ax.set_xlim(*x_lims)
            ax.set_ylim(0)

            ax.set_title(f'{side} {subcategory_key}')

            ax.set_xticklabels([])

            if side == 'right':
                ax.invert_xaxis()

        axs[0].legend()

        if cumulative:
            axs[0].set_ylabel('Cumulative percentage of reads', size=12)
        else:
            axs[0].set_ylabel('Percentage of reads', size=12)

    # just RT'ed
    subcategory_key = "RT\'ed"

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for ax, side in zip(axs, ['left', 'right']):
        for set_name, set_details in guide_sets.items():
            color = set_details['color']

            xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

            if xs is None:
                continue

            ax.plot(xs, ys, 'o-', label=set_name, markersize=marker_size, color=color, alpha=0.5)

        pegRNA_name = ti.pegRNA_names_by_side_of_read[side]

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = features[pegRNA_name, 'PBS'].end

        y_start = -0.1

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                       facecolor=ti.pegRNA_name_to_color[pegRNA_name],
                       clip_on=False,
                      )
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                       facecolor=feature.attribute['color'],
                       alpha=0.75,
                       clip_on=False,
                      )

            if feature_name == 'overlap':
                for x in [start, end]:
                    ax.axvline(x, color=feature.attribute['color'])

        ax.set_xlim(*x_lims)
        ax.set_ylim(0)

        ax.set_title(f'{side} {subcategory_key}')

        ax.set_xticklabels([])
            
        if side == 'right':
            ax.invert_xaxis()

    axs[0].legend()

    if cumulative:
        axs[0].set_ylabel('Cumulative percentage of reads', size=12)
    else:
        axs[0].set_ylabel('Percentage of reads', size=12)

    # overlap-extended
    subcategory_key = "RT'ed + overlap-extended"

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for ax, side in zip(axs, ['left', 'right']):
        for set_name, set_details in guide_sets.items():
            color = set_details['color']

            xs, ys = set_details['results'].get_xs_and_ys(side, subcategory_key, cumulative=cumulative, normalize=normalize)

            if xs is None:
                continue

            ax.plot(xs, ys, '.-', label=set_name, color=color, alpha=0.5)

        pegRNA_name = ti.pegRNA_names_by_side_of_read[side]
        other_pegRNA_name = ti.pegRNA_names_by_side_of_read[twin_prime_layout.other_side[side]]

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = features[pegRNA_name, 'PBS'].end

        y_start = -0.2

        for feature_name in ['PBS', 'RTT', 'overlap']:
            feature = features[pegRNA_name, feature_name]
            
            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=ti.pegRNA_name_to_color[pegRNA_name], clip_on=False)
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
            
        # The left side of the pegRNA overlap in the coordinate system is the 
        # end of the overlap feature on this side's pegRNA.
        overlap_start = PBS_end - features[pegRNA_name, 'overlap'].end

        other_overlap = features[other_pegRNA_name, 'overlap']

        overlap_start_offset = overlap_start - other_overlap.start

        y_start = y_start + ref_bar_height + feature_height + gap_between_refs

        for feature_name in ['PBS', 'RTT', 'overlap']:
            feature = features[other_pegRNA_name, feature_name]
            
            start, end = overlap_start_offset + feature.start - 0.5, overlap_start_offset + feature.end + 0.5
            
            ax.axvspan(start, end, y_start, y_start + ref_bar_height, facecolor=ti.pegRNA_name_to_color[other_pegRNA_name], clip_on=False)
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height, facecolor=feature.attribute['color'], alpha=0.75, clip_on=False)
            
        other_PBS_name = ti.PBS_names_by_side_of_read[twin_prime_layout.other_side[side]]
        other_protospacer_name = knock_knock.pegRNAs.protospacer_name(other_pegRNA_name)
        other_PBS_target = features[ti.target, other_PBS_name]
            
        other_PBS_start_offset = overlap_start_offset + features[other_pegRNA_name, 'PBS'].start

        y_start = y_start + ref_bar_height + feature_height + gap_between_refs

        for feature_name in [other_protospacer_name,
                                other_PBS_name,
                                ti.primers_by_side_of_read[twin_prime_layout.other_side[side]].ID,
                            ]:
            feature = features[ti.target, feature_name]
            
            if other_PBS_target.strand == '+':
                start, end = other_PBS_start_offset + (other_PBS_target.end - feature.end) - 0.5, other_PBS_start_offset + (other_PBS_target.end - feature.start) + 0.5
            else:
                start, end = other_PBS_start_offset + (feature.start - other_PBS_target.start) - 0.5, other_PBS_start_offset + (feature.end - other_PBS_target.start) + 0.5
                
            start = max(start, other_PBS_start_offset - 0.5)
            
            if feature_name == other_PBS_name:
                height = 0.015
            else:
                height = 0.03
            
            ax.axvspan(start, end,
                        y_start + ref_bar_height,
                        y_start + ref_bar_height + height,
                        facecolor=colors.get(feature_name, feature.attribute['color']),
                        clip_on=False,
                        )

        ax.axvspan(other_PBS_start_offset - 0.5, x_lims[1], y_start, y_start + ref_bar_height, facecolor='C0', clip_on=False)

        ax.set_xlim(*x_lims)
        ax.set_ylim(0)

        ax.set_title(f'{side} {subcategory_key}')

        ax.set_xticklabels([])
            
        if side == 'right':
            ax.invert_xaxis()

    axs[0].legend()

    if cumulative:
        axs[0].set_ylabel('Cumulative percentage of reads', size=12)
    else:
        axs[0].set_ylabel('Percentage of reads', size=12)

def plot_joint_RT_edges(target_info, exp_sets, v_max=0.4):
    features = target_info.features

    for set_name in exp_sets:
        results = exp_sets[set_name]['results']
        results.make_joint_array()
        percentages_array = results.joint_array * 100
        
        max_marginal = max(percentages_array.sum(axis=0).max(), percentages_array.sum(axis=1).max())

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(percentages_array, cmap=hits.visualize.blues, vmin=0, vmax=v_max)
        fig.suptitle(set_name)
        
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        
        ax_p = ax.get_position()
        
        top_ax = fig.add_axes((ax_p.x0, ax_p.y1, ax_p.width, ax_p.height * 0.25), sharex=ax)
        
        xs = np.arange(percentages_array.shape[1])
        ys = percentages_array.sum(axis=0)
        top_ax.plot(xs, ys, 'o-', markersize=2)
        plt.setp(top_ax.get_xticklabels(), visible=False)
        top_ax.spines[['top', 'right']].set_visible(False)
        top_ax.tick_params(axis='x', length=0)
        
        top_ax.set_ylim(0, max_marginal * 1.1)
        
        right_ax = fig.add_axes((ax_p.x1, ax_p.y0, ax_p.width * 0.25, ax_p.height), sharey=ax)
        
        ys = np.arange(percentages_array.shape[0])
        xs = percentages_array.sum(axis=1)
        right_ax.plot(xs, ys, 'o-', markersize=2)
        plt.setp(right_ax.get_yticklabels(), visible=False)
        right_ax.spines[['top', 'right']].set_visible(False)
        right_ax.set_xlim(0, max_marginal * 1.1)
        right_ax.tick_params(axis='y', length=0)
        
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 90)
        
        ref_bar_height = 0.025
        feature_height = 0.025
        
        pegRNA_name = target_info.pegRNA_names_by_side_of_read['right']

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        PBS_end = features[pegRNA_name, 'PBS'].end

        y_start = -3 * (ref_bar_height + feature_height)

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

            ax.axvspan(start, end, y_start, y_start + ref_bar_height,
                    facecolor=target_info.pegRNA_name_to_color[pegRNA_name],
                    clip_on=False,
                    )
            
            ax.axvspan(start, end, y_start + ref_bar_height, y_start + ref_bar_height + feature_height,
                    facecolor=feature.attribute['color'],
                    alpha=0.75,
                    clip_on=False,
                    )
            
            if feature_name == 'overlap':
                ax.axvline(start, color=feature.attribute['color'])
                ax.axvline(end, color=feature.attribute['color'])
            
        pegRNA_name = target_info.pegRNA_names_by_side_of_read['left']

        # By definition, the end of the PBS on this side's pegRNA 
        # is zero in the coordinate system.
        
        PBS_end = features[pegRNA_name, 'PBS'].end

        x_start = -2 * (ref_bar_height + feature_height)

        for feature_name in ['PBS', 'RTT', 'overlap', 'scaffold', 'protospacer']:
            feature = features[pegRNA_name, feature_name]

            # On this side's pegRNA, moving back from the PBS end is moving
            # forward in the coordinate system.
            start, end = PBS_end - feature.end - 0.5, PBS_end - feature.start + 0.5

            ax.axhspan(start, end, x_start, x_start - ref_bar_height, 
                    facecolor=target_info.pegRNA_name_to_color[pegRNA_name],
                    clip_on=False,
                    )
            
            ax.axhspan(start, end, x_start - ref_bar_height, x_start - ref_bar_height - feature_height, 
                    facecolor=feature.attribute['color'],
                    alpha=0.75,
                    clip_on=False,
                    )

            if feature_name == 'overlap':
                ax.axhline(start, color=feature.attribute['color'])
                ax.axhline(end, color=feature.attribute['color'])