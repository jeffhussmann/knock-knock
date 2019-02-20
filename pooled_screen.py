import shutil
import bisect
import pickle
import functools
from collections import Counter, defaultdict
from pathlib import Path

import scipy.sparse
import pandas as pd
import numpy as np
import yaml
import ipywidgets
import pysam

from sequencing import utilities, sam, fastq, mapping_tools, annotation
from knockin import experiment, target_info, collapse, coherence, pooled_layout, visualize

memoized_property = utilities.memoized_property

class SingleGuideExperiment(experiment.Experiment):
    def __init__(self, *args, **kwargs):
        super(SingleGuideExperiment, self).__init__(*args, **kwargs)
        
        self.fns.update({
            'R2': self.data_dir / 'by_guide' / '{}_R2.fastq.gz'.format(self.name),
            'collapsed_R2': self.dir / 'collapsed_R2.fastq',
            'collapsed_uncommon_R2': self.dir / 'collapsed_uncommon_R2.fastq',
            'guide_mismatch_rates': self.dir / 'guide_mismatch_rates.txt',

            'supplemental_STAR_prefix': lambda name: self.dir / '{}_alignments_STAR.'.format(name),
            'supplemental_bam': lambda name: self.dir / '{}_alignments.bam'.format(name),
            'supplemental_bam_by_name': lambda name: self.dir / '{}_alignments.by_name.bam'.format(name),

            'combined_bam': self.dir / 'combined.bam',
            'combined_bam_by_name': self.dir / 'combined.by_name.bam',

            'collapsed_UMI_outcomes': self.dir / 'collapsed_UMI_outcomes.txt',
            'cell_outcomes': self.dir / 'cell_outcomes.txt',
            'filtered_cell_outcomes': self.dir / 'filtered_cell_outcomes.txt',

            'filtered_cell_bam': self.dir / 'filtered_cell_aligments.bam',
            'reads_per_UMI': self.dir / 'reads_per_UMI.pkl',
            
            'genomic_insertions_bam': lambda name: self.dir / '{}_genomic_insertions.bam'.format(name),
        })
        
        self.layout_module = pooled_layout
        self.max_insertion_length = None
        self.max_qual = 41

        self.supplemental_indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'bosTau7': '/nvme/indices/bosTau7',
        }

        self.supplemental_headers = {n: sam.header_from_STAR_index(p) for n, p in self.supplemental_indices.items()}

        self.min_reads_per_cluster = 2
        self.min_reads_per_UMI = 4

        self.pool = PooledScreen(self.base_dir, self.group)
        self.use_memoized_outcomes = kwargs.get('use_memoized_outcomes', True)

    @property
    def reads(self):
        if not self.fns['R2'].exists():
            return []

        reads = fastq.reads(self.fns['R2'], up_to_space=True)
        reads = self.progress(reads)

        return reads

    def get_read_alignments(self, read_id, fn_key='filtered_cell_bam_by_name', outcome=None, check_for_common=True):
        looked_up_common = False

        if check_for_common:
            seq = self.names_with_common_seq.get(read_id)
            if seq is not None:
                als = self.pool.get_common_seq_alignments(seq)
                looked_up_common = True
            
        if not looked_up_common:
            als = super().get_read_alignments(read_id, fn_key=fn_key, outcome=outcome)

        return als
    
    def collapse_UMI_reads(self):
        ''' Takes R2_fn sorted by UMI and collapses reads with the same UMI and
        sufficiently similar sequence.
        '''

        def UMI_key(read):
            return collapse.Annotations['UMI_guide'].from_identifier(read.name)['UMI']

        def num_reads_key(read):
            return collapse.Annotations['collapsed_UMI'].from_identifier(read.name)['num_reads']

        R1_read_length = 45

        mismatch_counts = np.zeros(R1_read_length)
        total = 0

        expected_seq = self.pool.guides_df.loc[self.name, 'full_seq'][:R1_read_length]

        with self.fns['collapsed_R2'].open('w') as collapsed_fh:
            groups = utilities.group_by(self.reads, UMI_key)
            for UMI, UMI_group in groups:
                clusters = collapse.form_clusters(UMI_group, max_read_length=None, max_hq_mismatches=0)
                clusters = sorted(clusters, key=num_reads_key, reverse=True)

                for i, cluster in enumerate(clusters):
                    annotation = collapse.Annotations['collapsed_UMI'].from_identifier(cluster.name)
                    annotation['UMI'] = UMI
                    annotation['cluster_id'] = i

                    if annotation['num_reads'] >= self.min_reads_per_cluster:
                        total += 1
                        guide = annotation['guide']
                        if guide == expected_seq:
                            mismatch = -1
                        else:
                            qs = fastq.decode_sanger(annotation['guide_qual'])
                            mismatches = []
                            for i, (seen, expected, q) in enumerate(zip(guide, expected_seq, qs)):
                                if seen != expected and q >= 30:
                                    mismatches.append(i)

                            if len(mismatches) == 0:
                                mismatch = -1
                            elif len(mismatches) == 1:
                                mismatch = mismatches[0]
                            elif len(mismatches) > 1:
                                continue

                            mismatch_counts[mismatch] += 1

                        mismatch_annotation = collapse.Annotations['collapsed_UMI_mismatch'](annotation)
                        mismatch_annotation['mismatch'] = mismatch

                        cluster.name = str(mismatch_annotation)

                        collapsed_fh.write(str(cluster))

        mismatch_rates = mismatch_counts / (max(total, 1))
        np.savetxt(self.fns['guide_mismatch_rates'], mismatch_rates)

        return total
    
    @property
    def collapsed_reads(self):
        return self.progress(fastq.reads(self.fns['collapsed_R2']))

    def make_uncommon_sequence_fastq(self):
        with self.fns['collapsed_uncommon_R2'].open('w') as fh:
            for read in self.collapsed_reads:
                if read.seq not in self.pool.common_sequence_to_outcome:
                    fh.write(str(read))

    @memoized_property
    def names_with_common_seq(self):
        names = {}
        for read in self.collapsed_reads:
            if read.seq in self.pool.common_sequence_to_outcome:
                names[read.name] = read.seq
        return names

    @property
    def collapsed_uncommon_reads(self):
        return self.progress(fastq.reads(self.fns['collapsed_uncommon_R2']))

    def generate_supplemental_alignments(self):
        ''' Use STAR to produce local alignments, post-filtering spurious alignmnents.
        '''

        if self.use_memoized_outcomes:
            reads_fn_key = 'collapsed_uncommon_R2'
        else:
            reads_fn_key = 'collapsed_R2'
        
        for index_name, index in self.supplemental_indices.items():
            bam_fn = mapping_tools.map_STAR(self.fns[reads_fn_key],
                                            index,
                                            self.fns['supplemental_STAR_prefix'](index_name),
                                            sort=False,
                                            mode='permissive',
                                           )

            all_mappings = pysam.AlignmentFile(bam_fn)
            header = all_mappings.header
            new_references = ['{}_{}'.format(index_name, ref) for ref in header.references]
            new_header = pysam.AlignmentHeader.from_references(new_references, header.lengths)
            filtered_fn = str(self.fns['supplemental_bam_by_name'](index_name))

            with pysam.AlignmentFile(filtered_fn, 'wb', header=new_header) as fh:
                for al in all_mappings:
                    if al.query_alignment_length <= 20:
                        continue

                    if al.get_tag('AS') / al.query_alignment_length <= 0.8:
                        continue

                    fh.write(al)

            sam.sort_bam(self.fns['supplemental_bam_by_name'](index_name),
                         self.fns['supplemental_bam'](index_name),
                        )

    def combine_alignments(self):
        supplemental_fns = [self.fns['supplemental_bam'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam']] + supplemental_fns,
                                   self.fns['combined_bam'],
                                  )

        supplemental_fns = [self.fns['supplemental_bam_by_name'](index_name) for index_name in self.supplemental_indices]
        sam.merge_sorted_bam_files([self.fns['bam_by_name']] + supplemental_fns,
                                   self.fns['combined_bam_by_name'],
                                   by_name=True,
                                  )
        
    def categorize_outcomes(self):
        if self.fns['outcomes_dir'].is_dir():
            shutil.rmtree(str(self.fns['outcomes_dir']))

        self.fns['outcomes_dir'].mkdir()

        outcomes = defaultdict(list)

        total = 0
        required_sw = 0

        # iter wrap since tqdm objects are not iterators
        alignment_groups = iter(self.alignment_groups('combined_bam_by_name'))
        reads = self.collapsed_reads

        if self.use_memoized_outcomes:
            lookup = self.pool.common_sequence_to_outcome
        else:
            lookup = {}

        with self.fns['outcome_list'].open('w') as fh:
            for read in reads:
                if read.seq in lookup:
                    category, subcategory, details = lookup[read.seq]
                else:
                    name, als = next(alignment_groups)
                    if name != read.name:
                        raise ValueError('iters out of sync', name, read.name)

                    layout = self.layout_module.Layout(als, self.target_info, self.supplemental_headers)
                    total += 1
                    try:
                        category, subcategory, details = layout.categorize()
                    except:
                        print()
                        print(self.name, name)
                        raise
                
                    if layout.required_sw:
                        required_sw += 1

                    outcomes[category, subcategory].append(read.name)

                annotation = collapse.Annotations['collapsed_UMI_mismatch'].from_identifier(read.name)

                if category in ['uncategorized', 'SD-MMEJ'] and not self.use_memoized_outcomes:
                    if int(annotation['UMI']) < 1000: 
                        details = '{},{}_{}'.format(details, annotation['UMI'], annotation['num_reads'])

                UMI_outcome = coherence.Pooled_UMI_Outcome(annotation['UMI'],
                                                           annotation['mismatch'],
                                                           annotation['cluster_id'],
                                                           annotation['num_reads'],
                                                           category,
                                                           subcategory,
                                                           details,
                                                           read.name,
                                                          )
                fh.write(str(UMI_outcome) + '\n')

        # To make plotting easier, for each outcome, make a file listing all of
        # qnames for the outcome and a bam file (sorted by name) with all of the
        # alignments for these qnames.

        qname_to_outcome = {}
        bam_fhs = {}

        with pysam.AlignmentFile(str(self.fns['combined_bam_by_name'])) as full_bam_fh:
        
            for outcome, qnames in outcomes.items():
                outcome_fns = self.outcome_fns(outcome)
                outcome_fns['dir'].mkdir()
                bam_fhs[outcome] = pysam.AlignmentFile(str(outcome_fns['bam_by_name']), 'w', template=full_bam_fh)
                
                with outcome_fns['query_names'].open('w') as fh:
                    for qname in qnames:
                        qname_to_outcome[qname] = outcome
                        fh.write(qname + '\n')
            
            for al in full_bam_fh:
                if al.query_name in qname_to_outcome:
                    outcome = qname_to_outcome[al.query_name]
                    bam_fhs[outcome].write(al)

        for outcome, fh in bam_fhs.items():
            fh.close()

    @memoized_property
    def outcome_counts(self):
        counts = pd.read_table(self.fns['outcome_counts'],
                               header=None,
                               index_col=[0, 1, 2, 3],
                               squeeze=True,
                               na_filter=False,
                              )
        counts.index.names = ['perfect_guide', 'category', 'subcategory', 'details']
        return counts
    
    @memoized_property
    def perfect_guide_outcome_counts(self):
        return self.outcome_counts.xs(True)
    
    def collapse_UMI_outcomes(self):
        all_collapsed_outcomes, most_abundant_outcomes = coherence.collapse_pooled_UMI_outcomes(self.fns['outcome_list'])
        with self.fns['collapsed_UMI_outcomes'].open('w') as fh:
            for outcome in all_collapsed_outcomes:
                fh.write(str(outcome) + '\n')
        
        with self.fns['cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                fh.write(str(outcome) + '\n')
        
        counts = Counter()
        with self.fns['filtered_cell_outcomes'].open('w') as fh:
            for outcome in most_abundant_outcomes:
                if outcome.num_reads >= self.min_reads_per_UMI:
                    fh.write(str(outcome) + '\n')
                    perfect = (outcome.guide_mismatch == -1)
                    counts[perfect, outcome.category, outcome.subcategory, outcome.details] += 1

        counts = pd.Series(counts).sort_values(ascending=False)
        counts.to_csv(self.fns['outcome_counts'], sep='\t')
        
    def make_filtered_cell_bams(self):
        # Make bams containing only alignments from final cell assignments for IGV browsing.
        cells = self.filtered_cell_outcomes
        name_to_outcome = {}
        for _, row in cells.query('guide_mismatch == -1').iterrows():
            name_to_outcome[row['original_name']] = (row['category'], row['subcategory'])

        outcomes_seen = cells.groupby(by=['category', 'subcategory']).size().index.values

        # Because of common outcome memoization, outcome dirs may not exist for every outcome.
        for outcome in outcomes_seen:
            self.outcome_fns(outcome)['dir'].mkdir(exist_ok=True)

        with pysam.AlignmentFile(str(self.fns['combined_bam'])) as combined_bam_fh:
            header = combined_bam_fh.header

            sorters = {'all': sam.AlignmentSorter(self.fns['filtered_cell_bam'], header)}

            for outcome in outcomes_seen:
                bam_fn = self.outcome_fns(outcome)['filtered_cell_bam']
                sorters[outcome] = sam.AlignmentSorter(bam_fn, header)

            with sam.multiple_AlignmentSorters(list(sorters.values())):
                for alignment in self.progress(combined_bam_fh):
                    outcome = name_to_outcome.get(alignment.query_name)
                    if outcome is not None:
                        sorters['all'].write(alignment)
                        sorters[outcome].write(alignment)

        for outcome in outcomes_seen:
            in_fn = self.outcome_fns(outcome)['filtered_cell_bam']
            out_fn = self.outcome_fns(outcome)['filtered_cell_bam_by_name']
            sam.sort_bam(in_fn, out_fn, by_name=True)

    def make_genomic_insertion_bams(self):
        headers = self.supplemental_headers

        sorters = {}
        for organism, header in headers.items():
            fn = self.fns['genomic_insertions_bam'](organism)
            sorters[organism] = sam.AlignmentSorter(fn, header)
            
        with sam.multiple_AlignmentSorters(sorters.values()):
            rows = self.filtered_cell_outcomes.query('category == "genomic insertion"  and guide_mismatch == -1')
            for _, row in rows.iterrows():
                name = row['original_name']
                organism = row['subcategory']
                als = self.get_read_alignments(name, outcome=('genomic insertion', organism))
                layout = pooled_layout.Layout(als, self.target_info, supplemental_headers=headers)
                try:
                    original_al = layout.genomic_insertion[0]['original_alignment']
                except:
                    print(self.name, name)
                    raise

                original_al.query_name = name + '_' + self.name
                
                sorters[organism].write(original_al)

    def make_reads_per_UMI(self):
        reads_per_UMI = {}

        for category, rows in self.cell_outcomes.groupby(by=['category', 'subcategory']):
            reads_per_UMI[category] = Counter(rows['num_reads'])

        reads_per_UMI['all', 'all'] = Counter(self.cell_outcomes['num_reads'])

        with open(str(self.fns['reads_per_UMI']), 'wb') as fh:
            pickle.dump(reads_per_UMI, fh)

    @memoized_property
    def reads_per_UMI(self):
        with open(str(self.fns['reads_per_UMI']), 'rb') as fh:
            reads_per_UMI = pickle.load(fh)
        return reads_per_UMI

    @memoized_property
    def cell_outcomes(self):
        df = pd.read_table(self.fns['cell_outcomes'], header=None, names=coherence.Pooled_UMI_Outcome.columns)
        return df

    @memoized_property
    def filtered_cell_outcomes(self):
        df = pd.read_table(self.fns['filtered_cell_outcomes'], header=None, names=coherence.Pooled_UMI_Outcome.columns)
        return df

    def generate_alignments(self):
        if self.use_memoized_outcomes:
            reads = self.collapsed_uncommon_reads
        else:
            reads = self.collapsed_reads

        super().generate_alignments(reads=reads)

    def process(self, stage):
        if stage == 0:
            self.collapse_UMI_reads()
        elif stage == 1:
            self.make_uncommon_sequence_fastq()
            self.generate_alignments()
            self.generate_supplemental_alignments()
            self.combine_alignments()
            self.categorize_outcomes()
            self.collapse_UMI_outcomes()
            self.make_reads_per_UMI()
            self.make_filtered_cell_bams()
            self.make_genomic_insertion_bams()
            #self.make_outcome_plots(num_examples=3)
        else:
            raise ValueError(stage)

class CommonSequenceExperiment(SingleGuideExperiment):
    def __init__(self, base_dir, pool_name, chunk_name, progress=None):
        if progress is None:
            self.progress = utilities.identity
        else:
            self.progress = progress

        self.pool = PooledScreen(base_dir, pool_name)
        self.name = chunk_name

        self.dir = self.pool.fns['common_sequences_dir'] / chunk_name
        self.dir.mkdir(exist_ok=True)

        self.target_info = self.pool.target_info

        self.fns = {
            'bam': self.dir / 'alignments.bam',
            'bam_by_name': self.dir / 'alignments.by_name.bam',

            'outcomes_dir': self.dir / 'outcomes',
            'outcome_counts': self.dir / 'outcome_counts.csv',
            'outcome_list': self.dir / 'outcome_list.txt',

            'collapsed_R2': self.dir / 'collapsed_R2.fastq',

            'supplemental_STAR_prefix': lambda name: self.dir / '{}_alignments_STAR.'.format(name),
            'supplemental_bam': lambda name: self.dir / '{}_alignments.bam'.format(name),
            'supplemental_bam_by_name': lambda name: self.dir / '{}_alignments.by_name.bam'.format(name),

            'combined_bam': self.dir / 'combined.bam',
            'combined_bam_by_name': self.dir / 'combined.by_name.bam',
        }
        
        self.layout_module = pooled_layout
        self.max_insertion_length = None
        self.max_qual = 41

        self.supplemental_indices = {
            'hg19': '/nvme/indices/refdata-cellranger-hg19-1.2.0/star',
            'bosTau7': '/nvme/indices/bosTau7',
        }

        self.supplemental_headers = {n: sam.header_from_STAR_index(p) for n, p in self.supplemental_indices.items()}

        self.min_reads_per_cluster = 2
        self.min_reads_per_UMI = 4

        self.use_memoized_outcomes = False
    
    def get_read_alignments(self, read_id, outcome=None):
        return super().get_read_alignments(read_id, fn_key='combined_bam_by_name', outcome=outcome, check_for_common=False)
    
    def process(self):
        try:
            self.generate_alignments()
            self.generate_supplemental_alignments()
            self.combine_alignments()
            self.categorize_outcomes()
        except:
            print(self.name)
            raise

    @memoized_property
    def outcomes(self):
        return coherence.load_UMI_outcomes(self.fns['outcome_list'], pooled=True)

def collapse_categories(df):
    possibly_collapse = ['genomic insertion', 'donor insertion']
    to_collapse = [cat for cat in possibly_collapse if cat in df.index.levels[0]]

    new_rows = {}
    
    for category in to_collapse:
        subcats = sorted({s for c, s, v in df.index.values if c == category})
        for subcat in subcats:
            to_add = df.loc[category, subcat]
            new_rows[category, subcat, 'collapsed'] = to_add.sum()

    if 'donor' in df.index.levels[0]:
        all_details = set(d for s, d in df.loc['donor'].index.values)

        for details in all_details:
            new_rows['donor', 'collapsed', details] = df.loc['donor', :, details].sum()

        to_collapse.append('donor')

    df = df.drop(to_collapse, level=0)
    new_rows = pd.DataFrame.from_dict(new_rows, orient='index')

    return pd.concat((df, new_rows))

def memoized_with_key(f):
    @functools.wraps(f)
    def memoized_f(self, key):
        attr_name = '_' + f.__name__
        if not hasattr(self, attr_name):
            setattr(self, attr_name, {})

        already_computed = getattr(self, attr_name)
        if key in already_computed:
            value = already_computed[key]
        else:
            value = f(self, key)
            already_computed[key] = value

        return value

    return memoized_f

class PooledScreen():
    def __init__(self, base_dir, group, progress=None):
        self.base_dir = Path(base_dir)
        self.group = group

        if progress is None:
            progress = utilities.identity

        self.progress = progress

        sample_sheet_fn = self.base_dir / 'data' / group / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        self.sgRNA = sample_sheet.get('sgRNA')
        self.donor = sample_sheet.get('donor')
        self.target_name = sample_sheet['target_info_prefix']
        self.target_info = target_info.TargetInfo(self.base_dir, self.target_name, self.donor, self.sgRNA)

        self.fns = {
            'guides': self.base_dir / 'guides' / 'guides.txt',

            'outcome_counts': self.base_dir / 'results' / group / 'outcome_counts.npz',
            'total_outcome_counts': self.base_dir / 'results' / group / 'total_outcome_counts.txt',
            'collapsed_outcome_counts': self.base_dir / 'results' / group / 'collapsed_outcome_counts.npz',
            'collapsed_total_outcome_counts': self.base_dir / 'results' / group / 'collapsed_total_outcome_counts.txt',

            'filtered_cell_bam': self.base_dir / 'results' / group / 'filtered_cell_alignments.bam',
            'genomic_insertions_bam': lambda name: self.base_dir / 'results' / group / '{}_{}_genomic_insertions.bam'.format(self.group, name),
            'reads_per_UMI': self.base_dir / 'reads_per_UMI.pkl',

            'quantiles': self.base_dir / 'results' / group / 'quantiles.hdf5',

            'common_sequences_dir': self.base_dir / 'results' / group / 'common_sequences',
            'common_sequences_lookup_table': self.base_dir / 'results' / group / 'common_sequences' / 'lookup.txt',
            'all_sequences': self.base_dir / 'results' / group / 'common_sequences' / '{}_all_sequences.txt'.format(group),
            'common_sequence_outcomes': self.base_dir / 'results' / group / 'common_sequences' / 'outcomes.txt',
        }

    def single_guide_experiments(self):
        for guide in self.progress(self.guides):
            yield SingleGuideExperiment(self.base_dir, self.group, guide)

    def single_guide_experiment(self, guide):
        return SingleGuideExperiment(self.base_dir, self.group, guide)

    @memoized_property
    def guides_df(self):
        guides_df = pd.read_table(self.base_dir / 'guides' / 'guides.txt', index_col='short_name')
        guides_df.loc[guides_df['promoter'].isnull(), 'promoter'] = 'P1P2'
        guides_df['best_promoter'] = True
        for gene, promoter in self.best_promoters.items():
            not_best = guides_df.query('gene == @gene and promoter != @promoter').index
            guides_df.loc[not_best, 'best_promoter'] = False
            
        return guides_df
    
    @memoized_property
    def best_promoters(self):
        df = pd.read_table(self.base_dir / 'guides' / 'best_promoters.txt', index_col='gene', squeeze=True)
        return df

    @memoized_property
    def guides(self):
        guides = self.guides_df.index.values
        return guides

    @memoized_property
    def non_targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' in g]

    @memoized_property
    def targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' not in g and 'eGFP' not in g]

    @memoized_property
    def genes(self):
        return sorted(set(self.guides_df['gene']))

    def gene_guides(self, gene, only_best_promoter=False):
        query = 'gene == @gene'
        if only_best_promoter:
            query += ' and best_promoter'

        gene_guides = self.guides_df.query(query)

        return gene_guides.index

    def guide_to_gene(self, guide):
        return self.guides_df.loc[guide]['gene']

    def make_outcome_counts(self):
        all_counts = {}
        guides = self.guides

        for guide in self.progress(guides):
            exp = SingleGuideExperiment(self.base_dir, self.group, guide)
            try:
                all_counts[guide] = exp.outcome_counts
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        all_outcomes = set()

        for guide in all_counts:
            all_outcomes.update(all_counts[guide].index.values)
            
        outcome_order = sorted(all_outcomes)
        outcome_to_index = {outcome: i for i, outcome in enumerate(outcome_order)}

        counts = scipy.sparse.dok_matrix((len(outcome_order), len(guides)), dtype=int)

        for g, guide in enumerate(self.progress(guides)):
            if guide in all_counts:
                for outcome, count in all_counts[guide].items():
                    o = outcome_to_index[outcome]
                    counts[o, g] = count
                
        scipy.sparse.save_npz(self.fns['outcome_counts'], counts.tocoo())

        df = pd.DataFrame(counts.todense(),
                          columns=guides,
                          index=pd.MultiIndex.from_tuples(outcome_order),
                         )

        df.sum(axis=1).to_csv(self.fns['total_outcome_counts'])

        # Collapse potentially equivalent outcomes together.
        collapsed = pd.concat({pg: collapse_categories(df.loc[pg]) for pg in [True, False]})

        coo = scipy.sparse.coo_matrix(np.array(collapsed))
        scipy.sparse.save_npz(self.fns['collapsed_outcome_counts'], coo)

        collapsed.sum(axis=1).to_csv(self.fns['collapsed_total_outcome_counts'])

    @memoized_with_key
    def total_outcome_counts(self, collapsed):
        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'total_outcome_counts'

        return pd.read_csv(self.fns[key], header=None, index_col=[0, 1, 2, 3], na_filter=False)

    @memoized_with_key
    def outcome_counts_df(self, collapsed):
        guides = self.guides

        if collapsed:
            prefix = 'collapsed_'
        else:
            prefix = ''

        key = prefix + 'outcome_counts'

        sparse_counts = scipy.sparse.load_npz(self.fns[key])
        df = pd.DataFrame(sparse_counts.todense(),
                          index=self.total_outcome_counts(collapsed).index,
                          columns=guides,
                         )
        df.index.names = ('perfect_guide', 'category', 'subcategory', 'details')

        return df

    @memoized_with_key
    def outcome_counts(self, guide_status):
        if guide_status == 'all':
            outcome_counts = (self.outcome_counts('perfect') + self.outcome_counts('imperfect')).fillna(0).astype(int)
        else:
            perfect_guide = guide_status == 'perfect'
            outcome_counts = self.outcome_counts_df(True).loc[perfect_guide]

        return outcome_counts

    @memoized_with_key
    def UMI_counts(self, guide_status):
        return self.outcome_counts(guide_status).sum()
    
    @memoized_with_key
    def outcome_fractions(self, guide_status):
        per_guide_fractions = self.outcome_counts(guide_status) / self.UMI_counts(guide_status)
        nt_fractions = self.non_targeting_fractions(guide_status)
        return pd.concat([per_guide_fractions, nt_fractions], axis=1)
    
    @memoized_property
    def non_targeting_outcomes(self):
        guide_outcomes = {}
        for nt_guide in self.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, nt_guide)
            fn = exp.fns['filtered_cell_outcomes']

            outcomes = [coherence.Pooled_UMI_Outcome.from_line(line) for line in fn.open()]

            for outcome in outcomes:
                if outcome.category == 'genomic insertion':
                    outcome.details = 'n/a'
                
                if outcome.category == 'donor insertion':
                    outcome.details = 'n/a'

            guide_outcomes[nt_guide] = outcomes

        return guide_outcomes

    @memoized_with_key
    def non_targeting_counts(self, guide_status):
        counts = self.outcome_counts(guide_status)[self.non_targeting_guides]
        return counts.sum(axis=1).sort_values(ascending=False)
    
    @memoized_with_key
    def non_targeting_fractions(self, guide_status):
        counts = self.non_targeting_counts(guide_status)
        fractions = counts / counts.sum()
        fractions.name = 'non_targeting'
        return fractions

    @memoized_property
    def most_frequent_outcomes(self):
        return self.non_targeting_counts('all').index.values[:1000]

    @memoized_with_key
    def common_counts(self, guide_status):
        # Regardless of guide_status, use 'all' to define common non-targeting outcomes.
        common_counts = self.outcome_counts(guide_status).loc[self.most_frequent_outcomes] 
        leftover = self.UMI_counts(guide_status) - common_counts.sum()
        leftover_row = pd.DataFrame.from_dict({('uncommon', 'uncommon', 'collapsed'): leftover}, orient='index')
        common_counts = pd.concat([common_counts, leftover_row])
        return common_counts
    
    @memoized_property
    def common_non_targeting_counts(self):
        return self.common_counts('perfect')[self.non_targeting_guides].sum(axis=1)
    
    @memoized_property
    def common_non_targeting_fractions(self):
        counts = self.common_non_targeting_counts
        return counts / counts.sum()
    
    @memoized_with_key
    def common_fractions(self, guide_status):
        return self.common_counts(guide_status) / self.UMI_counts(guide_status)

    @memoized_with_key
    def fold_changes(self, guide_status):
        fractions = self.outcome_fractions(guide_status)
        return fractions.div(fractions['non_targeting'], axis=0)

    @memoized_with_key
    def log2_fold_changes(self, guide_status):
        fc = self.fold_changes(guide_status)
        fc = fc.fillna(2**5).replace(0, 2**-5)
        return np.log2(fc)

    def rational_outcome_order(self, num_outcomes=50, include_uncommon=False, by_frequency=False):
        def get_deletion_info(details):
            deletion = target_info.degenerate_indel_from_string(details)
            return {'num_MH_nts': len(deletion.starts_ats) - 1,
                    'start': min(deletion.starts_ats),
                    'length': deletion.length,
                    }

        def has_MH(details):
            info = get_deletion_info(details)
            return info['num_MH_nts'] >= 2 and info['length'] > 1

        conditions = {
            'insertions': lambda c, sc, d: c == 'insertion',
            'no_MH_deletions': lambda c, sc, d: c == 'deletion' and not has_MH(d),
            'MH_deletions': lambda c, sc, d: c == 'deletion' and has_MH(d),
            'donor': lambda c, sc, d: c == 'donor' and sc == 'collapsed',
            'wt': lambda c, sc, d: c == 'wild type' and sc != 'mismatches' and d != '____----',
            'uncat': lambda c, sc, d: c == 'uncategorized',
            'genomic': lambda c, sc, d: c == 'genomic insertion',
            'donor insertion': lambda c, sc, d: c == 'donor insertion',
            'SD-MMEJ': lambda c, sc, d: c == 'SD-MMEJ',
            'uncommon': [('uncommon', 'uncommon', 'collapsed')],
        }

        group_order = [
            'uncat',
            'genomic',
            'donor insertion',
            'wt',
            'donor',
            'insertions',
            'no_MH_deletions',
            'MH_deletions',
            'SD-MMEJ',
        ]
        if include_uncommon:
            group_order.append('uncommon')

        donor_order = [
            'ACGAGTTT',
            '___AGTTT',
            '____GTTT',
            '___AGTT_',
            '____GTT_',
            '____GT__',
            '____G___',
            'ACGAGTT_',
            'ACGAGT__',
            'ACGAG___',
            'ACGA____',
            'ACG_____',
            'ACG_GTTT',
            'ACAAGTTT',
            '___',
            'ACG',
        ]

        groups = {
            name: [o for o in self.most_frequent_outcomes[:num_outcomes] if condition(*o)] if name != 'uncommon' else condition
            for name, condition in conditions.items()
        }

        def donor_key(csd):
            details = csd[2]
            if ';' in details:
                variable_locii_details, deletion_details = details.split(';', 1)
            else:
                variable_locii_details = details
                deletion_details = None

            if variable_locii_details in donor_order:
                i = donor_order.index(variable_locii_details)
            else:
                i = 1000
            return i, deletion_details

        def deletion_key(csd):
            details = csd[2]
            length = get_deletion_info(details)['length']
            return length

        if not by_frequency:
            groups['donor'] = sorted(groups['donor'], key=donor_key)
            for k in ['no_MH_deletions', 'MH_deletions']:
                groups[k] = sorted(groups[k], key=deletion_key)

        ordered = []
        for name in group_order:
            ordered.extend(groups[name])

        sizes = [len(groups[name]) for name in group_order]
        return ordered, sizes

    def merge_filtered_bams(self):
        input_fns = []
        for guide in self.non_targeting_guides:
            exp = SingleGuideExperiment(self.base_dir, self.group, guide)
            input_fns.append(exp.fns['filtered_cell_bam'])

        sam.merge_sorted_bam_files(input_fns, self.fns['filtered_cell_bam'])
    
    def merge_genomic_insertion_bams(self):
        exp = self.single_guide_experiment(self.guides[0])
        organisms = sorted(exp.supplemental_headers)

        input_fns = defaultdict(list)
        i = 0
        for exp in self.single_guide_experiments():
            i += 1
            if i > 10000:
                break
            for organism in organisms:
                input_fn = exp.fns['genomic_insertions_bam'](organism)
                if input_fn.exists():
                    input_fns[organism].append(input_fn)
                else:
                    print(organism, guide)

        for organism, fns in input_fns.items():
            merged_fn = self.fns['genomic_insertions_bam'](organism)
            sam.merge_sorted_bam_files(fns, merged_fn)

    def merge_reads_per_UMI(self):
        reads_per_UMI = defaultdict(Counter)

        for exp in self.progress(self.single_guide_experiments()):
            for category, counts in exp.reads_per_UMI.items():
                reads_per_UMI[category].update(counts)

        with open(str(self.fns['reads_per_UMI']), 'wb') as fh:
            pickle.dump(dict(reads_per_UMI), fh)
    
    @memoized_property
    def reads_per_UMI(self):
        with open(str(self.fns['reads_per_UMI']), 'rb') as fh:
            reads_per_UMI = pickle.load(fh)

        for category, counts in reads_per_UMI.items():
            reads_per_UMI[category] = utilities.counts_to_array(counts)

        return reads_per_UMI

    def chi_squared_per_guide(self, relevant_outcomes=None):
        if relevant_outcomes is None:
            relevant_outcomes = 50
        if isinstance(relevant_outcomes, int):
            relevant_outcomes = self.most_frequent_outcomes

        counts = self.outcome_counts('perfect').loc[relevant_outcomes]
        
        # A column with zero counts causes problems.
        guide_counts = counts.sum()
        nonzero_guides = guide_counts[guide_counts > 0].index
        counts = counts[nonzero_guides]
        
        non_targeting_guides = sorted(set(self.non_targeting_guides) & set(nonzero_guides))
        
        UMI_counts = counts.sum()
        nt_totals = counts[non_targeting_guides].sum(axis=1)
        nt_fractions = nt_totals / nt_totals.sum()
        expected = pd.DataFrame(np.outer(nt_fractions, UMI_counts), index=counts.index, columns=nonzero_guides)
        difference = counts - expected
        return (difference**2 / expected).sum().sort_values(ascending=False)

    def explore(self, **kwargs):
        return explore(self.base_dir, self.group, **kwargs)

    def make_common_sequences(self):
        splitter = CommonSequenceSplitter(self)

        Annotation = collapse.Annotations['collapsed_UMI_mismatch']
        def Read_to_num_reads(r):
            return Annotation.from_identifier(r.name)['num_reads']

        for exp in self.progress(self.single_guide_experiments()):
            enough_reads_per_UMI = (r.seq for r in fastq.reads(exp.fns['collapsed_R2']) if Read_to_num_reads(r) >= 5)
            splitter.update_counts(enough_reads_per_UMI)

        splitter.write_files()

    @memoized_property
    def common_sequence_chunk_names(self):
        return sorted([d.name for d in self.fns['common_sequences_dir'].iterdir() if d.is_dir()])

    def common_sequence_chunks(self):
        for chunk_name in self.common_sequence_chunk_names:
            yield CommonSequenceExperiment(self.base_dir, self.group, chunk_name, progress=self.progress)

    @memoized_property
    def name_to_seq(self):
        name_to_seq = {}
        with self.fns['common_sequences_lookup_table'].open() as fh:
            for line in fh:
                name, seq = line.strip().split()
                name_to_seq[name] = seq

        return name_to_seq
    
    @memoized_property
    def seq_to_name(self):
        return utilities.reverse_dictionary(self.name_to_seq)

    @memoized_property
    def name_to_chunk(self):
        names = self.common_sequence_chunk_names
        starts = [int(n.split('-')[0]) for n in names]
        chunks = [CommonSequenceExperiment(self.base_dir, self.group, n) for n in names]
        Annotation = collapse.Annotations['collapsed_UMI_mismatch']

        def name_to_chunk(name):
            number = int(Annotation.from_identifier(name)['UMI'])
            start_index = bisect.bisect(starts, number) - 1 
            chunk = chunks[start_index]
            return chunk

        return name_to_chunk

    def get_read_alignments(self, name):
        chunk = self.name_to_chunk(name)
        als = chunk.get_read_alignments(name)
        return als

    def get_common_seq_alignments(self, seq):
        name = self.seq_to_name[seq]
        als = self.get_read_alignments(name)
        return als

    @memoized_property
    def common_sequence_outcomes(self):
        outcomes = []
        for exp in self.common_sequence_chunks():
            outcomes.extend(exp.outcomes)

        return outcomes

    def build_common_sequence_to_outcome(self):
        with self.fns['common_sequence_outcomes'].open('w') as fh:
            for outcome in self.common_sequence_outcomes:
                fields = [
                    self.name_to_seq[outcome.original_name],
                    outcome.category,
                    outcome.subcategory,
                    outcome.details,
                ]
                fh.write('\t'.join(fields) + '\n')

    @memoized_property
    def common_sequence_to_outcome(self):
        common_sequence_to_outcome = {}
        with self.fns['common_sequence_outcomes'].open() as fh:
            for line in fh:
                seq, category, subcategory, details = line.strip().split('\t')
                common_sequence_to_outcome[seq] = (category, subcategory, details)

        return common_sequence_to_outcome

class CommonSequenceSplitter():
    def __init__(self, pool, reads_per_chunk=1000):
        self.pool = pool
        self.reads_per_chunk = reads_per_chunk
        self.current_chunk_fh = None
        self.seq_counts = Counter()
        
        common_sequences_dir = self.pool.fns['common_sequences_dir']

        if common_sequences_dir.is_dir():
            shutil.rmtree(str(common_sequences_dir))
            
        common_sequences_dir.mkdir()

    def update_counts(self, seqs):
        counts = Counter(seqs)
        self.seq_counts.update(counts)
            
    def close(self):
        if self.current_chunk_fh is not None:
            self.current_chunk_fh.close()
            
    def possibly_make_new_chunk(self, i):
        if i % self.reads_per_chunk == 0:
            self.close()
            chunk_name = '{:010d}-{:010d}'.format(i, i + self.reads_per_chunk - 1)
            chunk_exp = CommonSequenceExperiment(self.pool.base_dir, self.pool.group, chunk_name)
            self.current_chunk_fh = chunk_exp.fns['collapsed_R2'].open('w')
            
    def write_read(self, i, read):
        self.possibly_make_new_chunk(i)
        self.current_chunk_fh.write(str(read))
        
    def write_files(self):
        seq_lengths = {len(s) for s in self.seq_counts}
        if len(seq_lengths) > 1:
            raise ValueError('More than one sequence length: ', seq_lengths)
        seq_length = seq_lengths.pop()

        # Include one value outside of the solexa range to allow automatic detection.
        qual = fastq.encode_sanger([25] + [40] * (seq_length - 1))
   
        tuples = []

        Annotation = collapse.Annotations['collapsed_UMI_mismatch']
       
        for i, (seq, count) in enumerate(self.seq_counts.most_common()):
            name = str(Annotation(UMI='{:010}'.format(i), cluster_id=0, num_reads=count, mismatch=-1))
            tuples.append((name, seq, count))
            if count > 1:
                read = fastq.Read(name, seq, qual)
                self.write_read(i, read)

        self.close()

        with self.pool.fns['common_sequences_lookup_table'].open('w') as lookup_fh, \
             self.pool.fns['all_sequences'].open('w') as all_sequences_fh:

            for name, seq, count in tuples:
                all_sequences_fh.write('{}\t{}\n'.format(seq, count))

                if count > 1:
                    lookup_fh.write('{}\t{}\n'.format(name, seq))

def explore(base_dir, group, initial_guide=None, by_outcome=True, **kwargs):
    pool = PooledScreen(base_dir, group)

    guides = pool.guides
    if initial_guide is None:
        initial_guide = guides[0]

    widgets = {
        'guide': ipywidgets.Select(options=guides, value=initial_guide, layout=ipywidgets.Layout(height='200px', width='450px')),
        'read_id': ipywidgets.Select(options=[], layout=ipywidgets.Layout(height='200px', width='600px')),
        'outcome': ipywidgets.Select(options=[], continuous_update=False, layout=ipywidgets.Layout(height='200px', width='450px')),
        'zoom_in': ipywidgets.FloatRangeSlider(value=[-0.02, 1.02], min=-0.02, max=1.02, step=0.001, continuous_update=False, layout=ipywidgets.Layout(width='1200px')),
        'save': ipywidgets.Button(description='Save'),
        'file_name': ipywidgets.Text(value=str(Path(base_dir) / 'figures')),
    }

    toggles = [
        'parsimonious',
        'relevant',
        'ref_centric',
        'draw_sequence',
        'draw_qualities',
        'draw_mismatches',
        'highlight_SNPs',
        'highlight_around_cut',
    ]
    for toggle in toggles:
        widgets[toggle] = ipywidgets.ToggleButton(value=kwargs.get(toggle, False))

    def save(change):
        fig = interactive.result
        fn = widgets['file_name'].value
        fig.savefig(fn, bbox_inches='tight')

    widgets['save'].on_click(save)

    # For some reason, the target widget doesn't get a label without this.
    for k, v in widgets.items():
        v.description = k

    output = ipywidgets.Output()

    def get_exp():
        guide = widgets['guide'].value
        exp = SingleGuideExperiment(base_dir, group, guide)
        return exp

    @output.capture()
    def populate_outcomes(change):
        previous_value = widgets['outcome'].value

        exp = get_exp()

        outcomes = {(c, sc) for c, sc, d in exp.perfect_guide_outcome_counts.index.values}

        widgets['outcome'].options = [('_'.join(outcome), outcome) for outcome in sorted(outcomes)]
        if outcomes:
            if previous_value in outcomes:
                widgets['outcome'].value = previous_value
                populate_read_ids(None)
            else:
                widgets['outcome'].value = widgets['outcome'].options[0][1]
        else:
            widgets['outcome'].value = None

    @output.capture()
    def populate_read_ids(change):
        exp = get_exp()

        df = exp.filtered_cell_outcomes

        if exp is None:
            return

        if by_outcome:
            outcome = widgets['outcome'].value
            if outcome is None:
                qnames = []
            else:
                category, subcategory = outcome
                right_outcome = df.query('category == @category and subcategory == @subcategory and guide_mismatch == -1')
                qnames = right_outcome['original_name'].values[:200]
        else:
            qnames = df['original_name'].values[:200]

        widgets['read_id'].options = qnames

        if len(qnames) > 0:
            widgets['read_id'].value = qnames[0]
            widgets['read_id'].index = 0
        else:
            widgets['read_id'].value = None
            
    if by_outcome:
        populate_outcomes({'name': 'initial'})

    populate_read_ids({'name': 'initial'})

    if by_outcome:
        widgets['outcome'].observe(populate_read_ids, names='value')
        widgets['guide'].observe(populate_outcomes, names='value')
    else:
        widgets['guide'].observe(populate_read_ids, names='value')

    @output.capture(clear_output=True)
    def plot(guide, read_id, **plot_kwargs):
        exp = get_exp()

        if exp is None:
            return

        if by_outcome:
            als = exp.get_read_alignments(read_id, fn_key='filtered_cell_bam_by_name', outcome=plot_kwargs['outcome'])
        else:
            als = exp.get_read_alignments(read_id)

        if als is None:
            return None

        l = pooled_layout.Layout(als, exp.target_info, supplemental_headers=exp.supplemental_headers)
        info = l.categorize()
        if widgets['relevant'].value:
            als = l.relevant_alignments

        diagram = visualize.ReadDiagram(als, exp.target_info,
                                        size_multiple=kwargs.get('size_multiple', 1),
                                        max_qual=exp.max_qual,
                                        flip_target=True,
                                        target_on_top=True,
                                        features_to_hide=['ssODN_Cpf1_deletion'],
                                        **plot_kwargs)
        fig = diagram.fig

        fig.axes[0].set_title(' '.join((l.name,) + info))

        if widgets['draw_sequence'].value:
            print(exp.group, exp.name)
            print(als[0].query_name)
            print(als[0].get_forward_sequence())

        return fig

    # Make a version of the widgets dictionary that excludes non-plot arguments.
    most_widgets = widgets.copy()
    most_widgets.pop('save')
    most_widgets.pop('file_name')

    interactive = ipywidgets.interactive(plot, **most_widgets)
    interactive.update()

    def make_row(keys):
        return ipywidgets.HBox([widgets[k] for k in keys])

    if by_outcome:
        top_row_keys = ['guide', 'outcome', 'read_id']
    else:
        top_row_keys = ['guide', 'read_id']

    layout = ipywidgets.VBox(
        [make_row(top_row_keys),
         make_row(toggles),
         make_row(['save',
                   'file_name',
                  ]),
         #widgets['zoom_in'],
         interactive.children[-1],
         output,
        ],
    )

    return layout

def get_all_pools(base_dir='/home/jah/projects/britt', progress=None):
    group_dirs = [p for p in (Path(base_dir) / 'data').iterdir() if p.is_dir()]

    pools = {}

    for group_dir in group_dirs:
        name = group_dir.name

        sample_sheet_fn = group_dir / 'sample_sheet.yaml'
        if sample_sheet_fn.exists():
            sample_sheet = yaml.load(sample_sheet_fn.read_text())
            pooled = sample_sheet.get('pooled', False)
            if pooled:
                pools[name] = PooledScreen(base_dir, name, progress=progress)

    return pools
