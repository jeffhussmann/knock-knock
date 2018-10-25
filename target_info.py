from pathlib import Path
from collections import defaultdict
import yaml
import pysam
import Bio.SeqIO

from sequencing import fasta, gff, utilities, mapping_tools, interval

class TargetInfo(object):
    def __init__(self, base_dir, name, sgRNA=None):
        self.name = name
        self.dir = Path(base_dir) / 'targets' / name

        manifest_fn = self.dir / 'manifest.yaml'
        manifest = yaml.load(manifest_fn.open())
        self.target = manifest['target']
        self.donor = manifest.get('donor', None)
        self.sources = manifest['sources']

        self.sgRNA = sgRNA
        self.knockin = manifest.get('knockin', 'GFP') 

        self.primer_names = {
            5: manifest.get('forward primer', 'forward primer'),
            3: manifest.get('reverse primer', 'reverse primer'),
        }

        self.fns = {
            'ref_fasta': self.dir / 'refs.fasta',
            'ref_gff': self.dir / 'refs.gff',

            'bowtie2_index': self.dir / 'refs',

            'degenerate_insertions': self.dir / 'degenerate_insertions.txt',
            'degenerate_deletions': self.dir / 'degenerate_deletions.txt',
        }

    @utilities.memoized_property
    def sgRNAs(self):
        if self.sgRNA is None:
            sgRNAs = sorted(n for t, n in self.all_sgRNA_features)
        else:
            sgRNAs = [self.sgRNA]

        return sgRNAs

    def make_references(self):
        ''' Generate fasta and gff files from genbank inputs. '''
        gbs = [self.dir / (source + '.gb') for source in self.sources]
        
        fasta_records = []
        all_gff_features = []
        
        for gb in gbs:
            fasta_record, gff_features = parse_benchling_genbank(gb)
            fasta_records.append(fasta_record)
            all_gff_features.extend(gff_features)
            
        with self.fns['ref_fasta'].open('w') as fasta_fh:
            for record in fasta_records:
                fasta_fh.write(str(record))
                
        pysam.faidx(str(self.fns['ref_fasta']))
                
        with self.fns['ref_gff'].open('w') as gff_fh:
            gff_fh.write('##gff-version 3\n')
            for feature in sorted(all_gff_features):
                gff_fh.write(str(feature) + '\n')

    def make_bowtie2_index(self):
        mapping_tools.build_bowtie2_index(self.fns['bowtie2_index'], [self.fns['ref_fasta']])

    @utilities.memoized_property
    def features(self):
        features = {
            (f.seqname, f.attribute['ID']): f
            for f in gff.get_all_features(self.fns['ref_gff'])
            if 'ID' in f.attribute
        }
        return features

    @utilities.memoized_property
    def reference_sequences(self):
        if self.fns['ref_fasta'].exists():
            seqs = fasta.to_dict(self.fns['ref_fasta'])
        else:
            seqs = {}

        return seqs

    @utilities.memoized_property
    def target_sequence(self):
        return self.reference_sequences[self.target]
    
    @utilities.memoized_property
    def donor_sequence(self):
        return self.reference_sequences[self.donor]

    @utilities.memoized_property
    def PAM_range(self):
        sgRNA = self.sgRNA_feature

        if sgRNA.strand == '+':
            start = sgRNA.end + 1
            end = sgRNA.end + 3

        elif sgRNA.strand == '-':
            start = sgRNA.start - 3
            end = sgRNA.start - 1

        return start, end
    
    @utilities.memoized_property
    def PS_ranges(self):
        return [(s.start, s.end) for s in self.sgRNA_features]
        
    @utilities.memoized_property
    def sgRNA_features(self):
        return [self.features[self.target, sgRNA] for sgRNA in self.sgRNAs]

    @utilities.memoized_property
    def all_sgRNA_features(self):
        return {name: feature for name, feature in self.features.items() if feature.feature == 'sgRNA'}
    
    @utilities.memoized_property
    def cut_afters(self):
        cut_afters = []
        seq = self.target_sequence
        for feature in self.sgRNA_features:
            if feature.strand == '+':
                PAM = seq[feature.end + 1:feature.end + 4]
                cut_after = feature.end - 3

            elif feature.strand == '-':
                PAM = utilities.reverse_complement(seq[feature.start - 3:feature.start])
                cut_after = feature.start + 2

            if PAM[-2:] != 'GG':
                raise ValueError('non-NGG PAM: {0}'.format(PAM))

            cut_afters.append(cut_after)
        
        return cut_afters
    
    @utilities.memoized_property
    def cut_after(self):
        ''' when processing assumes there will be only one cut, use this '''
        if len(self.cut_afters) > 1:
            raise ValueError(self.cut_afters)
        else:
            return self.cut_afters[0]

    def around_cuts(self, each_side):
        intervals = [interval.Interval(cut_after - each_side + 1, cut_after + each_side) for cut_after in self.cut_afters]
        return interval.DisjointIntervals(intervals)

    @utilities.memoized_property
    def around_cut_features(self):
        fs = {}
        for feature, cut_after in zip(self.sgRNA_features, self.cut_afters):
            upstream = gff.Feature.from_fields(self.target, '.', '.', cut_after - 25, cut_after, '.', feature.strand, '.', '.')
            downstream = gff.Feature.from_fields(self.target, '.', '.', cut_after + 1, cut_after + 25, '.', feature.strand, '.', '.')

            distal, proximal = upstream, downstream
            if feature.strand == '+':
                distal, proximal = upstream, downstream
            else:
                distal, proximal = downstream, upstream
                
            distal.attribute['color'] = '#b7e6d7'
            proximal.attribute['color'] = '#85dae9'

            distal.attribute['ID'] = 'PAM-distal\n{} cut'.format(feature.attribute['ID'])
            proximal.attribute['ID'] = 'PAM-proximal\n{} cut'.format(feature.attribute['ID'])

            fs.update({(self.target, f.attribute['ID']): f for f in [distal, proximal]})

        return fs

    @utilities.memoized_property
    def homology_arms(self):
        HAs = {}

        for source in ['donor', 'target']:
            for side in [5, 3]:
                name = "{0}' HA".format(side)
                HAs[source, side] = self.features[getattr(self, source), name] 

        return HAs

    @utilities.memoized_property
    def HA_ref_p_to_offset(self):
        ref_p_to_offset = {}

        for name, side in self.homology_arms:
            feature = self.homology_arms[name, side]
            
            if feature.strand == '+':
                order = range(feature.start, feature.end + 1)
            else:
                order = range(feature.end, feature.start - 1, -1)
                
            ref_p_to_offset[name, side] = {ref_p: offset for offset, ref_p in enumerate(order)}  

        return ref_p_to_offset

    @utilities.memoized_property
    def primers(self):
        primers = {side: self.features[self.target, self.primer_names[side]] for side in [5, 3]}
        return primers

    @utilities.memoized_property
    def fingerprints(self):
        fps = {
            self.target: [],
            self.donor: [],
        }

        SNP_names = sorted([name for seq_name, name in self.features if seq_name == self.target and name.startswith('SNP')])
        for name in SNP_names:
            fs = {k: self.features[k, name] for k in (self.target, self.donor)}
            ps = {k: (f.strand, f.start) for k, f in fs.items()}
            bs = {k: self.reference_sequences[k][p:p + 1] for k, (strand, p) in ps.items()}
            
            for k in bs:
                if fs[k].strand == '-':
                    bs[k] = utilities.reverse_complement(bs[k])
                
                fps[k].append((ps[k], bs[k]))

        return fps

    @utilities.memoized_property
    def SNPs(self):
        SNPs = {}

        SNP_names = sorted([name for seq_name, name in self.features if seq_name == self.target and name.startswith('SNP')])

        for key, seq_name in [('target', self.target), ('donor', self.donor)]:
            SNPs[key] = {}
            seq = self.reference_sequences[seq_name]
            for name in SNP_names:
                feature = self.features[seq_name, name]
                strand = feature.strand
                position = feature.start
                b = seq[position:position + 1]
                if strand == '-':
                    b = utilities.reverse_complement(b)

                SNPs[key][name] = {
                    'position': position,
                    'strand': strand,
                    'base': b
                }
            
        return SNPs

    @utilities.memoized_property
    def wild_type_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.target]])
    
    @utilities.memoized_property
    def donor_locii(self):
        return ''.join([b for _, b in self.fingerprints[self.donor]])

    def identify_degenerate_indels(self):
        degenerate_dels = defaultdict(list)

        possible_starts = range(self.primers[5].start, self.primers[3].end)
        for starts_at in possible_starts:
            before = self.target_sequence[:starts_at]
            for length in range(1, 200):
                after = self.target_sequence[starts_at + length:]
                result = before + after
                deletion = DegenerateDeletion([starts_at], length)
                degenerate_dels[result].append(deletion)

        with self.fns['degenerate_deletions'].open('w') as fh:
            classes = sorted(degenerate_dels.values(), key=len, reverse=True)
            for degenerate_class in classes:
                if len(degenerate_class) > 1:
                    collapsed = DegenerateDeletion.collapse(degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

        degenerate_inss = defaultdict(list)

        mers = {length: list(utilities.mers(length)) for length in range(1, 5)}

        for starts_after in possible_starts:
            before = self.target_sequence[:starts_after + 1]
            after = self.target_sequence[starts_after + 1:]
            for length in mers:
                for mer in mers[length]:
                    result = before + mer + after
                    insertion = DegenerateInsertion([starts_after], [mer])
                    degenerate_inss[result].append(insertion)

        with self.fns['degenerate_insertions'].open('w') as fh:
            classes = sorted(degenerate_inss.values(), key=len, reverse=True)
            for degenerate_class in classes:
                if len(degenerate_class) > 1:
                    collapsed = DegenerateInsertion.collapse(degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

    @utilities.memoized_property
    def degenerate_indels(self):
        singleton_to_full = {}

        for line in self.fns['degenerate_deletions'].open():
            deletion = degenerate_indel_from_string(line.strip())

            for singleton in deletion.singletons():
                singleton_to_full[singleton] = deletion
        
        for line in self.fns['degenerate_insertions'].open():
            insertion = degenerate_indel_from_string(line.strip())

            for singleton in insertion.singletons():
                singleton_to_full[singleton] = insertion

        return singleton_to_full

    def expand_degenerate_indel(self, indel):
        return self.degenerate_indels.get(indel, indel)

def degenerate_indel_from_string(full_string):
    kind, details_string = full_string.split(':')

    if kind == 'D':
        return DegenerateDeletion.from_string(details_string)
    elif kind == 'I':
        return DegenerateInsertion.from_string(details_string)

class DegenerateDeletion():
    def __init__(self, starts_ats, length):
        self.kind = 'D'
        self.starts_ats = tuple(starts_ats)
        self.length = length

    @classmethod
    def from_string(cls, details_string):
        starts_string, length_string = details_string.split(',')

        starts_ats = [int(s) for s in starts_string.strip('{}').split('|')]
        length = int(length_string)

        return DegenerateDeletion(starts_ats, length)

    @classmethod
    def collapse(cls, degenerate_deletions):
        lengths = {d.length for d in degenerate_deletions}
        if len(lengths) > 1:
            for d in degenerate_deletions:
                print(d)
            raise ValueError
        length = lengths.pop()

        starts_ats = set()
        for d in degenerate_deletions:
            starts_ats.update(d.starts_ats)

        starts_ats = sorted(starts_ats)

        return DegenerateDeletion(starts_ats, length)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_ats))
        if len(self.starts_ats) > 1:
            starts_string = '{' + starts_string + '}'

        full_string = 'D:{0},{1}'.format(starts_string, self.length)

        return full_string

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.starts_ats == other.starts_ats and self.length == other.length

    def __hash__(self):
        return hash((self.starts_ats, self.length))

    def singletons(self):
        return (DegenerateDeletion([starts_at], self.length) for starts_at in self.starts_ats)
    
class DegenerateInsertion():
    def __init__(self, starts_afters, seqs):
        self.kind = 'I'
        self.starts_afters = tuple(starts_afters)
        self.seqs = tuple(seqs)
        
        lengths = set(len(seq) for seq in self.seqs)
        if len(lengths) > 1:
            raise ValueError
        self.length = lengths.pop()
    
        self.pairs = list(zip(self.starts_afters, self.seqs))

    @classmethod
    def from_string(cls, details_string):
        starts_string, seqs_string = details_string.split(',')
        starts_afters = [int(s) for s in starts_string.strip('{}').split('|')]
        seqs = [seq for seq in seqs_string.strip('{}').split('|')]

        return DegenerateInsertion(starts_afters, seqs)
    
    @classmethod
    def from_pairs(cls, pairs):
        starts_afters, seqs = zip(*pairs)
        return DegenerateInsertion(starts_afters, seqs)

    def __str__(self):
        starts_string = '|'.join(map(str, self.starts_afters))
        seqs_string = '|'.join(self.seqs)

        if len(self.starts_afters) > 1:
            starts_string = '{' + starts_string + '}'
            seqs_string = '{' + seqs_string + '}'

        full_string = 'I:{0},{1}'.format(starts_string, seqs_string)

        return full_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.starts_afters == other.starts_afters and self.seqs == other.seqs
    
    def __hash__(self):
        return hash((self.starts_afters, self.seqs))

    def singletons(self):
        return (DegenerateInsertion([starts_after], [seq]) for starts_after, seq in self.pairs)
    
    @classmethod
    def collapse(cls, degenerate_insertions):
        all_pairs = []

        for d in degenerate_insertions:
            all_pairs.extend(d.pairs)

        all_pairs = sorted(all_pairs)

        return DegenerateInsertion.from_pairs(all_pairs)

def parse_benchling_genbank(genbank_fn):
    convert_strand = {
        -1: '-',
        1: '+',
    }

    gb_record = Bio.SeqIO.read(str(genbank_fn), 'genbank')

    fasta_record = fasta.Read(gb_record.name, str(gb_record.seq))

    gff_features = []

    for gb_feature in gb_record.features:
        feature = gff.Feature.from_fields(
            gb_record.id,
            '.',
            gb_feature.type,
            gb_feature.location.start,
            gb_feature.location.end - 1,
            '.',
            convert_strand[gb_feature.location.strand],
            '.',
            '.',
        )
        attribute = {k: v[0] for k, v in gb_feature.qualifiers.items()}

        if 'label' in attribute:
            attribute['ID'] = attribute.pop('label')

        if 'ApEinfo_fwdcolor' in attribute:
            attribute['color'] = attribute.pop('ApEinfo_fwdcolor')

        if 'ApEinfo_revcolor' in attribute:
            attribute.pop('ApEinfo_revcolor')

        feature.attribute = attribute

        gff_features.append(feature)

    return fasta_record, gff_features

def get_all_targets(base_dir):
    targets_dir = Path(base_dir) / 'targets'
    names = (p.name for p in targets_dir.glob('*') if p.is_dir())
    targets = [TargetInfo(base_dir, n) for n in names]
    return targets