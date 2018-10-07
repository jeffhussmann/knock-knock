from pathlib import Path
from collections import defaultdict
import yaml
import pysam
import Bio.SeqIO

from sequencing import fasta, gff, utilities, mapping_tools

class TargetInfo(object):
    def __init__(self, base_dir, name):
        self.name = name
        self.dir = Path(base_dir) / 'targets' / name

        manifest_fn = self.dir / 'manifest.yaml'
        manifest = yaml.load(manifest_fn.open())
        self.target = manifest['target']
        self.donor = manifest['donor']
        self.sources = manifest['sources']

        self.sgRNA = manifest.get('sgRNA', 'sgRNA') 
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
        return fasta.to_dict(self.fns['ref_fasta'])

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
    def PS_range(self):
        sgRNA = self.sgRNA_feature
        return sgRNA.start, sgRNA.end
        
    @utilities.memoized_property
    def sgRNA_feature(self):
        return self.features[self.target, self.sgRNA]
    
    @utilities.memoized_property
    def cut_after(self):
        sgRNA = self.sgRNA_feature
        seq = self.target_sequence

        if sgRNA.strand == '+':
            PAM = seq[sgRNA.end + 1:sgRNA.end + 4]
            cut_after = sgRNA.end - 3

        elif sgRNA.strand == '-':
            PAM = utilities.reverse_complement(seq[sgRNA.start - 3:sgRNA.start])
            cut_after = sgRNA.start + 2

        if PAM[-2:] != 'GG':
            raise ValueError('non-NGG PAM: {0}'.format(PAM))
        
        return cut_after

    @utilities.memoized_property
    def around_cut_features(self):
        upstream = gff.Feature.from_fields(self.target, '.', '.', self.cut_after - 50, self.cut_after, '.', self.sgRNA_feature.strand, '.', '.')
        downstream = gff.Feature.from_fields(self.target, '.', '.', self.cut_after + 1, self.cut_after + 50, '.', self.sgRNA_feature.strand, '.', '.')

        distal, proximal = upstream, downstream
        if self.sgRNA_feature.strand == '+':
            distal, proximal = upstream, downstream
        else:
            distal, proximal = downstream, upstream
            
        distal.attribute['color'] = '#b7e6d7'
        proximal.attribute['color'] = '#85dae9'

        distal.attribute['ID'] = 'PAM-distal of cut'
        proximal.attribute['ID'] = 'PAM-proximal of cut'

        fs = {(self.target, f.attribute['ID']): f for f in [distal, proximal]}

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
                degenerate_dels[result].append((starts_at, length))

        with self.fns['degenerate_deletions'].open('w') as fh:
            classes = sorted(degenerate_dels.values(), key=len, reverse=True)
            for degenerate_class in classes:
                degenerate_class = sorted(degenerate_class)
                if len(degenerate_class) > 1:
                    collapsed  = degenerate_indel_to_string('D', degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

        degenerate_inss = defaultdict(list)

        mers = {length: list(utilities.mers(length)) for length in range(1, 5)}

        for starts_after in possible_starts:
            before = self.target_sequence[:starts_after + 1]
            after = self.target_sequence[starts_after + 1:]
            for length in mers:
                for mer in mers[length]:
                    result = before + mer + after
                    degenerate_inss[result].append((starts_after, mer))

        with self.fns['degenerate_insertions'].open('w') as fh:
            classes = sorted(degenerate_inss.values(), key=len, reverse=True)
            for degenerate_class in classes:
                if len(degenerate_class) > 1:
                    collapsed = degenerate_indel_to_string('I', degenerate_class)
                    fh.write('{0}\n'.format(collapsed))

    @utilities.memoized_property
    def degenerate_indels(self):
        indel_to_rep = {}

        for line in self.fns['degenerate_deletions'].open():
            deletion = degenerate_indel_from_string(line.strip())
            kind, details = deletion

            for starts_at, length in details:
                indel_to_rep['D', (starts_at, length)] = deletion
        
        for line in self.fns['degenerate_insertions'].open():
            insertion = degenerate_indel_from_string(line.strip())
            kind, details = insertion

            for starts_after, base in details:
                indel_to_rep['I', (starts_after, base)] = insertion

        return indel_to_rep

    def expand_degenerate_indel(self, indel):
        expanded = self.degenerate_indels.get(indel)

        if expanded is None:
            kind, details = indel
            expanded = (kind, [details])

        return expanded

def degenerate_indel_to_string(kind, degenerate_class):
    degenerate_class = sorted(degenerate_class)

    if kind == 'D':
        lengths = set(length for starts_at, length in degenerate_class)
        if len(lengths) > 1:
            print(lengths)
            print(degenerate_class)
            raise ValueError

        length = lengths.pop()
        
        all_starts_at = '|'.join(str(s) for s, l in sorted(degenerate_class))
        if len(degenerate_class) > 1:
            all_starts_at = '{' + all_starts_at + '}'

        collapsed = 'D:{0},{1}'.format(all_starts_at, length)

    elif kind == 'I':
        all_starts_after = '|'.join(str(starts_after) for starts_after, seq in degenerate_class)
        all_seqs = '|'.join(seq for starts_after, seq in degenerate_class)

        if len(degenerate_class) > 1:
            all_starts_after = '{' + all_starts_after + '}'
            all_seqs = '{' + all_seqs + '}'

        collapsed = 'I:{0},{1}'.format(all_starts_after, all_seqs)

    return collapsed

def degenerate_indel_from_string(collapsed):
    kind, details = collapsed.split(':')
    starts_field, other_field = details.split(',')

    if '{' in starts_field:
        all_starts = [int(s) for s in starts_field.strip('{}').split('|')]
    else:
        all_starts = [int(starts_field)]

    if kind == 'D':
        all_starts_at = all_starts
        length = int(other_field)

        return kind, [(starts_at, length) for starts_at in all_starts_at]

    elif kind == 'I':
        if '{' in other_field:
            all_bases = [bs for bs in other_field.strip('{}').split('|')]
        else:
            all_bases = [other_field]

        all_starts_after = all_starts

        return kind, list(zip(all_starts_after, all_bases))

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
