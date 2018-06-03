from pathlib import Path
from collections import defaultdict
import yaml
import pysam
import Bio.SeqIO

from sequencing import fasta, gff, utilities

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

        self.fns = {
            'ref_fasta': self.dir / 'refs.fasta',
            'ref_gff': self.dir / 'refs.gff',
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
    def cut_after(self):
        sgRNA = self.features[self.target, self.sgRNA]
        seq = self.target_sequence

        if sgRNA.strand == '+':
            PAM = seq[sgRNA.end + 1:sgRNA.end + 4]
            cut_after = sgRNA.end - 3

        elif sgRNA.strand == '-':
            PAM = utilities.reverse_complement(seq[sgRNA.start -3:sgRNA.start])
            cut_after = sgRNA.start + 2

        if PAM[-2:] != 'GG':
            raise ValueError('non-NGG PAM: {0}'.format(PAM))
        
        return cut_after

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
        primers = {
            5: self.features[self.target, 'forward primer'],
            3: self.features[self.target, 'reverse primer'],
        }
        return primers

    @utilities.memoized_property
    def fingerprints(self):
        fps = {
            self.target: [],
            self.donor: [],
        }

        for name in ['SNP{0}'.format(i + 1) for i in range(7)]:
            fs = {k: self.features[k, name] for k in (self.target, self.donor)}
            ps = {k: (f.strand, f.start) for k, f in fs.items()}
            bs = {k: self.reference_sequences[k][p:p + 1] for k, (strand, p) in ps.items()}
            
            for k in bs:
                if fs[k].strand == '-':
                    bs[k] = utilities.reverse_complement(bs[k])
                
                fps[k].append((ps[k], bs[k]))

        return fps

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
                    starts_at = '|'.join(str(starts_at) for starts_at, length in degenerate_class)
                    
                    length = set(length for starts_at, length in degenerate_class)
                    if len(length) > 1:
                        raise ValueError
                    length = length.pop()
                    
                    rep = 'D:{{{0}}},{1}'.format(starts_at, length)
                    class_string = ';'.join('{},{}'.format(starts_at, length) for starts_at, length in degenerate_class)
                    fh.write('{0}\t{1}\n'.format(rep, class_string))

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
                degenerate_class = sorted(degenerate_class)
                if len(degenerate_class) > 1:
                    starts_after = '|'.join(str(starts_after) for starts_after, seq in degenerate_class)
                    seq = '|'.join(sorted(set(seq for starts_after, seq in degenerate_class)))
                    rep = 'I:{{{0}}},{1}'.format(starts_after, seq)
                    class_string = ';'.join('{},{}'.format(starts_after, seq) for starts_after, seq in degenerate_class)
                    fh.write('{0}\t{1}\n'.format(rep, class_string))

    @utilities.memoized_property
    def degenerate_indels(self):
        indel_to_rep = {}

        for line in self.fns['degenerate_deletions'].open():
            rep, dels = line.strip().split('\t')
            dels = [('D', tuple(map(int, d.split(',')))) for d in dels.split(';')]
            for d in dels:
                indel_to_rep[d] = rep
        
        for line in self.fns['degenerate_insertions'].open():
            rep, inss = line.strip().split('\t')
            fields = [ins.split(',') for ins in inss.split(';')]
            inss = [('I', (int(starts_after), seq)) for starts_after, seq in fields]
            for i in inss:
                indel_to_rep[i] = rep

        return indel_to_rep


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


