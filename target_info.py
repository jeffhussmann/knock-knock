from pathlib import Path

import yaml
import pysam
import Bio.SeqIO

import Sequencing.fasta as fasta
import Sequencing.gff as gff
import Sequencing.utilities as utilities

base_dir = Path('/home/jah/projects/manu/targets')

class TargetInfo(object):
    def __init__(self, name):
        self.name = name
        self.dir = base_dir / name

        manifest_fn = self.dir / 'manifest.yaml'
        manifest = yaml.load(manifest_fn.open())
        self.target = manifest['target']
        self.donor = manifest['donor']
        self.sources = manifest['sources']

        self.fns = {
            'ref_fasta': self.dir / 'refs.fasta',
            'ref_gff': self.dir / 'refs.gff',
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
        sgRNA = self.features[self.target, 'sgRNA']
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

def get_all_targets():
    names = (p.name for p in base_dir.glob('*') if p.is_dir())
    targets = [TargetInfo(n) for n in names]
    return targets
