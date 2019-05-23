from collections import defaultdict

import pysam

from sequencing import interval, utilities, paired_end, sw, fastq, sam

class TCellLayout():
    def __init__(self, alignments, target_info):
        self.alignments = alignments
        self.target_info = target_info

        self.read = {}
        self.seq = {}
        self.qual = {}

        if 'stitched' in self.alignments:
            al = self.alignments['stitched'][0]
            self.name = al.query_name
            if not al.is_unmapped:
                self.read['stitched'] = sam.mapping_to_Read(al)
                self.seq['stitched'] = self.read['stitched'].seq
                self.qual['stitched'] = self.read['stitched'].qual
            self.original_header = al.header

        if 'R1' in self.alignments:
            for which in ['R1', 'R2']:
                al = self.alignments[which][0]
                self.read[which] = sam.mapping_to_Read(al)
                self.name = self.read[which].name
                self.seq[which] = self.read[which].seq
                self.qual[which] = self.read[which].qual
                self.original_header = al.header

    @utilities.memoized_property
    def read_start_aligned_near_cut(self):
        target = self.target_info.target
        around_cuts = self.target_info.around_cuts(10**5)

        for k in self.alignments:
            for al in self.alignments[k]:
                if al.is_unmapped:
                    continue

                query_interval = sam.query_interval(al)
                if k == 'stitched':
                    edge = (0 in query_interval) or (al.query_alignment_length - 1 in query_interval)
                else:
                    edge = 0 in query_interval

                on_target = al.reference_name == target

                near_cut = len(sam.reference_interval(al) & around_cuts) > 0

                if edge and on_target and near_cut:
                    return True
        
        return False

    @utilities.memoized_property
    def full_length_alignments(self):
        def is_full_length(al):
            covered = interval.get_covered(al)
            return covered.start == 0 and covered.end >= al.query_length - 5
        
        full_length = defaultdict(list)
        for k in self.alignments:
            for al in self.alignments[k]:
                if is_full_length(al):
                    full_length[k].append(al)

        return full_length

    @utilities.memoized_property
    def spans_cut(self):
        if self.reference_interval is None:
            return False
        else:
            reference_name, reference_interval = self.reference_interval
            if reference_name == self.target_info.target:
                return any(c in reference_interval for c in self.target_info.cut_afters)
            else:
                return False
    
    @utilities.memoized_property
    def spans_cut_messy(self):

        min_position = 10**9
        max_position = -1
        for k in self.target_alignments:
            for al in self.target_alignments[k]:
                min_position = min(min_position, al.reference_start)
                max_position = max(max_position, al.reference_end - 1)

        target_interval = interval.Interval(min_position, max_position)
        print(target_interval)

        return any(c in target_interval for c in self.target_info.cut_afters)

    @utilities.memoized_property
    def reference_interval(self):
        full_length = self.full_length_alignments

        if len(full_length['R1']) == 1 and len(full_length['R2']) == 1:
            R1_al, = full_length['R1']
            R2_al, = full_length['R2']
            if R1_al.reference_name == R2_al.reference_name:
                return R1_al.reference_name, paired_end.reference_interval(R1_al, R2_al)
            else:
                return None

        elif len(full_length['stitched']) == 1:
            al, = full_length['stitched']
            return al.reference_name, interval.Interval(al.reference_start, al.reference_end - 1)
        
        else:
            return None

    def categorize(self):
        if self.reference_interval is None:
            return None, None, None
        else:
            reference_name, reference_interval = self.reference_interval
            on_target = (reference_name == self.target_info.target)
            return on_target, self.spans_cut, len(reference_interval)

    @utilities.memoized_property
    def target_alignments(self):
        t_als = {}
        for k in self.alignments:
            t_als[k] = [al for al in self.alignments[k] if al.reference_name == self.target_info.target]
        
        return t_als
    
    @utilities.memoized_property
    def short_edge_alignments(self):
        ''' Look for short alignments at the end of the read to sequence around the cut site. '''
        ti = self.target_info
        
        if len(ti.cut_afters) > 1:
            raise NotImplementedError('more than one cut')

        cut_after = ti.cut_afters[0]
        before_cut = utilities.reverse_complement(ti.target_sequence[:cut_after + 1])[:25]
        after_cut = ti.target_sequence[cut_after + 1:][:25]

        new_targets = [
            ('edge_before_cut', before_cut),
            ('edge_after_cut', after_cut),
        ]

        full_refs = list(self.original_header.references) + [name for name, seq in new_targets]
        full_lengths = list(self.original_header.lengths) + [len(seq) for name, seq in new_targets]

        expanded_header = pysam.AlignmentHeader.from_references(full_refs, full_lengths)

        edge_alignments = sw.align_read(self.read, new_targets, 3, expanded_header, both_directions=False, alignment_type='query_end')

        def comparison_key(al):
            return (al.reference_name, al.reference_start, al.is_reverse, al.cigarstring)

        #already_seen = {comparison_key(al) for al in self.target_alignments}
        already_seen = set()

        new_alignments = []

        for alignment in edge_alignments:
            if alignment.reference_name == 'edge_before_cut':
                alignment.is_reverse = True
                alignment.cigar = alignment.cigar[::-1]
                qual = alignment.query_qualities[::-1]
                alignment.query_sequence = utilities.reverse_complement(alignment.query_sequence)
                alignment.query_qualities = qual
                alignment.reference_start = cut_after + 1 - sam.total_reference_nucs(alignment.cigar)
            else:
                alignment.reference_start = cut_after + 1
                
            alignment.reference_name = ti.target

            if comparison_key(alignment) not in already_seen:
                already_seen.add(comparison_key(alignment))
                new_alignments.append(alignment)
        
        return new_alignments
