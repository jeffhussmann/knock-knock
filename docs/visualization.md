The centerpiece of knock-knock's output are diagrams representing the architecture of local alignments that make up each amplicon sequencing read, like this:
![](legend_v2_unannotated.png)

This particular diagram shows an outcome in which the genomic locus has been modified by insertion of payload that consists of two concatenated copies of the intended PCR-product HDR donor, with a clean handoff from genomic sequence to donor sequence at homology arms on each side of the cut. To understand how this diagram represents this information, here is the same diagram again with some overlaid explanations:

![](legend_v2_annotated.png)

1. The black arrow in the center of a diagram represents a sequencing read of a single editing outcome. The thick orange line on top represents the sequence of an HDR donor,  and the thick blue line on bottom represents the targeted genome in the vicinity of the cut site.
1. Arrows represent local alignments between part of the sequencing read and part of a reference sequence. The horizontal extent of each arrow marks the aligned portion of the sequencing read, and a parallelogram connects to the corresponding aligned region of a reference sequence.
1. Annotated features on reference sequences are marked with colored rectangles and labeled by name. Rectangles of the same colors connect sections of alignments that overlap features to corresponding sections of the sequencing read.
1. Mismatches are marked with crosses. Crosses at low-quality read positions are faded.
1. Deletions are marked by upward indentations and insertions by downward indentations (1 nt long if unlabeled, otherwise labeled with length).
1. Vertical lines connect edges of alignments to their position on the sequencing read.
1.  Numbers next to alignment ends indicate position in relevant reference sequence.
1. The cut site is marked by a vertical dashed line.