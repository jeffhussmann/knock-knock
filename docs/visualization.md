# Alignment diagrams

knock-knock's output is centered around diagrams representing the architecture of local alignments that make up each amplicon sequencing read.
Here is an example diagram showing a Pacbio sequencing read of an editing outcome in which the genomic locus has been modified by knock-in of two concatenated copies of the intended PCR-product HDR donor, with a clean handoff from genomic sequence to donor sequence at homology arms on each side of the cut:
![](legend_v2_unannotated.png)

To understand how this diagram represents this information, here is the same diagram again with some overlaid explanations:

![](legend_v2_annotated.png)

1. The black arrow in the center of a diagram represents a sequencing read of a single editing outcome.
The thick orange line on top represents the sequence of an HDR donor, and the thick blue line on bottom represents the targeted genome in the vicinity of the cut site.
1. Arrows represent local alignments between part of the sequencing read and part of a reference sequence.
The horizontal extent of each arrow marks the aligned portion of the sequencing read, and a parallelogram connects to the corresponding aligned region of a reference sequence.
1. Annotated features on reference sequences are marked with colored rectangles and labeled by name.
Rectangles of the same colors connect sections of alignments that overlap features to corresponding sections of the sequencing read.
1. Mismatches are marked with crosses.
Crosses at low-quality read positions are faded.
1. Deletions are marked by upward indentations and insertions by downward indentations (1 nt long if unlabeled, otherwise labeled with length).
1. Vertical lines connect edges of alignments to their position on the sequencing read.
1. Numbers next to alignment ends indicate position in relevant reference sequence.
1. The cut site is marked by a vertical dashed line.
    
To demonstrate additional features not present in the above example, here is a different diagram, this time of an Illumina sequencing read of an editing outcome in which a portion of an ssDNA HDR donor has been inserted at the cut site in the opposite orientation of the intended outcome:

![](legend_v2_reverse.png)

9. If an alignment is to the reverse strand of a reference sequence, its arrowhead points left instead of right, and the lines connecting to the reference cross each other. 
1. When two alignments overlap (that is, both cover the same part of the sequencing read) at the junction between them, this indicates that there is a stretch of microhomology between the relevant parts of the two reference sequences. 


In another Pacbio read, a stretch of human genomic DNA not originating from the immediate vicinity of the cut site has been integrated.

![](legend_v2_supplementary_ref.png)

11. Alignments to reference sequences that aren't the donor sequence or the immediate vicinity of the cut site are represented by grey arrows, with the name of the reference sequence annotated on the far right (in this case, chromosome 15 of the hg38 genome assembly).

# Interactive exploration of outcomes

## Tables

knock-knock generates interactive tables of outcomes frequencies showing how often each different outcome category (column) was observed in each experimental sample (row).
Hovering over the percentage in a cell pops up an example diagram of that outcome category from that sample.

![](table_diagrams_demo.gif)

Hovering over numbers in the 'Total relevant reads' column pops up a plot of the distribution of amplicon read lengths in that sample.

![](table_lengths_demo.gif)

Clicking on the percentage in a cell brings up a window with more information about occurences of that outcome in that sample.
At the top of the window, a plot shows how the distribution of amplicon lengths for that outcome compare to all other reads in the sample.
Scroll down to see diagrams of more example reads.
Click outside the window or press Esc to return to the main table.

![](table_modal_demo.gif)

Click the name of a sample to open an [outcome browser](#Browser) for that sample in a new tab.

## Length distribution browser

To further explore the diversity of sequence archictures present within each category, knock-knock produces interactive plots showing the distribution of amplicon lengths for each category, with the ability to pop up an example diagram for each observed length.
These plots initially have a single panel that is maximally zoomed out on the y-axis.
To see what is happening at lower frequency ranges, click on the bracket in the lower right corner to pop up a panel that zooms in on the bracketed range of the y-axis. 
(Repeat in the newly-created panel to zoom even further in.)
Clicking an already-open bracket closes it (and all panels below it).

![](browser_zoom_demo.gif)

To explore the outcomes that make up a particular outcome cateogry, click the colored line next to the outcome category in the legend of the appropriately-zoomed panel.
This will highlight that category in the plot, and dim all other categories.
Once highlighted, hovering over lengths at which that category is observed pops up a diagram with an example of that outcome at that length.

![](browser_popovers_demo.gif)

Although knock-knock was designed to analyze knock-in experiments, it can also be useful for exploring editing outcomes in other settings.
As an example, here is data from an experiment in which a gene was cut with three guides within a few hundred nts of each other (vertical dashed lines in the single example outcome below) to drive efficient disruption of gene function by producing excision of the sequence between the cuts.

![](no_donor_example.png)

Exploring the distribution of amplicon lengths produced by this strategy reveals three major peaks consisting of exicsion between the first and second, second and third, or first and third cut sites.
Tails of slightly shorter outcomes beneath each peak consist of outcomes in which an excisions between two cuts is accompanied by a deletion at the remaining cut.

![](browser_no_donor_demo.gif)