## Instructions for creating a manual target in benchling

Constucting a manual target consists of annotating the relevant genomic sequence with the locations of amplicon primers and protospacer(s), exporting the resulting annotated sequence in genbank format to the appropriate location, and telling knock-knock to convert this genbank file into the format it uses internally.

First, in benchling, create a 'DNA sequence' and populate it with the sequence of the entire amplicon region plus >=200 nts of flanking genomic sequence on either side.

Next, annotate the primer binding locations with the names 'foward_primer' and 'reverse_primer' and with the 'Strand' field of the primer whose 5'-to-3' sequence is the bottom strand of benchling's display set to 'reverse'.
Annotate the protospacer sequence with the name 'sgRNA'. Do not include the PAM sequence in this annotation (knock-knock will check for the existence of a PAM next to the protospacer). Set the strand to 'reverse' if the 5'-to-3' sequence of the protospacer is on the bottom strand.
Set the 'Annotation type' field to 'sgRNA_SpCas9' (this tells knock-knock what the PAM should match and where the cut is expected).

Next, export this annotated sequence into a knock-knock project directory.
For a target named TARGET_NAME in project directory PROJECT_DIR, create the subdirectory `{PROJECT_DIR}/targets/{TARGET_NAME}`.
Export from benchling by clicking the 'Information' button on the right side of the benchling interface, selecting 'Genbank (.gb)' from the 'Export Data' menu, clicking 'Download', and saving the file into `{PROJECT_DIR}/targets/{TARGET_NAME}`.

Finally, tell knock-knock to extract the information it needs from this file by running
```knock-knock build-manual-target {PROJECT_DIR} {TARGET_NAME}```.