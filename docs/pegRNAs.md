pegRNAs.csv in targets should have columns:
    name - ID of pegRNA
    effector - the effector the pegRNA was designed to work with, typically SaCas9H840A or SpCas9H840A. Code will check that the targetted protospacer identified has an appropriately positioned PAM for the annotated effector.
    protospacer
    scaffold
    extension - RTT, then PBS, then possibly 3′ extension

Concatentated together, protospacer+scaffold+extension should be the full 5′ to 3′ sequence of the pegRNA.
Decomposing them in this way simplifies the process of inferring the programmed edit.
