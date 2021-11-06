# gamma=x.sth
This file type contains a predicted consensus secondary structure in Stockholm format, and this predicted consensus structure is under the prediction accuracy control parameter "x."

# bpp_mats.dat
This file contains average probabilistic consistency based on posterior nucleotide pair-matching probabilities. You can treat this average probabilistic consistency like conventional nucleotide base-pairing probabilities. Nucleotide positions are indexed starting from zero.

# bpp_mats_on_ss.dat
This file contains McCaskill's nucleotide base-pairing probabilities on RNA secondary structures. These nucleotide base-pairing probabilities are used to accelerate ConsProb's inside-outside algorithm.

# bpp_mats_2.dat
This file contains average probabilistic consistency per nucleotide. This average probabilistic consistency is obtained by marginalizing one nucleotide for average probabilistic consistency in "bpp_mats.dat."

# upp_mats_on_x.dat
This file type contains average probabilistic consistency per nucleotide. This average probabilistic consistency is for nucleotide unpairing and under the structural context "x." "hl," "2l," "ml," "el" stand for hairpin loops, 2-loops, multi-loops, external loops, respectively.