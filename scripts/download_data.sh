#!/bin/bash
# downloads bam, bim, bed files from dropbox

# Create the data folder if it doesn't exist.
mkdir -p data

# Define the URLs with dl=1 to force a direct download.
URL_BED="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AGpa4DQta19mLwghQ9RxWdM/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.bed?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"
URL_BIM="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/ADjjXhKolBgF8VwtGmEvw38/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.bim?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"
URL_FAM="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AC0jarjSKKup55pbAvRWOJo/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.fam?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"

# Download the files into the data folder.
wget -O data/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.bed "$URL_BED"
wget -O data/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.bim "$URL_BIM"
wget -O data/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet.fam "$URL_FAM"

# Optional additional downloads:

# URL_METADATA="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AMkWRkSmlElkhvgrgfW1_t4/gnomad_derived_metadata_with_filtered_sampleids.csv?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"
# URL_PCA="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AMTFYeIURGFrYYvQtByQ-yI/pca_scores_hailcomputed.csv?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"
#
# wget -O data/gnomad_derived_metadata_with_filtered_sampleids.csv "$URL_METADATA"
# wget -O data/pca_scores_hailcomputed.csv "$URL_PCA"
