#!/bin/bash
# downloads bam, bim, bed files from dropbox

# Create the nested directory if it doesn't exist.
mkdir -p data/HGDP+1KGP/genotypes

# Define the URLs with dl=1 to force a direct download.
URL_BED="https://www.dropbox.com/scl/fo/wtegmicc9cxyd1ggwbpsf/AAGNKU3k4LYDLOnQPyWzj8c/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.bed?rlkey=udwf9dw0d0mt8dk3s3lpyhaoq&dl=1"
URL_BIM="https://www.dropbox.com/scl/fo/wtegmicc9cxyd1ggwbpsf/AAYw1FGAOy-Ipc5Cq4zQN3o/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.bim?rlkey=udwf9dw0d0mt8dk3s3lpyhaoq&dl=1"
URL_FAM="https://www.dropbox.com/scl/fo/wtegmicc9cxyd1ggwbpsf/AM-eWoWihBzq4DOZR41Vx24/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.fam?rlkey=udwf9dw0d0mt8dk3s3lpyhaoq&dl=1"

# Download the files into the new folder.
wget -O data/HGDP+1KGP/genotypes/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.bed "$URL_BED"
wget -O data/HGDP+1KGP/genotypes/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.bim "$URL_BIM"
wget -O data/HGDP+1KGP/genotypes/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.fam "$URL_FAM"

# Optional additional downloads:

# URL_METADATA="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AMkWRkSmlElkhvgrgfW1_t4/gnomad_derived_metadata_with_filtered_sampleids.csv?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"
# URL_PCA="https://www.dropbox.com/scl/fo/3wyjvc06ch4agtvslfrk4/AMTFYeIURGFrYYvQtByQ-yI/pca_scores_hailcomputed.csv?rlkey=bkanr0l2lvbqrd037j7kjeovy&e=1&dl=1"

# wget -O data/HGDP+1KGP/gnomad_derived_metadata_with_filtered_sampleids.csv "$URL_METADATA"
# wget -O data/HGDP+1KGP/pca_scores_hailcomputed.csv "$URL_PCA"