#!/bin/bash
# Downloads HGDP-1KGP zip file from Dropbox

# Dropbox direct download link
root="data/HGDP+1KGP"
plink_root="${root}/genotypes/no_intersection_less_stringent"
plink_fname="gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned_500_50_0.05.woLowComplexity_noHLA"

python scripts/clean_up_admixture.py --plink_prefix ${plink_root}/${plink_fname} --metadata_path ${root}/"gnomad_derived_metadata_with_filtered_sampleids.csv" --admix_file_root ${root}/"admixture"
