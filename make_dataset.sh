#!/bin/bash

path_to_raw_data='/lustre06/project/6065672/shared/1000G_HGDP_merged_dataset/onlyPASS_onlySNPs/noDuplicates/forMatt/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05.noHLA.woLowComplexity.recoded.vcf'

proc_data_name='gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05.noHLA.woLowComplexity.recoded'
proc_data_path='/lustre06/project/6065672/shared/1000G_HGDP_merged_dataset/PASSfiltered'

class_labels='/lustre06/project/6065672/grenier2/DietNet/Generalisation/datasets_112023/HGDP_1KGP/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'

bash make_h5.sh $path_to_raw_data ${proc_data_path}/${proc_data_name} $class_labels
