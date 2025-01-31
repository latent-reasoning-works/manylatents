#TO DO IT ASIDE FROM THE OTHER DATASETS 
#COULD MAKE VARY THE LD PRUNING FILTER STEP

#LD PRUNING STEP : 100 50 0.05

for f in $(seq 1 22)
do
echo '#!/bin/bash' > pruning.chr$f.sh
echo "module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc --maf 0.01 --out gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01 --real-ref-alleles --make-bed
plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01 --indep-pairwise 100 50 0.05 --out gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.prune100_50_0.05
plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01 --extract gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.prune100_50_0.05.prune.in --make-bed --out gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05 --real-ref-alleles" >> pruning.chr$f.sh

sbatch --chdir $(pwd) --account=def-hussinju --time=2:00:00 --output=pruning.chr$f.%j.out --error=pruning.chr$f.%j.err --mem=40G --cpus-per-task=6 pruning.chr$f.sh
done


cp [path to lowcomplexity regions in GRCh38]/{HG38.Anderson2010.bed,HLA_region.bed} .
cat *.bed > lowComplexity.HLA.bed
sed -i 's/chr//' lowComplexity.HLA.bed
awk '{print $0"\t"NR}' lowComplexity.HLA.bed > lowComplexity.HLA.2.bed

for f in $(seq 1 22)
do 
echo "../gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05" >> allFiles.txt
done

plink --merge-list allFiles.txt --real-ref-alleles --make-bed --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05


plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05 --exclude range lowComplexity.HLA.2.bed --real-ref-alleles --make-bed --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05.noHLA.woLowComplexity

plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05.noHLA.woLowComplexity --recode vcf --real-ref-alleles --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned100_50_0.05.noHLA.woLowComplexity.recoded



mkdir forRaph
for f in $(seq 1 22)
do 
echo "../gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05" >> allFiles_maf5.txt
done

plink --merge-list allFiles_maf5.txt --real-ref-alleles --make-bed --out forRaph/gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05


cd forRaph

plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05 --exclude range ../lowComplexity.HLA.2.bed --real-ref-alleles --make-bed --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05.noHLA.woLowComplexity

plink --bfile gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05.noHLA.woLowComplexity --recode vcf --real-ref-alleles --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.05.noHLA.woLowComplexity.recoded
