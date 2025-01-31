#1000G + HGDP dataset pre-filtering steps

cd [working directory]

#1 Keep PASS variants and SNPs only
for f in $(seq 1 22)
do
i=gnomad.genomes.v3.1.2.hgdp_tgp.chr$f
echo '#!/bin/bash' > chr$f.PASS.sh
echo "module load [bcftools 1.9 module]
bcftools view --threads 2 -Oz -f PASS -o $i.PASSfiltered.vcf.gz 1000G_HGDP_merged_dataset/$i.vcf.bgz
zcat $i.PASSfiltered.vcf.gz | head -n 5000 | grep \"#\" > $i.PASSfiltered.header
zcat $i.PASSfiltered.vcf.gz | grep -v \"#\" | cut -f 1-5 | awk '{print \$1\"\t\"\$2\"\t\"\$1\":\"\$2\":\"\$4\":\"\$5\"\t\"\$4\"\t\"\$5}'> $i.PASSfiltered.5cols
zcat $i.PASSfiltered.vcf.gz | grep -v \"#\" | cut -f 6- > $i.PASSfiltered.lastCols
paste $i.PASSfiltered.5cols $i.PASSfiltered.lastCols > $i.PASSfiltered.noHeader
rm $i.PASSfiltered.5cols $i.PASSfiltered.lastCols
awk '{if(length(\$4)==1 && length(\$5)==1)print}' $i.PASSfiltered.noHeader > $i.PASSfiltered.noHeader.onlySNPs
cat $i.PASSfiltered.header $i.PASSfiltered.noHeader.onlySNPs > $i.PASSfiltered.newIDs.onlySNPs.vcf
rm $i.PASSfiltered.header $i.PASSfiltered.noHeader.onlySNPs $i.PASSfiltered.noHeader
bgzip $i.PASSfiltered.newIDs.onlySNPs.vcf
tabix -p vcf $i.PASSfiltered.newIDs.onlySNPs.vcf.gz" >> chr$f.PASS.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=20:00:00 --output=chr$f.PASS.%j.out --error=chr$f.PASS.%j.err --mem=40G --cpus-per-task=2  chr$f.PASS.sh
done






#2 For multi-allelic variants, keep only the one allele with the highest MAF.
#Do after the QC filters : HWE 1e-6 and 5% missing rate

for f in $(seq 1 22)
do 
i=gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs
echo '#!/bin/bash' > chr$f.filterPos.sh
echo "module load [R 4.1.0 module] [plink 1.9 module]
plink --vcf $i.vcf.gz --make-bed --real-ref-alleles --out $i --double-id
plink --bfile $i --freq --out $i.FREQ --double-id

paste <(sed -e 's/^ \\+//' -e 's/ \\+/\\t/g' $i.FREQ.frq) <(awk '{print \$1\"_\"\$4}' $i.bim | sed '1s/^/Position\n/') > $i.FREQ.frq.withPosCol

j=$i.FREQ.frq.withPosCol
echo \"library(dplyr)
library(data.table)

table_freq=read.table(\\\"\$j\\\",h=T,sep=\\\"\t\\\")
table_freq_2=arrange(table_freq,Position,desc(MAF))
table_freq_3=distinct(table_freq_2,Position,.keep_all=1)
write.table(table_freq_3,file=\\\"\$j.noDuplicate\\\",sep=\\\"\t\\\",quote=F,row.names=F)\" > \$j.removeDup.R

R CMD BATCH \$j.removeDup.R

cut -f 2 $i.FREQ.frq.withPosCol.noDuplicate > $i.FREQ.frq.withPosCol.noDuplicate.ids


# Extract resulting selected positions and do the QC steps HWE and Missing 5%
plink --bfile $i --extract $i.FREQ.frq.withPosCol.noDuplicate.ids --geno 0.05 --make-bed --out $i.noDuplicatePos.noMiss5perc --real-ref-alleles --double-id" >> chr$f.filterPos.sh

sbatch --chdir $(pwd) --account=def-hussinju --time=5:00:00 --output=chr$f.filterPos.%j.out --error=chr$f.filterPos.%j.err --mem=40G --cpus-per-task=12  chr$f.filterPos.sh
done


#3 Remove chr in front of the chromosome names in the resulting bim files
for f in $(seq 1 22)
do
cp gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.bim gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.bim.bkp
sed -i 's/chr//' gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.bim
done





#Optional : 
#4 Do the overlap with the GSA arrays 
mkdir overlap_1000G_CAG_MHI
cd overlap_1000G_CAG_MHI

#1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.bim
cp [PATH TO THE 1000G 2504 WGS30X bim files used for another project (DietNet)] .

for f in $(seq 1 22)
do
plink --bfile ../gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc --extract 1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.bim --make-bed --out gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet --real-ref-alleles
done



for f in $(seq 1 22) ; do echo "gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet" >> hgpg_tgp.files.txt; done

plink --merge-list hgpg_tgp.files.txt --real-ref-alleles --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet --make-bed

rm gnomad.genomes.v3.1.2.hgdp_tgp.chr*