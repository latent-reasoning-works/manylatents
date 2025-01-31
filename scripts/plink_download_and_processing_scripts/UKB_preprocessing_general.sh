#UKB preprocessing and phasing for Matt's project
#FROM LIFTOVER FILES containing SNPs only


#Make sure to withdraw all individuals that need to be withdrawn
grep -Fwf [path to file containing ids to be withdrawn] ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.fam > indivWithdrawn_20240613.txt

plink --bfile ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs --remove indivWithdrawn_20240613.txt --make-bed --real-ref-alleles --out ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613




prefix=ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613

# Remove variant duplicates.  Keep the ones with the lowest missingness.
module load [R 4.1.0 module]

for i in $prefix.bim
do
i=${i/\.bim/}
plink --bfile $i --missing --out $i.MISSING
paste <(sed '1d' $i.MISSING.lmiss | awk '{print $2"\t"$5}') <(awk '{print $1"_"$4"_"$5"_"$6}' $i.bim) > $i.MISSING.lmiss.withPosCol

sed -i '1s/^/SNP\tMISS\tPosition\n/' $i.MISSING.lmiss.withPosCol


f=$i.MISSING.lmiss.withPosCol
echo "library(dplyr)
library(data.table)

table_freq=read.table(\"$f\",h=T,sep=\"\t\")
table_freq_2=arrange(table_freq,Position,MISS)
table_freq_3=distinct(table_freq_2,Position,.keep_all=1)
write.table(table_freq_3,file=\"$f.noSamePos\",sep=\"\t\",quote=F,row.names=F)" > $f.removeSame.R

R CMD BATCH $f.removeSame.R

plink -bfile $i --extract $i.MISSING.lmiss.withPosCol.noSamePos --make-bed --out $i.woSamePos --real-ref-alleles



# For multi-allelic variants, keep only the one allele with the highest MAF.
plink --bfile $i.woSamePos --freq --out $i.woSamePos.FREQ
paste <(sed -e 's/^ \+//' -e 's/ \+/\t/g' $i.woSamePos.FREQ.frq) <(awk '{print $1"_"$4}' $i.woSamePos.bim | sed '1s/^/Position\n/') > $i.woSamePos.FREQ.frq.withPosCol

f=$i.woSamePos.FREQ.frq.withPosCol
echo "library(dplyr)
library(data.table)

table_freq=read.table(\"$f\",h=T,sep=\"\t\")
table_freq_2=arrange(table_freq,Position,desc(MAF))
table_freq_3=distinct(table_freq_2,Position,.keep_all=1)
write.table(table_freq_3,file=\"$f.noDuplicate\",sep=\"\t\",quote=F,row.names=F)" > $f.removeDup.R

R CMD BATCH $f.removeDup.R

cut -f 2 $i.woSamePos.FREQ.frq.withPosCol.noDuplicate > $i.woSamePos.FREQ.frq.withPosCol.noDuplicate.ids


#3.d. Extract the selected positions and do the QC steps HWE and Missing 5%
plink --bfile $i.woSamePos --extract $i.woSamePos.FREQ.frq.withPosCol.noDuplicate.ids --hwe 0.000001 midp --geno 0.05 --make-bed --out $i.woSamePos.noDuplicatePos.QC --real-ref-alleles

done



# Cleanup
mkdir log_files r_scripts
rm *noDuplicate *.noDuplicate.ids *.withPosCol *.noSamePos *.imiss *.lmiss *.frq
mv *.Rout *.log log_files
mv *.R r_scripts


for f in *.QC.bim ; do cp $f $f.bkp ; done


#Rename the SNP ids
for f in *.QC.bim ; do awk '{print $1"\t"$1":"$4":"$6":"$5"\t"$3"\t"$4"\t"$5"\t"$6}' $f.bkp > $f ; done

#remove samples with too many missing data
for f in *.QC.bim ; do f=${f/\.bim/}  ; plink --bfile $f --mind 0.5 --make-bed --out $f.maxMiss0.5 --real-ref-alleles ; done


#convert in vcf format
for f in *.QC.maxMiss0.5.bim ; do f=${f/\.bim/} ; plink --bfile $f --recode vcf --out $f --real-ref-alleles ; done


mkdir noINDELs_fixref_vcf
# Merged version

echo '#!/bin/bash' > fixref.sh
echo "module load [bcftools 1.9 module] [plink 1.9 module]

prefix=ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613

export BCFTOOLS_PLUGINS=[path to bcftool 1.9 plugins]


bcftools +fixref \$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.vcf --threads 48 -o noINDELs_fixref_vcf/\$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz -Oz -- -f [path to GRCh38 fasta reference] -m flip -d 2> noINDELs_fixref_vcf/\$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.log" >> fixref.sh

sbatch --chdir $(pwd) --account=ctb-hussinju --time=20:00:00 --output=fixref.%j.out --error=fixref.%j.err --mem=200G --cpus-per-task=48 fixref.sh

cd noINDELs_fixref_vcf
module load htslib
tabix -p vcf $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz


# Remove monomorphic sites
module load [bcftools 1.9 module] [plink 1.9 module]

plink --vcf $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz --freq --out $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz.FREQ

awk '{if($5!=0)print $2}' $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz.FREQ.frq > polymorphs.ids


echo '#!/bin/bash' > onlyPolymorph.sh 
echo "module load [bcftools 1.9 module] [plink 1.9 module]

prefix=ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613

bcftools filter --threads 12 -i 'ID=@polymorphs.ids' -Oz -o \$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.vcf.gz \$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.vcf.gz
tabix -p vcf \$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.vcf.gz" >> onlyPolymorph.sh 
sbatch --chdir $(pwd) --account=ctb-hussinju --time=20:00:00 --output=onlyPolymorph.%j.out --error=onlyPolymorph.%j.err --mem=200G --cpus-per-task=12 onlyPolymorph.sh




#REMOVE BAD POSITIONS (TOPMED) :

prefix=ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613

echo '#!/bin/bash' > $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.wrayner.sh

echo "module load [plink 1.9 module]
plink --vcf $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.vcf.gz --make-bed --real-ref-alleles --out $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK

plink --bfile $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK --freq --out $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.FREQ --real-ref-alleles
path_wrayner_script=[path to HRC-1000G-check-bim script]
topmed_path=[path to TOPMED reference file]
perl \$path_wrayner_script -b $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.bim -f $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.FREQ.frq -r \$topmed_path -h" >> $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.wrayner.sh

sbatch --chdir $(pwd) --account=ctb-hussinju --time=10:00:00 --output=$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.wrayner.%j.out --error=$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.wrayner.%j.err --mem=40G --cpus-per-task=1  $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.wrayner.sh




mkdir wrayner_tmp_files

mv Strand-Flip* Position* Chromosome* Run-plink.sh LOG* ID* FreqPlot* Force-Allele1* wrayner_tmp_files/

echo "module load [bcftools 1.9 module]
bcftools filter --threads 12 -e 'ID=@Exclude-$prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK-HRC.txt' -Oz -o $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.vcf.gz $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.vcf.gz" > filterBadSNPs.sh

bash filterBadSNPs.sh

tabix -p vcf $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.vcf.gz


prefix=ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613

#NEED TO RENAME THE SNP ids 
for f in $prefix.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.vcf.gz
do
i=$(echo $f| sed -e 's/.*\///' -e 's/.vcf.gz//')
echo '#!/bin/bash' > $i.ids.sh
echo "module load [bcftools 1.9 module]
zcat $f | head -n 5000 | grep \"#\" > \$SLURM_TMPDIR/$i.header
zcat $f | grep -v \"#\" | cut -f 1-5 | awk '{print \$1\"\t\"\$2\"\t\"\$1\":\"\$2\":\"\$4\":\"\$5\"\t\"\$4\"\t\"\$5}'> \$SLURM_TMPDIR/$i.5cols
zcat $f | grep -v \"#\" | cut -f 6- > \$SLURM_TMPDIR/$i.lastCols
paste \$SLURM_TMPDIR/$i.5cols \$SLURM_TMPDIR/$i.lastCols > \$SLURM_TMPDIR/$i.noHeader
rm \$SLURM_TMPDIR/$i.5cols \$SLURM_TMPDIR/$i.lastCols
cat \$SLURM_TMPDIR/$i.header \$SLURM_TMPDIR/$i.noHeader > $i.newIDs.vcf
bgzip $i.newIDs.vcf
tabix -p vcf $i.newIDs.vcf.gz
rm \$SLURM_TMPDIR/$i.header \$SLURM_TMPDIR/$i.lastCols" >> $i.ids.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=20:00:00 --output=$i.ids.%j.out --error=$i.ids.%j.err --mem=40G --cpus-per-task=6 $i.ids.sh
done




cd ..
mkdir common_1000G_HGDP
cd common_1000G_HGDP

#GET THE INTERSECT WITH 1000G + HGDP GRCh38 phase3 phased dataset
ln -s [path to 1000G_HGDP] 1000G_HGDP_GRCh38



plink --vcf ../ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.vcf.gz --write-snplist --out UKB_ids.txt
#do intersection


for f in $(seq 1 22)
do
echo '#!/bin/bash' > $f.extractCommon.sh

echo "module load [bcftools 1.9 module] [plink 1.9 module]
plink --bfile 1000G_HGDP_GRCh38/gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc --extract UKB_ids.txt.snplist --make-bed --real-ref-alleles --out gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB 
tabix -p vcf ALL.chr$f.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.onlySNPs.newIDs.commonUKB.vcf.gz" >> $f.extractCommon.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=5:00:00 --output=$f.extractCommon.%j.out --error=$f.extractCommon.%j.err --mem=40G --cpus-per-task=6 $f.extractCommon.sh
done




#Filters on 1000G

for f in $(seq 1 22)
do
echo '#!/bin/bash' > $f.Maf0.05_pruning.sh
prefix=gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB
echo "module load [bcftools 1.9 module] [plink 1.9 module]
plink --bfile $prefix --out $prefix.maf0.05 --make-bed --real-ref-alleles --maf 0.05
plink --bfile $prefix.maf0.05 --indep-pairwise 50 5 0.5 --out $prefix.maf0.05.LDpruning
plink --bfile $prefix.maf0.05 --extract $prefix.maf0.05.LDpruning.prune.in --real-ref-alleles --out $prefix.maf0.05.LDpruned --make-bed" >> $f.Maf0.05_pruning.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=1:00:00 --output=$f.Maf0.05_pruning.%j.out --error=$f.Maf0.05_pruning.%j.err --mem=40G --cpus-per-task=6 $f.Maf0.05_pruning.sh
done

#remove Low complexity and HLA
#those files should be provided
cp [path to lowcomplexity regions in GRCh38]/{HG38.Anderson2010.bed,HLA_region.bed} .
cat HLA_region.bed HG38.Anderson2010.bed |sed 's/chr//' > lowComplexity_HLA.bed
awk '{print $0"\tregion"NR}' lowComplexity_HLA.bed > lowComplexity_HLA.txt


for f in $(seq 1 22) ; do echo "gnomad.genomes.v3.1.2.hgdp_tgp.chr$f.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB.maf0.05.LDpruned" >> allFiles.UKB.txt ; done

plink --merge-list allFiles.UKB.txt --make-bed --real-ref-alleles --out gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB.maf0.05.LDpruned

prefix=gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB
plink --bfile $prefix.maf0.05.LDpruned --make-bed --real-ref-alleles --exclude range lowComplexity_HLA.txt --out $prefix.maf0.05.LDpruned.woLowComplexity_noHLA




#Do last filters on UKB
#one last check..

grep -P "^-" ../ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.PLINK.fam > samplesToExcludeUKB.txt

plink --vcf ../ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.vcf.gz --remove samplesToExcludeUKB.txt --extract gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB.maf0.05.LDpruned.woLowComplexity_noHLA.bim --make-bed --real-ref-alleles --out ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.common1000G_HGDP.noBadSamples.woLowComplexity_noHLA


plink --bfile ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.common1000G.noBadSamples --extract gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.commonUKB.maf0.05.LDpruned.woLowComplexity_noHLA.bim --make-bed --real-ref-alleles --out ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.common1000G_HGDP.noBadSamples.woLowComplexity_noHLA