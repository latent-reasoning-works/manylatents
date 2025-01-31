#MHI and Cartagene preprocessing and phasing for Matt's project

cd [path to working directory]

#copy files from Camille's Dietnet temp files where those datasets were processed 

#MHI
cp [MHI_FILE].{bed,bim,fam} .

#Cartagene
 cp [CaG_FILE].{bed,bim,fam} .

#LOAD PLINK MODULES
module load [plink 1.9 module]

#1 Extract the selected positions and do the QC steps HWE and Missing 5%
for f in *.bim ; 
do
f=${f/\.bim/}
plink --bfile $f --hwe 0.000001 midp --geno 0.05 --make-bed --out $f.QC --real-ref-alleles
done

#do intersection between datasets
cut -f 2 [CaG_prefix].QC.bim [MHI_prefix].QC.bim | sort | uniq -c | awk '{if($1==2) print $2}' > intersection_mhi_cag17k.txt
#511916 positions



#2 Cleanup
mkdir log_files
mv *.log log_files

for f in *.QC.bim ; do cp $f $f.bkp ; done


#Rename the SNP ids
for f in *.QC.bim ; do awk '{print $1"\t"$1":"$4":"$6":"$5"\t"$3"\t"$4"\t"$5"\t"$6}' $f.bkp > $f ; done

#remove samples with too many missing data
for f in *.QC.bim ; do f=${f/\.bim/}  ; plink --bfile $f --mind 0.5 --make-bed --out $f.maxMiss0.5 --real-ref-alleles ; done

#0 indiv removed

#convert in vcf format
for f in *.QC.maxMiss0.5.bim ; do f=${f/\.bim/} ; plink --bfile $f --recode vcf --out $f --real-ref-alleles ; done
for f in *.vcf ; do sed -i -e 's/ID=23/ID=X/' -e 's/ID=24/ID=Y/' -e 's/^23/X/' -e 's/^24/Y/' -e 's/\t23:/\tX:/' -e 's/\t24:/\tY:/' $f ; done








mkdir noINDELs_fixref_vcf

for f in *.QC.maxMiss0.5.bim
do
f=${f/\.bim/}
echo '#!/bin/bash' > $f.fixref.sh
echo "module load tabix
module load [bcftools 1.9 module] [plink 1.9 module]

export BCFTOOLS_PLUGINS=[path to bcftools 1.9 plugins]

reference=[path to GRCh38 fasta refence]
bcftools +fixref $f.vcf --threads 48 -o noINDELs_fixref_vcf/$f.fixref.vcf.gz -Oz -- -f \$reference -m flip -d 2> noINDELs_fixref_vcf/$f.fixref.log" >> $f.fixref.sh

sbatch --chdir $(pwd) --account=ctb-hussinju --time=20:00:00 --output=$f.fixref.%j.out --error=$f.fixref.%j.err --mem=200G --cpus-per-task=48 $f.fixref.sh
done


cd noINDELs_fixref_vcf
module load htslib


tabix -p vcf [CaG_prefix].QC.maxMiss0.5.fixref.vcf.gz
tabix -p vcf [MHI_prefix].QC.maxMiss0.5.fixref.vcf.gz



#REMOVE BAD POSITIONS (TOPMED) :

for f in *.QC.maxMiss0.5.fixref.vcf.gz
do
f=${f/\.vcf\.gz/}
echo '#!/bin/bash' > $f.wrayner.sh
echo "module load [plink1.9]
plink --vcf $f.vcf.gz --make-bed --real-ref-alleles --out $f.PLINK
plink --bfile $f.PLINK --freq --out $f.PLINK.FREQ --real-ref-alleles
topmed_path=[path to TOPMED reference file]
path_wrayner_script=[path to HRC-1000G-check-bim script]
perl \$path_wrayner_script -b $f.PLINK.bim -f $f.PLINK.FREQ.frq -r \$topmed_path -h" >> $f.wrayner.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=10:00:00 --output=$f.wrayner.%j.out --error=$f.wrayner.%j.err --mem=40G --cpus-per-task=1  $f.wrayner.sh
done

mkdir wrayner_tmp_files
mv Strand-Flip* Position* Chromosome* Run-plink.sh LOG* ID* FreqPlot* Force-Allele1* wrayner_tmp_files/


for f in *.QC.maxMiss0.5.fixref.vcf.gz
do
f=${f/\.vcf\.gz/}
echo "module load [bcftools 1.9 module]
bcftools filter --threads 12 -e 'ID=@Exclude-$f.PLINK-HRC.txt' -Oz -o $f.noBadPosTopMed.vcf.gz $f.vcf.gz" > filterBadSNPs.sh
bash filterBadSNPs.sh
tabix -p vcf $f.noBadPosTopMed.vcf.gz
done



#NEED TO RENAME THE SNP ids 
for f in *.noBadPosTopMed.vcf.gz
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



mkdir common_1000G_HGDP
cd common_1000G_HGDP

#GET THE INTERSECT WITH 1000G + HGDP GRCh38 phase3 phased dataset
#link to 1000G dataset withj positions renamed and positions filtered for PASS criteria, duplicated and multiallelic positions also accounted for. Only SNPs retained and removed positions with more than 5% missing rate.
#new prefix for the files gnomad.genomes.v3.1.2.hgdp_tgp.chr*.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.vcf.gz
ln -s [path to 1000G_HGDP] 1000G_HGDP_GRCh38

#find overlap with MHI and CaG combined files
for f in ../*.newIDs.vcf.gz
do
f=${f/\.vcf\.gz/}
plink --vcf $f.vcf.gz --write-snplist --out $f.ids.txt
done

#do intersection and get unique set of positions that are present in both datasets
cat ../*.snplist | sort | uniq -d > intersection_mhni_cag17k.txt

for f in $(seq 1 22)
do
echo '#!/bin/bash' > $f.extractCommon.sh

echo "module load [bcftools 1.9 module] [plink 1.9 module]
plink --bfile 1000G_HGDP_GRCh38/[1000G_HGDP per chromosome prefix - f variable] --extract intersection_mhni_cag17k.txt --make-bed --real-ref-alleles --out [1000G_HGDP per chromosome prefix - f variable].commonMHI_CaG17k 
tabix -p vcf [1000G_HGDP per chromosome prefix - f variable].commonMHI_CaG17k.vcf.gz" >> $f.extractCommon.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=5:00:00 --output=$f.extractCommon.%j.out --error=$f.extractCommon.%j.err --mem=40G --cpus-per-task=6 $f.extractCommon.sh
done




#Filters on 1000G

for f in $(seq 1 22)
do
echo '#!/bin/bash' > $f.missing10perc.Maf0.05_pruning.sh
prefix=[1000G_HGDP per chromosome prefix - f variable].commonMHI_CaG17k

echo "module load [bcftools 1.9 module] [plink 1.9 module]
plink --bfile $prefix --out $prefix.miss10perc.maf0.05 --make-bed --real-ref-alleles --geno 0.1 --maf 0.05
plink --bfile $prefix.miss10perc.maf0.05 --indep-pairwise 50 5 0.5 --out $prefix.miss10perc.maf0.05.LDpruning
plink --bfile $prefix.miss10perc.maf0.05 --extract $prefix.miss10perc.maf0.05.LDpruning.prune.in --real-ref-alleles --out $prefix.miss10perc.maf0.05.LDpruned --make-bed" >> $f.missing10perc.Maf0.05_pruning.sh
sbatch --chdir $(pwd) --account=ctb-hussinju --time=1:00:00 --output=$f.missing10perc.Maf0.05_pruning.%j.out --error=$f.missing10perc.Maf0.05_pruning.%j.err --mem=40G --cpus-per-task=6 $f.missing10perc.Maf0.05_pruning.sh
done

#remove Low complexity and HLA
#those files should be provided
cp [path to lowcomplexity regions in GRCh38]/{HG38.Anderson2010.bed,HLA_region.bed} .
cat HLA_region.bed HG38.Anderson2010.bed |sed 's/chr//' > lowComplexity_HLA.bed
awk '{print $0"\tregion"NR}' lowComplexity_HLA.bed > lowComplexity_HLA.txt


for f in $(seq 1 22) ; do echo "[1000G_HGDP per chromosome prefix - f variable].commonMHI_CaG17k.miss10perc.maf0.05.LDpruned" >> allFiles.1000G_HGDP.txt ; done

plink --merge-list allFiles.1000G_HGDP.txt --make-bed --real-ref-alleles --out [1000G_HGDP prefix].commonMHI_CaG17k.miss10perc.maf0.05.LDpruned

prefix=[1000G_HGDP prefix].commonMHI_CaG17k
plink --bfile $prefix.miss10perc.maf0.05.LDpruned --make-bed --real-ref-alleles --exclude range lowComplexity_HLA.txt --out $prefix.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA



#Do last filters on MHI and CaG
for f in *.QC.maxMiss0.5.fixref.noBadPosTopMed.newIDs.vcf.gz 
do
f=$(echo $f | sed -e 's/\.\.\///' -e 's/\.vcf\.gz//')
plink --vcf ../$f.vcf.gz --extract [1000G_HGDP prefix].commonMHI_CaG17k.miss10perc.maf0.05.LDpruned.woLowComplexity_noHLA.bim --make-bed --real-ref-alleles --out $f.common1000G_HGDP.woLowComplexity_noHLA
done

