#working directory :
cd [working directory]

ln -s [path to ukbiobank plink files directory] plink


#merge the entire set

for f in $(seq 1 22)
do
echo "plink/ukb_chr${f}_v2" >> allFiles.txt 
done


echo '#!/bin/sh' > merge.sh 
echo "module load [plink2 module]
plink2 --pmerge-list allFiles.txt bfile --make-bed --real-ref-alleles --out ukb_v2" >> merge.sh 
sbatch --chdir $(pwd) --account=ctb-hussinju --time=10:00:00 --output=merge.%j.out --error=merge.%j.err --mem=100G --cpus-per-task=24 merge.sh


mkdir liftover_GRCh38
cd liftover_GRCh38/

file=ukb_v2

awk '{print "chr"$1"\t"$4-1"\t"$4"\t"$2"\t"$5"_"$6}' ../$file.bim > $file.hg19.pos.bed

liftOver $file.hg19.pos.bed [path to chain files]/hg19ToHg38.over.chain $file.GRCh38.pos.bed $file.hg19.unmapped
cut -f 4 $file.GRCh38.pos.bed  > tmp.toExtract.txt

plink --bfile ../$file --extract tmp.toExtract.txt --out $file.hg19.liftOverCommon --make-bed --real-ref-alleles

mkdir GRCh38
cp $file.hg19.liftOverCommon.bed GRCh38/$file.liftOverCommon.GRCh38.bed
cp $file.hg19.liftOverCommon.fam GRCh38/$file.liftOverCommon.GRCh38.fam


sed 's/^chr//' $file.GRCh38.pos.bed | awk '{print $1"\t"$4"\t0\t"$3"\t"$5}' | rev | sed 's/_/\t/' | rev > GRCh38/$file.liftOverCommon.GRCh38.bim
cd GRCh38


plink --bfile $file.liftOverCommon.GRCh38 --make-bed --out $file.liftOverCommon.GRCh38.sorted --allow-extra-chr --real-ref-alleles

#REMOVE INDELS
awk '{if(length($5) == 1 && length($6)==1) print }' $file.liftOverCommon.GRCh38.sorted.bim | sed -e '/random/d' -e '/alt/d' > $file.liftOverCommon.GRCh38.sorted.SNPs.txt


plink --bfile $file.liftOverCommon.GRCh38.sorted --make-bed --out $file.liftOverCommon.GRCh38.sorted.onlySNPs --extract $file.liftOverCommon.GRCh38.sorted.SNPs.txt --real-ref-alleles --allow-extra-chr
