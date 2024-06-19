#!/bin/bash

#define bash variable: 

# what you put into plink
# --bfile when you have .bed, .bim, .fam (do not include suffix)
# --vcf when you have .vcf
infile=$1

# output file
basename=$2

# class labels
classlabels=$3

# paths
codePath='/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/Dietnet' #path for dietnet code 
pythonEnvPath='/lustre06/project/6065672/sciclun4/Envs/dietnetwork'

# load modules and env
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64
source $pythonEnvPath/bin/activate

# run scripts
plink --vcf $infile --recode A --real-ref-alleles --double-id --out $basename 

cut -d' ' -f1,7- $basename.raw | tr ' ' \\t > $basename.raw.cols1and7toend.tab 

python ${codePath}/create_dataset.py --exp-path $(pwd) --genotypes $basename.raw.cols1and7toend.tab --ncpus 20 --out $basename --class-labels $classlabels
