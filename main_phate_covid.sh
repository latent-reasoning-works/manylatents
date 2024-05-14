#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=12:00:00 
#SBATCH --mem=124GB
#SBATCH --job-name=run_phates
#SBATCH --output=slurm_files/run_phates_%j.out
#SBATCH --error=slurm_files/run_phates_%j.err

source /lustre06/project/6065672/sciclun4/Envs/phate_env/bin/activate

# Define paths
data_path_1000G='/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/oldnow/MattsPlace/1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1.hdf5'
data_path_covid_metadata='/lustre06/project/6065672/shared/covid-19/database/data/intra/all_filtered_iSNVs/metadata/metadata.tsv'
data_path_covid='/lustre06/project/6065672/shared/covid-19/database/data/intra/all_filtered_iSNVs/experiment_V2/pca_iSNVs_consensus_embeddings.tsv'


# Define ranges for hyperparameters
decays=(10 20 40 60)
knns=(5 10 20 50 100 200)

gammas=(1 0)
ts=('auto' 5 25 100)

# Iterate over hyperparameters
for decay in "${decays[@]}"; do
    for knn in "${knns[@]}"; do
        for gamma in "${gammas[@]}"; do
            for t in "${ts[@]}"; do
                echo "Running PHATE with t=$t, decay=$decay, knn=$knn, gamma=$gamma"
                python main.py --data_path $data_path_covid \
                               --metadata_path $data_path_covid_metadata \
                               --dataset_type 'covid' \
                               --manifold_algo 'phate' \
                               --plot \
                               --label_positions \
                               --t $t \
                               --decay $decay \
                               --knn $knn \
                               --gamma $gamma
            done
        done
    done
done