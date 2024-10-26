#!/bin/bash

python3 compute_pca.py --plink_file_root Data/1KGPHGDPAOU_V7 \
                       --plink_file_name extractedChrAllPruned \
                       --output_folder Data/1KGPHGDPAOU_V7_EMBS \
                       --pca_components 50 \
                       --chunk_size 5000 \
                       --subset '1KGP'