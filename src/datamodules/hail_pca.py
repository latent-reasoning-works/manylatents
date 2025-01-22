import hail as hl




def compute_pca_from_hail(prefix, 
                          pca_cache_file, 
                          metadata, 
                          to_fit_on, 
                          to_transform_on, 
                          n_components=50):
    n_components = 50  # Adjust based on the number of PCs

    # Initialize Hail
    hl.init(spark_conf={
        'spark.driver.memory': '16g',      # Memory for the driver
        'spark.executor.memory': '16g'     # Memory for each executor
    })

    # Load the Hail Table
    mt = hl.import_plink(bed=prefix + '.bed',
                         bim=prefix + '.bim',
                         fam=prefix + '.fam',
                         reference_genome='GRCh38')

    # Create sets for efficient lookup
    to_fit_on_set = hl.literal(set(metadata.index[to_fit_on]))
    to_transform_on_set = hl.literal(set(metadata.index[to_transform_on]))

    mt = mt.annotate_cols(
        subset = hl.case()
            .when(to_fit_on_set.contains(mt.s), 'fit')
            .when(to_transform_on_set.contains(mt.s), 'transform')
            .default('exclude')
    )

    # Step 1: Compute PCA on the 'fit' subset
    mt_fit = mt.filter_cols(mt.subset == 'fit')

    # Mean impute and standardize genotypes
    eigenvalues, scores, loadings = hl.hwe_normalized_pca(mt_fit.GT, k=n_components, compute_loadings=True)

    # Save the loadings to use for projection
    #loadings_path = 'pca_loadings.ht'
    #loadings.write(loadings_path, overwrite=True)

    allele_frequencies = mt_fit.annotate_rows(af=hl.agg.mean(mt_fit.GT.n_alt_alleles()) / 2).rows()

    # Filter 'mt_transform' to match PCA variants
    mt_transform = mt.filter_cols(mt.subset == 'transform')
    mt_transform = mt_transform.filter_rows(hl.is_defined(loadings[mt_transform.row_key]))

    # Project transform samples
    projected_scores = hl.experimental.pc_project(
        call_expr=mt_transform.GT,
        loadings_expr=loadings[mt_transform.row_key].loadings,
        af_expr=allele_frequencies[mt_transform.row_key].af
    )

    # Combine scores and export
    scores = scores.annotate(subset='fit')
    projected_scores = projected_scores.annotate(subset='transform')

    # Combine the two tables
    all_scores = scores.union(projected_scores)
    
    import pdb
    pdb.set_trace()

    # Expand scores into individual columns
    all_scores = all_scores.annotate(
        **{f"PC{i + 1}": all_scores.scores[i] for i in range(n_components)}
    ).drop("scores")  # Drop the original scores array

    # Export to CSV
    all_scores.export(pca_cache_file)

    # Stop Hail session
    hl.stop()