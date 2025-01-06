import numpy as np

def calculate_maf(genotypes):
    """
    Calculate minor allele frequency (MAF) for each SNP (column) in the genotype matrix.

    Args:
        genotypes (numpy.ndarray): Genotype matrix of shape (num_samples, num_SNPs) with values 0, 1, or 2.
        
    Returns:
        numpy.ndarray: MAF for each SNP.
    """
    allele_sum = np.sum(genotypes, axis=0)
    num_samples = genotypes.shape[0]
    
    # Frequency of the minor allele
    maf = allele_sum / (2 * num_samples)
    
    # Ensure MAF is for the minor allele (MAF is the smaller of the two allele frequencies)
    maf = np.minimum(maf, 1 - maf)
    
    return maf

def maf_scale(genotypes):
    """
    Apply MAF scaling to genotype data.

    Args:
        genotypes (numpy.ndarray): Genotype data array with values 0, 1, or 2 (shape: (num_samples, num_SNPs)).
        
    Returns:
        numpy.ndarray: MAF-scaled genotype data.
    """
    maf = calculate_maf(genotypes)  # Calculate MAF for each SNP
    scaled_data = (genotypes - 2 * maf) / np.sqrt(2 * maf * (1 - maf))  # Apply MAF scaling
    
    return scaled_data
