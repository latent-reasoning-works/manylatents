import copy

import numpy as np

# ------------------------------
# HGDP Mapping Function
# ------------------------------

def make_palette_label_order_HGDP(populations, superpopulations):
    """
    Constructs HGDP palette dictionaries and label orders from arrays of populations and superpopulations.
    
    The function adjusts the coarse palette by replacing 'SAS' with 'CSA' and adding new keys ('MID', 'OCE').
    For the fine palette, it uses the 1000G fine palette (with some alias adjustments) and falls back to the
    coarse color for populations missing from the fine mapping.
    
    Args:
        populations (np.ndarray): Array of fine population labels.
        superpopulations (np.ndarray): Array of corresponding superpopulation labels.
    
    Returns:
        tuple: (pop_palette_hgdp_coarse, pop_palette_hgdp_fine, label_order_hgdp_coarse, label_order_hgdp_fine)
    """
    # Adjust the coarse palette: rename SAS to CSA and add MID and OCE.
    pop_palette_hgdp_coarse = copy.deepcopy(POP_PALETTE_1000G_COARSE)
    pop_palette_hgdp_coarse['CSA'] = POP_PALETTE_1000G_COARSE.get('SAS', 'grey')
    pop_palette_hgdp_coarse.pop('SAS', None)
    pop_palette_hgdp_coarse['MID'] = 'grey'
    pop_palette_hgdp_coarse['OCE'] = 'yellow'

    label_order_hgdp_coarse = copy.deepcopy(LABEL_ORDER_1000G_COARSE)
    if 'SAS' in label_order_hgdp_coarse:
        label_order_hgdp_coarse.remove('SAS')
    label_order_hgdp_coarse += ['CSA', 'MID', 'OCE']

    # Build the fine label order: for each unique superpopulation, add sorted unique populations.
    label_order_hgdp_fine = []
    for super_pop in np.unique(superpopulations):
        pops = sorted(np.unique(populations[superpopulations == super_pop]).tolist())
        label_order_hgdp_fine.extend(pops)

    # Start with the fine 1000G palette as a base mapping.
    mapping_26 = copy.deepcopy(POP_PALETTE_1000G_FINE)
    mapping_26['GBR'] = mapping_26.get('CEUGBR')
    mapping_26['CEU'] = mapping_26.get('CEUGBR')
    mapping_26['STU'] = mapping_26.get('STUITU')
    mapping_26['ITU'] = mapping_26.get('STUITU')

    pop_palette_hgdp_fine = {}
    for super_pop in np.unique(superpopulations):
        pops = sorted(np.unique(populations[superpopulations == super_pop]).tolist())
        for pop in pops:
            if pop not in mapping_26:
                # Fallback: use the coarse color for the superpopulation.
                pop_palette_hgdp_fine[pop] = pop_palette_hgdp_coarse.get(super_pop, 'gray')
            else:
                pop_palette_hgdp_fine[pop] = mapping_26[pop]

    return pop_palette_hgdp_coarse, pop_palette_hgdp_fine, label_order_hgdp_coarse, label_order_hgdp_fine
