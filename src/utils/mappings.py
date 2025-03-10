import copy

import numpy as np
import seaborn as sns

# ------------------------------
# 1000 Genomes (1000G) Mappings
# ------------------------------

SUPER_POPS_1000G = {
    'EAS': ['JPT', 'CHB', 'CHS', 'CDX', 'KHV'],
    'EUR': ['CEUGBR', 'TSI', 'FIN', 'IBS'],
    'AFR': ['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ACB', 'ASW'],
    'AMR': ['PUR', 'CLM', 'PEL', 'MXL'],
    'SAS': ['PJL', 'BEB', 'GIH', 'STUITU']
}

LABEL_ORDER_1000G_FINE = [
    'YRI', 'ESN', 'GWD', 'LWK', 'MSL', 'ACB', 'ASW',
    'IBS', 'CEUGBR', 'TSI', 'FIN', 'PJL', 'BEB', 'GIH', 'STUITU',
    'CHB', 'CHS', 'CDX', 'KHV', 'JPT', 'MXL', 'CLM', 'PEL', 'PUR'
]

LABEL_ORDER_1000G_COARSE = ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']

# Raph's requested coarse palette for 1000G
POP_PALETTE_1000G_COARSE = {
    'EAS': 'blue',
    'EUR': 'purple',
    'AMR': 'red',
    'AFR': 'green',
    'SAS': 'orange'
}

# Generate fine palette for 1000G.
# Here we use a custom list of colors so that each population gets a distinct shade.
POP_COLORS_1000G = [
    "#C7E9C0", "#A1D99B", "#74C476", "#41AB5D", "#238B45", "#006D2C", "#00441B",
    "#EFBBFF", "#D896FF", "#BE29EC", "#800080",
    "#FEEDDE", "#FDBE85", "#FD8D3C", "#E6550D",
    "#DEEBF7", "#9ECAE1", "#008080", "#0ABAB5", "#08519C",
    "#BC544B", "#E3242B", "#E0115F", "#900D09", "#7E2811"
]
POP_PALETTE_1000G_FINE = {label: color for label, color in zip(LABEL_ORDER_1000G_FINE, POP_COLORS_1000G)}

# ------------------------------
# SARS-CoV-2 Mappings
# ------------------------------

REPLACE_DICT_COVID = {
    'Delta (B.1.617.2-like)': 'Delta',
    'Delta (AY.4-like)': 'Delta',
    'Delta (B.1.617.2-like) +K417N': 'Delta',
    'Delta (AY.4.2-like)': 'Delta',
    'Beta (B.1.351-like)': 'Beta',
    'Omicron (Unassigned)': 'Omicron'
}

ALLOWED_LABELS_COVID = [
    'Delta', 'Alpha', 'Omicron', 'Unassigned', 'Gamma', 'Beta',
    'Omicron (BA.5-like)', 'Omicron (BA.2-like)', 'Omicron (BA.1-like)',
    'Omicron (XBB.1.5-like)', 'Omicron (XBB.1-like)', 'Omicron (XBB-like)',
    'Omicron (BA.4-like)', 'Omicron (XBB.1.16-like)', 'Omicron (XE-like)',
    'Omicron (BA.3-like)'
]

WHO_LABELS = [
    'Delta', 'Alpha', 'Beta', 'Gamma', 'Omicron', 'Omicron (BA.5-like)',
    'Omicron (BA.2-like)', 'Omicron (BA.1-like)', 'Omicron (XBB.1.5-like)',
    'Omicron (XBB.1-like)', 'Omicron (XBB-like)',
    'Omicron (BA.4-like)', 'Omicron (XBB.1.16-like)',
    'Omicron (XE-like)', 'Omicron (BA.3-like)', 'Other', 'Unassigned'
]

POP_PALETTE_COVID_COARSE = {
    'Delta': 'red',
    'Alpha': 'blue',
    'Beta': 'green',
    'Omicron': 'orange',
    'Gamma': 'purple',
    'Other': 'pink',
    'Unassigned': 'grey'
}

# Additional COVID labels and colors (using a seaborn palette)
ADDITIONAL_LABELS_COVID = [
    'Omicron (BA.5-like)', 'Omicron (BA.2-like)', 'Omicron (BA.1-like)',
    'Omicron (XBB.1.5-like)', 'Omicron (XBB.1-like)', 'Omicron (XBB-like)',
    'Omicron (BA.4-like)', 'Omicron (XBB.1.16-like)', 'Omicron (XE-like)',
    'Omicron (BA.3-like)'
]
ADDITIONAL_COLORS_COVID = sns.color_palette("YlOrBr", n_colors=len(ADDITIONAL_LABELS_COVID))

POP_PALETTE_COVID_FINE = copy.copy(POP_PALETTE_COVID_COARSE)
POP_PALETTE_COVID_FINE.update(dict(zip(ADDITIONAL_LABELS_COVID, ADDITIONAL_COLORS_COVID)))

LABEL_ORDER_COVID_FINE = sorted(WHO_LABELS)
LABEL_ORDER_COVID_COARSE = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Omicron', 'Other', 'Unassigned']

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
