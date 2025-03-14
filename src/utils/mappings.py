import copy
import numpy as np

def make_palette_label_order_HGDP(metadata):
    metadata = metadata[1:]
    # get color palette
    pop_pallette_1000G_coarse = {'East_Asia': 'blue',
                                'Europe': 'purple',
                                'America': 'red',
                                'Africa': 'green',
                                'Central_South_Asia': 'orange'
                               }
    label_order_1000G_fine = ['YRI', 'ESN', 'GWD', 'LWK', 'MSL', 'ACB', 'ASW',
                               'IBS',  'CEUGBR', 'TSI', 'FIN',
                               'PJL', 'BEB', 'GIH', 'STUITU',
                               'CHB', 'CHS', 'CDX', 'KHV', 'JPT',
                               'MXL', 'CLM', 'PEL', 'PUR']
    pop_colors=["#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B",
                "#EFBBFF","#D896FF","#BE29EC","#800080",
                "#FEEDDE","#FDBE85","#FD8D3C","#E6550D",
                "#DEEBF7","#9ECAE1","#008080","#0ABAB5","#08519C",
               "#BC544B","#E3242B","#E0115F","#900D09","#7E2811"]
    pop_pallette_1000G_fine = {label:color for label,color in zip(label_order_1000G_fine, pop_colors)}

    pop_palette_hgdp_coarse = copy.deepcopy(pop_pallette_1000G_coarse)
    pop_palette_hgdp_coarse['Middle_East'] = 'grey'
    pop_palette_hgdp_coarse['Oceania'] = 'yellow'

    # create tmp object to hold the original 26 populations
    mapping_26 = copy.deepcopy(pop_pallette_1000G_fine)
    mapping_26['GBR'] = mapping_26['CEUGBR']
    mapping_26['CEU'] = mapping_26['CEUGBR']
    mapping_26['STU'] = mapping_26['STUITU']
    mapping_26['ITU'] = mapping_26['STUITU']

    pop_palette_hgdp_fine = {}
    superpopulations = metadata['Genetic_region_merged']
    populations = metadata['Population']

    for super_pop in np.unique(superpopulations):
        for pop in np.unique(populations[superpopulations==super_pop]):
            if pop not in mapping_26.keys():
                # just use superpop color for now
                pop_palette_hgdp_fine[pop] = pop_palette_hgdp_coarse[super_pop]
            else:
                pop_palette_hgdp_fine[pop] = mapping_26[pop]
    return pop_palette_hgdp_fine, pop_palette_hgdp_coarse
