import os
import numpy as np
import pandas as pd
import copy
from pyplink import PyPlink
import tqdm

def make_palette_label_order_HGDP(metadata):
    # get color palette
    pop_pallette_1000G_coarse = {'East Asia': 'blue',
                                'Europe': 'purple',
                                'America': 'red',
                                'Africa': 'green',
                                'Central South Asia': 'orange'
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
    pop_palette_hgdp_coarse['Middle East'] = 'grey'
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
    return pop_palette_hgdp_coarse, pop_palette_hgdp_fine

def load_data_HGDP(exp_path):
    # Load HGDP data
    try:
        genotypes_array = np.load(exp_path + 'V4_raw_genotypes.npy')
    except:
        fname = 'gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.match1000G_GSAs_dietnet'
        data_path = os.path.join(exp_path, fname)
        pedfile = PyPlink(data_path)
        genotypes_array = np.zeros([pedfile.get_nb_samples(), pedfile.get_nb_markers()], dtype=np.int8)

        for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile)):
            genotypes_array[:,i] = genotypes

        np.save(exp_path + 'V4_raw_genotypes.npy', genotypes_array)

    labels = pd.read_csv(os.path.join(exp_path, 'gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv'), sep='\t')

    # remove duplicate info (appears in filter info too)
    labels = labels.drop(columns=['Project', 'Population', 'Genetic_region'])
    genotypes_array = genotypes_array[1:]  # remove first row
    labels = labels[1:]  # remove first row

    # Load filter data
    filter_info = pd.read_csv(os.path.join(exp_path, '4.3/gnomad_derived_metadata_with_filtered_sampleids.csv'), sep=',', index_col=1)

    merged_metadata = labels.set_index('sample').merge(filter_info, left_index=True, right_index=True)

    # Define mapping for legend names
    legend_labels_map = {
        "East_Asia": "East Asia",
        "Central_South_Asia": "Central South Asia",
        "Middle_East": "Middle East"
    }
    # Update the column in the DataFrame
    merged_metadata['Genetic_region_merged'] = merged_metadata['Genetic_region_merged'].replace(legend_labels_map)

    # load relatedness
    relatedness = pd.read_csv(os.path.join(exp_path,'HGDP+1KGP_MattEstimated_related_samples.tsv'), 
                              sep='\t', 
                              index_col=0)

    
    pop_palette_hgdp_coarse, pop_palette_hgdp_fine = make_palette_label_order_HGDP(merged_metadata)
    
    return merged_metadata, relatedness, genotypes_array, (pop_palette_hgdp_coarse, pop_palette_hgdp_fine)