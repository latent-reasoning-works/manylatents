import copy
import numpy as np

# just hard-code it

# HGDP+1KGP
cmap_pop = {
     'ACB': '#006D2C',
     'ASW': '#00441B',
     'BantuKenya': 'green',
     'BantuSouthAfrica': 'green',
     'BiakaPygmy': 'green',
     'ESN': '#A1D99B',
     'GWD': '#74C476',
     'LWK': '#41AB5D',
     'MSL': '#238B45',
     'Mandenka': 'green',
     'MbutiPygmy': 'green',
     'San': 'green',
     'YRI': '#C7E9C0',
     'Yoruba': 'green',
     'CLM': '#E3242B',
     'Colombian': 'red',
     'Karitiana': 'red',
     'MXL': '#BC544B',
     'Maya': 'red',
     'PEL': '#E0115F',
     'PUR': '#900D09',
     'Pima': 'red',
     'Surui': 'red',
     'BEB': '#FDBE85',
     'Balochi': 'orange',
     'Brahui': 'orange',
     'Burusho': 'orange',
     'GIH': '#FD8D3C',
     'Hazara': 'orange',
     'ITU': '#E6550D',
     'Kalash': 'orange',
     'Makrani': 'orange',
     'PJL': '#FEEDDE',
     'Pathan': 'orange',
     'STU': '#E6550D',
     'Sindhi': 'orange',
     'CDX': '#008080',
     'CHB': '#DEEBF7',
     'CHS': '#9ECAE1',
     'Cambodian': 'blue',
     'Dai': 'blue',
     'Daur': 'blue',
     'Han': 'blue',
     'Hezhen': 'blue',
     'JPT': '#08519C',
     'Japanese': 'blue',
     'KHV': '#0ABAB5',
     'Lahu': 'blue',
     'Miao': 'blue',
     'Mongola': 'blue',
     'Naxi': 'blue',
     'Oroqen': 'blue',
     'She': 'blue',
     'Tu': 'blue',
     'Tujia': 'blue',
     'Uygur': 'blue',
     'Xibo': 'blue',
     'Yakut': 'blue',
     'Yi': 'blue',
     'Adygei': 'purple',
     'Basque': 'purple',
     'CEU': '#D896FF',
     'FIN': '#800080',
     'French': 'purple',
     'GBR': '#D896FF',
     'IBS': '#EFBBFF',
     'Italian': 'purple',
     'Orcadian': 'purple',
     'Russian': 'purple',
     'Sardinian': 'purple',
     'TSI': '#BE29EC',
     'Tuscan': 'purple',
     'Bedouin': 'grey',
     'Druze': 'grey',
     'Mozabite': 'grey',
     'Palestinian': 'grey',
     'Melanesian': 'yellow',
     'Papuan': 'yellow',
           }

# UKBB
cmap_ukbb_pops = {
    # African ancestry (green family)
    'African': '#228B22',                     # forest green
    'Caribbean': '#66CDAA',                   # medium aquamarine
    'Any other Black background': '#2E8B57',  # sea green
    'Black or Black British': '#006400',      # dark green
    'White and Black African': '#8FBC8F',     # dark sea green

    # European ancestry (purple family)
    'British': '#9370DB',                     # medium purple
    'Irish': '#8A2BE2',                       # blue violet
    'White': '#BA55D3',                       # medium orchid
    'Any other white background': '#DDA0DD',  # plum


    # South/Central Asian ancestry (orange family)
    'Indian': '#FFA500',                      # orange
    'Pakistani': '#FF8C00',                   # dark orange
    'Bangladeshi': '#FFB347',                 # light orange


    # East Asian ancestry (blue family)
    'Chinese': '#1E90FF',                     # dodger blue
    'Asian or Asian British': '#4682B4',      # steel blue

    # Mixed or unknown (gray family)
    'White and Black Caribbean': '#D3D3D3',   
    'White and Asian': '#D3D3D3',             
    'Any other mixed background': '#D3D3D3',  
    'Mixed': '#D3D3D3',                       
    'Other ethnic group': '#D3D3D3',          
    'Prefer not to answer': '#D3D3D3',        
    'Do not know': '#D3D3D3',
    'Any other Asian background': '#D3D3D3',

    # Middle Eastern / ambiguous
    'Other': '#D3D3D3'
}

cmap_ukbb_superpops = {
    "AFR": "green",
    "EUR": "purple",
    "CSA": "orange",
    "EAS": "blue",
    "MID": "gray",
    "AMR": "red",
    "Do not know": "lightgrey"
}

cmap_mhi_superpops = {
    "Black":            "#228B22",   # forest green
    "Caucasian":        "#9370DB",   # medium purple
    "Asian":            "#1E90FF",   # dodger blue
    "Hispanic":         "#FF4500",   # orange-red
    "Native American":  "#8B0000",   # dark red / maroon

    # catch-alls
    "Other":            "#808080",   # medium gray (similar to “MID”)
    "Unlabelled":       "#D3D3D3",   # light gray (like “Do not know”)
}

# AoU
cmap_hgdp_aou_intersection = {
     'ACB': '#006D2C',
     'ASW': '#00441B',
     'BantuKenya': 'green',
     'BantuSouthAfrica': 'green',
     'BiakaPygmy': 'green',
     'ESN': '#A1D99B',
     'GWD': '#74C476',
     'LWK': '#41AB5D',
     'MSL': '#238B45',
     'Mandenka': 'green',
     'MbutiPygmy': 'green',
     'San': 'green',
     'YRI': '#C7E9C0',
     'Yoruba': 'green',
     'CLM': '#E3242B',
     'Colombian': 'red',
     'Karitiana': 'red',
     'MXL': '#BC544B',
     'Maya': 'red',
     'PEL': '#E0115F',
     'PUR': '#900D09',
     'Pima': 'red',
     'Surui': 'red',
     'BEB': '#FDBE85',
     'Balochi': 'orange',
     'Brahui': 'orange',
     'Burusho': 'orange',
     'GIH': '#FD8D3C',
     'Hazara': 'orange',
     'ITU': '#E6550D',
     'Kalash': 'orange',
     'Makrani': 'orange',
     'PJL': '#FEEDDE',
     'Pathan': 'orange',
     'STU': '#E6550D',
     'Sindhi': 'orange',
     'CDX': '#008080',
     'CHB': '#DEEBF7',
     'CHS': '#9ECAE1',
     'Cambodian': 'blue',
     'Dai': 'blue',
     'Daur': 'blue',
     'Han': 'blue',
     'Hezhen': 'blue',
     'JPT': '#08519C',
     'Japanese': 'blue',
     'KHV': '#0ABAB5',
     'Lahu': 'blue',
     'Miao': 'blue',
     'Mongola': 'blue',
     'Naxi': 'blue',
     'Oroqen': 'blue',
     'She': 'blue',
     'Tu': 'blue',
     'Tujia': 'blue',
     'Uygur': 'blue',
     'Xibo': 'blue',
     'Yakut': 'blue',
     'Yi': 'blue',
     'Adygei': 'purple',
     'Basque': 'purple',
     'CEU': '#D896FF',
     'FIN': '#800080',
     'French': 'purple',
     'GBR': '#D896FF',
     'IBS': '#EFBBFF',
     'Italian': 'purple',
     'Orcadian': 'purple',
     'Russian': 'purple',
     'Sardinian': 'purple',
     'TSI': '#BE29EC',
     'Tuscan': 'purple',
     'Bedouin': 'grey',
     'Druze': 'grey',
     'Mozabite': 'grey',
     'Palestinian': 'grey',
     'Melanesian': 'yellow',
     'Papuan': 'yellow',
     'PapuanSepik': 'yellow',
     'BergamoItalian': 'purple',
     'Biaka': 'green',
     'PapuanHighlands': 'yellow',
     'NorthernHan': 'blue',
     'Mongolian': 'blue',
     'Mbuti': 'green',
     'Bougainville': 'yellow'  
           }

# Define race to color mapping
race_ethnicity_only_pca_colors = {
    "Asian": "#B71C1C",
    "Black or African American": "#283593",
    "Native Hawaiian or Other Pacific Islander": "#E040FB",
    "Middle Eastern or North African": "#80461B",
    "Hispanic or Latino": "#41C9F8",
    "White": "#FFA000",
    "More than one population": "#26867d", #"#9E9E9E",
    "No information": "#e8e8e8"
}

# Define new race to color mapping (from paper)
race_only_pca_colors = {
    "Asian": "#B71C1C",
    "White": "#FFA000",
    "Black or African American": "#283593",
    "Middle Eastern or North African": "#80461B",
    "Native Hawaiian or Other Pacific Islander": "#E040FB",
    "No information": "#9E9E9E",
    "Hispanic or Latino": "#9E9E9E", # hack
    "More than one population": "#26867d",
}

# Define ethnicity to color mapping
ethnicity_only_pca_colors = {
    "Hispanic or Latino": "#41C9F8",
    "Not Hispanic or Latino": "#283593",
    "No Information": "#9E9E9E",
}

# Create a new mapping of HGDP-AoU keys to gray
gray_mapping = {k: '#d3d3d3' for k in cmap_hgdp_aou_intersection}

# Update race_ethnicity_only_pca_colors with these gray values
race_ethnicity_only_pca_colors.update(gray_mapping)
race_only_pca_colors.update(gray_mapping)
ethnicity_only_pca_colors.update(gray_mapping)