import copy
import seaborn as sns
import numpy as np

# 1000G
super_pops_1000G = {'EAS': ['JPT', 'CHB', 'CHS', 'CDX', 'KHV'], 
                  'EUR': ['CEUGBR', 'TSI', 'FIN', 'IBS'],
                  'AFR': ['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ACB', 'ASW'],
                  'AMR': ['PUR', 'CLM', 'PEL', 'MXL'],
                  'SAS': ['PJL', 'BEB', 'GIH', 'STUITU']}

label_order_1000G_fine =["YRI",
                         "ESN",
                         "GWD",
                         "LWK",
                         "MSL",
                         "ACB",
                         "ASW",
                         "IBS",
                         "CEUGBR",
                         "TSI",
                         "FIN",
                         "PJL",
                         "BEB",
                         "GIH",
                         "STUITU",
                         "CHB",
                         "CHS",
                         "CDX",
                         "KHV",
                         "JPT",
                         "MXL",
                         "CLM",
                         "PEL",
                         "PUR"]

label_order_1000G_coarse = ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']

pop_pallette_1000G_fine = {
    'JPT': '#0000FF',  # Classic Blue
    'CHB': '#0033CC',  # Deep blue
    'CHS': '#0066FF',  # Bright blue
    'CDX': '#0099FF',  # Sky blue
    'KHV': '#00CCCC',  # Cyan

    'CEUGBR': '#FF0000',  # Classic Red
    'TSI': '#CC0000',  # Dark red
    'FIN': '#FF6666',  # Lighter Red
    'IBS': '#FF3333',  # Medium Light Red

    'YRI': '#800080',  # Classic Purple
    'LWK': '#660066',  # Dark purple
    'GWD': '#993399',  # Muted purple
    'MSL': '#CC66CC',  # Lavender purple
    'ESN': '#D896D8',  # Pale purple
    'ACB': '#A040A0',  # Rich purple
    'ASW': '#8A2BE2',  # Vivid violet

    'PUR': '#FFFF00',  # Classic Yellow
    'CLM': '#FFCC00',  # Golden yellow
    'PEL': '#FFD700',  # Bright gold
    'MXL': '#FFA500',  # Orange

    'PJL': '#008000',  # Classic Green
    'BEB': '#004D00',  # Dark green
    'GIH': '#00FF00',  # Neon green
    'STUITU': '#4DFF4D'  # Light green
}



pop_pallette_1000G_coarse = {'EAS': 'blue',
                            'EUR': 'red',
                            'AMR': 'yellow',
                            'AFR': 'purple',
                            'SAS': 'green'
                           }


# SARS-CoV-2
replace_dict = {
    'Delta (B.1.617.2-like)': 'Delta',
    'Delta (AY.4-like)': 'Delta',
    'Delta (B.1.617.2-like) +K417N': 'Delta',
    'Delta (AY.4.2-like)': 'Delta',
    'Beta (B.1.351-like)': 'Beta',
    'Omicron (Unassigned)': 'Omicron'
}

allowed_labels = ['Delta', 'Alpha', 'Omicron', 'Unassigned', 'Gamma', 'Beta', 'Omicron (BA.5-like)',
                  'Omicron (BA.2-like)', 'Omicron (BA.1-like)', 'Omicron (XBB.1.5-like)',
                  'Omicron (XBB.1-like)', 'Omicron (XBB-like)', 'Omicron (BA.4-like)',
                  'Omicron (XBB.1.16-like)', 'Omicron (XE-like)', 'Omicron (BA.3-like)']

WHO_labels = ['Delta', 'Alpha', 'Beta', 'Gamma', 'Omicron', 'Omicron (BA.5-like)', 'Omicron (BA.2-like)',
              'Omicron (BA.1-like)', 'Omicron (XBB.1.5-like)',
              'Omicron (XBB.1-like)', 'Omicron (XBB-like)', 
              'Omicron (BA.4-like)', 'Omicron (XBB.1.16-like)',
              'Omicron (XE-like)', 'Omicron (BA.3-like)', 'Other', 'Unassigned']

# WHO colour dict
pop_pallette_covid_coarse = {
    'Delta': 'red',
    'Alpha': 'blue',
    'Beta': 'green',
    'Omicron': 'orange',
    'Gamma': 'purple',
    'Other': 'pink',
    'Unassigned': 'grey',
}

# Additional labels
additional_labels = [
    'Omicron (BA.5-like)', 'Omicron (BA.2-like)', 'Omicron (BA.1-like)',
    'Omicron (XBB.1.5-like)', 'Omicron (XBB.1-like)', 'Omicron (XBB-like)',
    'Omicron (BA.4-like)', 'Omicron (XBB.1.16-like)', 'Omicron (XE-like)',
    'Omicron (BA.3-like)'
]

# Generate distinct shades of orange for additional labels using the "Purples" palette
additional_colors = sns.color_palette("YlOrBr", n_colors=len(additional_labels))

# Update the color dictionary with new labels and colors
pop_pallette_covid_fine = copy.copy(pop_pallette_covid_coarse)
pop_pallette_covid_fine.update(zip(additional_labels, additional_colors))

label_order_covid_fine = np.sort(WHO_labels)
label_order_covid_coarse = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Omicron', 'Other', 'Unassigned']


# Raph request I use this color palette
pop_pallette_1000G_coarse = {'EAS': 'blue',
                            'EUR': 'purple',
                            'AMR': 'red',
                            'AFR': 'green',
                            'SAS': 'orange'
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
