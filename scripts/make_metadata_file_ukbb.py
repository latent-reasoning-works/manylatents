
import os
import sys
import numpy as np
import pandas as pd
import argparse
from pyproj import Transformer
from pyplink import PyPlink
from tqdm import tqdm
from collections import Counter


def fast_mode(row):
    values = [v for v in row if pd.notnull(v)]
    if values:
        return Counter(values).most_common(1)[0][0]
    return np.nan


def load_csv(data_root, filename, index_col=None, debug=False, **read_csv_kwargs):
    """
    Load a CSV file with optional debugging mode (top 10 rows).
    
    Args:
        data_root (str): Path to the root data directory.
        filename (str): Name of the CSV file.
        index_col (str, optional): Column to set as index. Defaults to None.
        sep (str, optional): Separator used in CSV file. Defaults to ','.
        debug (bool, optional): Whether to use only the top 10 rows. Defaults to False.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = os.path.join(data_root, filename)
    df = pd.read_csv(file_path, **read_csv_kwargs)

    if debug:
        df = df.head(10)  # Limit to first 10 rows
        print(f"Loaded {filename} in debug mode: {df.shape}")

    if index_col:
        df = df.set_index(index_col)

    return df


def bng_to_wgs84(eastings, northings):
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(eastings, northings)
    return lat, lon


def map_ethnicity(df):
    ethnicity_mapping = {
        1: "White", 1001: "British", 1002: "Irish", 1003: "Any other white background",
        2: "Mixed", 2001: "White and Black Caribbean", 2002: "White and Black African",
        2003: "White and Asian", 2004: "Any other mixed background",
        3: "Asian or Asian British", 3001: "Indian", 3002: "Pakistani",
        3003: "Bangladeshi", 3004: "Any other Asian background",
        4: "Black or Black British", 4001: "Caribbean", 4002: "African",
        4003: "Any other Black background", 5: "Chinese", 6: "Other ethnic group",
        -1: "Do not know", -3: "Prefer not to answer"
    }
    df['value'] = df['value'].map(ethnicity_mapping)
    return df

def map_urban_rural(df):
    urban_rural_mapping = {
        1: "England/Wales - Urban - sparse",
        2: "England/Wales - Town and Fringe - sparse",
        3: "England/Wales - Village - sparse",
        4: "England/Wales - Hamlet and Isolated dwelling - sparse",
        5: "England/Wales - Urban - less sparse",
        6: "England/Wales - Town and Fringe - less sparse",
        7: "England/Wales - Village - less sparse",
        8: "England/Wales - Hamlet and Isolated Dwelling - less sparse",
        9: "Postcode not linkable",
        11: "Scotland - Large Urban Area",
        12: "Scotland - Other Urban Area",
        13: "Scotland - Accessible Small Town",
        14: "Scotland - Remote Small Town",
        15: "Scotland - Very Remote Small Town",
        16: "Scotland - Accessible Rural",
        17: "Scotland - Remote Rural",
        18: "Scotland - Very Remote Rural"
    }
    df['value'] = df['value'].map(urban_rural_mapping)
    return df

def main(data_root, compute_lat_lon, plink_filename):
    """Make UKBB metadata, with optional debug mode (top 10 rows)."""

    # Load metadata and add both indices (metadata and from fam file)
    metadata = load_csv(data_root, 'all_pops_non_eur_pruned_within_pop_pc_covs.tsv', sep='\t')
    index_map = load_csv(data_root, 'ukb20168bridge31063.txt', sep=' ', header=None)

    index_map = index_map.rename(columns={0: 'IDs', 1: 'MetadataIDs'})

    # Merge datasets
    metadata = index_map.merge(metadata, left_on='MetadataIDs', right_on='s', how='left')
    metadata = metadata.set_index('IDs')

    # Load additional metadata from csv files.
    north_coord = load_csv(data_root, '129.csv', index_col='sample_id')
    east_coord = load_csv(data_root, '130.csv', index_col='sample_id')
    self_described_ancestry = load_csv(data_root, '21000.csv', 'sample_id')
    urban_rural = load_csv(data_root, '20118.csv', index_col='sample_id')

    # remap when needed
    self_described_ancestry = map_ethnicity(self_described_ancestry)
    urban_rural = map_urban_rural(urban_rural)

    # Combine additional datasets together
    assert len(north_coord['index'].unique()) == 1
    assert len(east_coord['index'].unique()) == 1
    north_coord = north_coord.drop(columns=['index'])
    east_coord = east_coord.drop(columns=['index'])

    assert len(north_coord['variable_id'].unique()) == 1
    assert len(east_coord['variable_id'].unique()) == 1
    north_coord = north_coord.drop(columns=['variable_id'])
    east_coord = east_coord.drop(columns=['variable_id'])

    # rename vars
    north_coord = north_coord.rename(columns={'instance': 'north_coord_instance',
                               'value': 'north_coord'})

    east_coord = east_coord.rename(columns={'instance': 'east_coord_instance',
                               'value': 'east_coord'})

    coords = north_coord.merge(east_coord, left_index=True, right_index=True)
    # only keep first instance of coords. Most likely to correlate with genetic ancestry
    coords = coords[(coords['north_coord_instance'] == 0) & (coords['east_coord_instance'] == 0)]    

    self_described_ancestry_wide = self_described_ancestry.reset_index().pivot_table(
        index=['sample_id', 'variable_id', 'index'],
        columns='instance',
        values='value',
        aggfunc='first'  # Use 'first' in case of duplicates
    )

    # Rename the columns to something easier to read
    self_described_ancestry_wide.columns = [f'sire_instance_{col}' for col in self_described_ancestry_wide.columns]

    # Reset index to get a flat DataFrame
    self_described_ancestry_wide = self_described_ancestry_wide.reset_index()

    # Define the columns to check
    instance_cols = ['sire_instance_0', 'sire_instance_1', 'sire_instance_2', 'sire_instance_3']
    # Compute row-wise mode (most common non-NaN value)
    #self_described_ancestry_wide['most_common_ethnicity'] = self_described_ancestry_wide[instance_cols].mode(axis=1)[0]
    self_described_ancestry_wide['most_common_sire'] = self_described_ancestry_wide[instance_cols].apply(fast_mode, axis=1)
    self_described_ancestry_wide = self_described_ancestry_wide.drop(columns=['variable_id', 'index']).set_index('sample_id')
    
    urban_rural = urban_rural.drop(columns=['variable_id','instance', 'index'])
    urban_rural = urban_rural.rename(columns={'value': 'urban / rural'})

    additional_data = self_described_ancestry_wide.merge(coords, left_index=True, right_index=True, how='outer')
    additional_data = additional_data.merge(urban_rural, left_index=True, right_index=True, how='outer')

    # merge all metadata together
    metadata = metadata.merge(additional_data, left_index=True, right_index=True, how='left')

    # Add progress bar using tqdm
    tqdm.pandas(desc="Converting BNG to WGS84")

    # convert Eastings and Northings to lat and long
    if compute_lat_lon:
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(metadata['east_coord'], 
                                         metadata['north_coord'])
        metadata['latitude'] = lat
        metadata['longitude'] = lon
        #other_metadata[['latitude', 'longitude']] = other_metadata.progress_apply(
        #    lambda row: pd.Series(bng_to_wgs84(row['east_coord'], row['north_coord'])), axis=1
        #)
    else:
        # just set to dummy vars to save time
        metadata['latitude'] = 0
        metadata['longitude'] = 0

    # Load plink data
    pedfile = PyPlink(os.path.join(data_root, plink_filename))
    all_samples = pedfile.get_fam()

    # align datasets
    merged_df = metadata.loc[all_samples['iid'].astype(int)]

    def first_non_nan(group):
        if len(group) == 1:  # If there's only one row, return it
            return group.iloc[0]
        return group.iloc[0].where(~group.iloc[0].isna(), group.iloc[1])

    merged_df = merged_df.rename_axis("IDs").reset_index()
    final_df = merged_df.groupby("IDs").apply(first_non_nan)

    # Fix invalid lat/lon
    
    # fix NaN to -1 placeholder
    subset = final_df['north_coord'].isna() | final_df['east_coord'].isna()
    final_df.loc[subset, 'east_coord'] = -1
    final_df.loc[subset, 'north_coord'] = -1

    to_filter = (final_df['east_coord'] == -1) & (final_df['north_coord'] == -1)
    final_df.loc[to_filter, ['latitude', 'longitude']] = np.nan

    # Fill missing values
    for col in ['sire_instance_0', 'sire_instance_1', 'sire_instance_2', 'sire_instance_3', 'most_common_sire']:
        final_df.loc[final_df[col].isna(), col] = 'Do not know'
    final_df.loc[final_df['pop'].isna(), 'pop'] = 'Do not know'

    # Drop unused columns
    final_df = final_df.drop(columns=['s'])
    
    # rename to final name
    final_df = final_df.rename(columns={'most_common_sire': 'self_described_ancestry'})
    
    # Add relatedness set (from JC)
    relateds = pd.read_csv(os.path.join(data_root, 'related_set_to_remove.ids'), 
                           header=None)
    final_df['filter_related'] = final_df['IDs'].isin(relateds[0])

    # reorder columns
    column_order = ['IDs', 'MetadataIDs', 
                    'pop', 'self_described_ancestry', 
                    'age',
                    'latitude', 'longitude',
                    'north_coord_instance', 'north_coord', 'east_coord_instance',
                    'east_coord', 'urban / rural',
                    'filter_related', 'related', 'sex', 'age_sex', 'age2', 'age2_sex',
                    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
                    'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17',
                    'PC18', 'PC19', 'PC20']  # New order
    final_df = final_df[column_order]

    # Output filename
    output_filename = 'UKBB_metadata.csv'
    output_path = os.path.join(data_root, output_filename)
    
    print(f"Saving file to {output_path}")
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UKBB metadata and create derived metadata.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data directory. Assumes all files needed are here.")
    parser.add_argument("--compute_lat_lon", action="store_true", help="Compute latitude/longitude coordinates")
    parser.add_argument("--plink_filename", required=True, help="plink filename (no suffix). Needed for alignment")

    args = parser.parse_args()
    main(args.data_root, args.compute_lat_lon, args.plink_filename)

#python scripts/make_metadata_file_ukbb.py --data_root '/lustre07/scratch/sciclun4/data/UKBB' --plink_filename 'ukb_v2.liftOverCommon.GRCh38.sorted.onlySNPs.woWithdrawn20240613.woSamePos.noDuplicatePos.QC.maxMiss0.5.fixref.onlyPolymorph.noBadPosTopMed.newIDs.common1000G_HGDP.noBadSamples.woLowComplexity_noHLA' --compute_lat_lon 
