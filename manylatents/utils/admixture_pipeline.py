"""
Admixture processing pipeline for converting neural admixture outputs
to ManyLatents format and generating quality control visualizations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AdmixtureProcessor:
    """Process neural admixture outputs for different datasets (UKBB, HGDP+1KGP)."""
    
    def __init__(self, dataset_type: str = "UKBB", k_range: Tuple[int, int] = (2, 10),
                 q_train_prefix: str = "neuralAdmixture", q_test_prefix: str = "random_data_unseen"):
        """
        Initialize admixture processor.

        Args:
            dataset_type: Type of dataset ('UKBB' or 'HGDP')
            k_range: Range of K values to process (min_k, max_k)
            q_train_prefix: Prefix for training Q files
            q_test_prefix: Prefix for test Q files
        """
        self.dataset_type = dataset_type
        self.k_min, self.k_max = k_range
        self.q_train_prefix = q_train_prefix
        self.q_test_prefix = q_test_prefix
        self.required_files = {}
        self.processed_data = {}
        
    def validate_files(self, admixture_dir: str, metadata_path: str, 
                      samples_train: Optional[str] = None, 
                      samples_test: Optional[str] = None) -> bool:
        """
        Validate that all required files exist.
        
        Args:
            admixture_dir: Directory containing neural admixture outputs
            metadata_path: Path to metadata CSV file
            samples_train: Path to samples.txt (optional, auto-detect if None)
            samples_test: Path to samples_unseen.txt (optional, auto-detect if None)
            
        Returns:
            True if all required files exist, False otherwise
        """
        admix_path = Path(admixture_dir)
        
        # Check metadata file
        if not Path(metadata_path).exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
            
        self.required_files['metadata'] = metadata_path
        
        # Auto-detect samples files if not provided
        if samples_train is None:
            samples_train = admix_path / "samples.txt"
        if samples_test is None:
            samples_test = admix_path / "samples_unseen.txt"
            
        # Check samples files
        for samples_file, name in [(samples_train, 'train'), (samples_test, 'test')]:
            if not Path(samples_file).exists():
                logger.error(f"Samples file not found: {samples_file}")
                return False
            self.required_files[f'samples_{name}'] = str(samples_file)
        
        # Check neural admixture Q files for each K
        missing_train_files = []
        missing_test_files = []
        for k in range(self.k_min, self.k_max + 1):
            # Check training Q file
            q_train_file = admix_path / f"{self.q_train_prefix}.{k}.Q"
            if not q_train_file.exists():
                missing_train_files.append(str(q_train_file))
            else:
                self.required_files[f'admixture_train_k{k}'] = str(q_train_file)

            # Check test Q file
            q_test_file = admix_path / f"{self.q_test_prefix}.{k}.Q"
            if not q_test_file.exists():
                missing_test_files.append(str(q_test_file))
            else:
                self.required_files[f'admixture_test_k{k}'] = str(q_test_file)

        # Raise errors for missing files
        if missing_train_files:
            logger.error(f"Missing training Q files: {missing_train_files}")
            return False

        if missing_test_files:
            logger.error(f"Missing test Q files: {missing_test_files}")
            return False
            
        # Store admixture directory
        self.required_files['admixture_dir'] = str(admix_path)
        
        logger.info(f"All required files validated for {self.dataset_type}")
        return True
        
    def _load_fam_files(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load .fam files to get sample ID ordering."""
        admix_dir = Path(self.required_files['admixture_dir'])
        
        # Find .fam files - pattern varies by dataset
        if self.dataset_type == "UKBB":
            fam_patterns = [
                "random*EUR_and_others.fam",
                "random*EUR_and_others_unseen.fam"
            ]
        else:  # HGDP
            fam_patterns = [
                "*.fam",
                "*_unseen.fam"
            ]
            
        fam_files = []
        for pattern in fam_patterns:
            matches = list(admix_dir.glob(pattern))
            fam_files.extend(matches)
            
        if len(fam_files) < 2:
            logger.warning("Could not find both train and test .fam files, using samples.txt files instead")
            return self._load_samples_txt()
            
        # Load fam files (space-separated, no header, first column is ID)
        fam_train = pd.read_csv(fam_files[0], sep=' ', header=None).rename(columns={0: 'ID'})
        fam_test = pd.read_csv(fam_files[1], sep=' ', header=None).rename(columns={0: 'ID'})
        
        return fam_train, fam_test
        
    def _load_samples_txt(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load samples.txt files as fallback."""
        samples_train = pd.read_csv(self.required_files['samples_train'], header=None).rename(columns={0: 'ID'})
        samples_test = pd.read_csv(self.required_files['samples_test'], header=None).rename(columns={0: 'ID'})
        return samples_train, samples_test
        
    def process_admixture_data(self) -> Dict[int, pd.DataFrame]:
        """
        Process neural admixture data for all K values.
        
        Returns:
            Dictionary mapping K values to processed DataFrames
        """
        # Load metadata
        metadata = pd.read_csv(self.required_files['metadata'])
        logger.info(f"Loaded metadata with {len(metadata)} samples")
        
        # Load sample ordering
        try:
            fam_train, fam_test = self._load_fam_files()
            logger.info(f"Loaded FAM files: {len(fam_train)} train, {len(fam_test)} test samples")
        except Exception as e:
            logger.error(f"Error loading sample files: {e}")
            return {}
            
        admix_dir = Path(self.required_files['admixture_dir'])
        processed_data = {}
        
        for k in range(self.k_min, self.k_max + 1):
            train_key = f'admixture_train_k{k}'
            test_key = f'admixture_test_k{k}'

            if train_key not in self.required_files or test_key not in self.required_files:
                logger.error(f"Skipping K={k}, required files not found in validation")
                continue

            try:
                # Load Q files for train and test
                q_train = pd.read_csv(self.required_files[train_key], sep=' ', header=None)
                q_test = pd.read_csv(self.required_files[test_key], sep=' ', header=None)

                logger.info(f"K={k}: Loaded train ({len(q_train)}) and test ({len(q_test)}) Q files")
                
                # Combine train and test data with sample IDs
                q_train_with_ids = pd.concat([q_train, fam_train[['ID']]], axis=1)
                q_test_with_ids = pd.concat([q_test, fam_test[['ID']]], axis=1)
                q_combined = pd.concat([q_train_with_ids, q_test_with_ids], ignore_index=True)
                
                # Merge with metadata
                metadata_cols = self._get_metadata_columns()
                q_merged = pd.merge(q_combined, metadata[metadata_cols], 
                                  left_on='ID', right_on=metadata_cols[0])
                
                # Create final dataframe in exact HGDP format: sample_id, component0, component1, ..., population, super_population
                id_col = metadata_cols[0]
                
                # Get component columns (they are integers: 0, 1, 2, ...)
                component_cols = list(range(k))  # [0, 1, 2, ..., k-1]
                
                # Get metadata columns (excluding the ID column)
                metadata_cols_clean = [col for col in metadata_cols if col != id_col]
                
                # Create column order: [ID] + [components] + [metadata]
                column_order = ['ID'] + component_cols + metadata_cols_clean
                
                # Reorder columns
                final_df = q_merged[column_order]
                
                # Clean and format
                final_df = final_df.dropna()
                processed_data[k] = final_df
                
                logger.info(f"K={k}: Processed {len(final_df)} samples")
                
            except Exception as e:
                logger.error(f"Error processing K={k}: {e}")
                continue
                
        self.processed_data = processed_data
        return processed_data
        
    def _get_metadata_columns(self) -> List[str]:
        """Get relevant metadata columns based on dataset type."""
        if self.dataset_type == "UKBB":
            return ['IDs', 'Population', 'self_described_ancestry']
        else:  # HGDP
            return ['sample', 'population_name', 'super_population']
            
    def save_processed_data(self, output_dir: str, prefix: Optional[str] = None) -> List[str]:
        """
        Save processed admixture data to TSV files.
        
        Args:
            output_dir: Output directory
            prefix: File prefix (defaults to dataset_type)
            
        Returns:
            List of saved file paths
        """
        if not self.processed_data:
            logger.error("No processed data to save. Run process_admixture_data() first.")
            return []
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = self.dataset_type
            
        saved_files = []
        
        for k, df in self.processed_data.items():
            output_file = output_path / f"{prefix}.{k}_metadata.tsv"
            # Save without headers and without index, exactly like HGDP format
            df.to_csv(output_file, sep='\t', header=False, index=False)
            saved_files.append(str(output_file))
            logger.info(f"Saved K={k} data to {output_file} ({len(df)} samples, {len(df.columns)} columns)")
            
        return saved_files
        

class AdmixtureVisualizer:
    """Generate admixture visualization plots for quality control."""
    
    def __init__(self, dataset_type: str = "UKBB"):
        self.dataset_type = dataset_type
        
    def load_processed_admixture(self, data_dir: str, prefix: str, 
                                k_values: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Load processed admixture data from TSV files.
        
        Args:
            data_dir: Directory containing processed TSV files
            prefix: File prefix 
            k_values: List of K values to load
            
        Returns:
            Dictionary mapping K to DataFrames
        """
        data_path = Path(data_dir)
        admixture_data = {}
        
        for k in k_values:
            file_path = data_path / f"{prefix}.{k}_metadata.tsv"
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                # Load data without headers, columns will be integers: 0, 1, 2, ..., k-1, k, k+1, ...
                df = pd.read_csv(file_path, sep='\t', header=None)
                
                # No need to rename columns - keep them as integers
                # Structure: column 0 = sample_id, columns 1...k = components, columns k+1... = metadata
                        
                df = df.dropna()
                admixture_data[k] = df
                logger.info(f"Loaded K={k}: {df.shape[0]} samples, {df.shape[1]} total columns")
                
            except Exception as e:
                logger.error(f"Error loading K={k}: {e}")
                
        return admixture_data
        
    def plot_admixture_barplot(self, df: pd.DataFrame, group_col: str, 
                              ax: Optional[plt.Axes] = None, 
                              colors: Optional[List[str]] = None) -> plt.Axes:
        """
        Create admixture bar plot for a single K value.
        
        Args:
            df: DataFrame with admixture data
            group_col: Column to group by (population)
            ax: Matplotlib axes (optional)
            colors: List of colors for components (optional)
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 4))
            
        # Get component columns (they are the integer-named columns)
        component_cols = [col for col in df.columns if isinstance(col, int)]
        component_cols = sorted(component_cols)  # Sort to ensure proper order (0, 1, 2, ...)
        
        # Plot stacked bar chart
        if colors and len(colors) >= len(component_cols):
            df[component_cols].plot(kind='bar', stacked=True, ax=ax, 
                                  width=1.0, edgecolor='none', color=colors[:len(component_cols)])
        else:
            df[component_cols].plot(kind='bar', stacked=True, ax=ax, 
                                  width=1.0, edgecolor='none')
        
        # Remove x-axis labels and legend
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.get_legend().remove()
        
        # Add population separators and labels
        self._add_population_labels(ax, df, group_col)
        
        return ax
        
    def _add_population_labels(self, ax: plt.Axes, df: pd.DataFrame, group_col: str):
        """Add population labels and separators to plot."""
        # Calculate population boundaries
        grouped = df.groupby(group_col, sort=False)
        cutoffs = [0]
        pop_labels = []
        
        for pop_name, group in grouped:
            cutoffs.append(cutoffs[-1] + len(group))
            pop_labels.append(pop_name)
            
        # Add vertical lines between populations
        for pos in cutoffs[1:-1]:
            ax.axvline(x=pos-0.5, linestyle='--', color='black', alpha=0.7)
            
        # Add population labels
        midpoints = [(cutoffs[i] + cutoffs[i + 1]) / 2 for i in range(len(cutoffs) - 1)]
        ax.set_xticks(midpoints)
        ax.set_xticklabels(pop_labels, rotation=45, ha='right')
        
    def _get_geographic_order(self, dataset_type: str) -> List[str]:
        """Get geographic ordering for populations."""
        if dataset_type == "UKBB":
            return [
                # African populations
                'African', 'Caribbean', 'Any other Black background', 'Black or Black British',
                'White and Black African', 'White and Black Caribbean',
                # Mixed populations
                'Any other mixed background', 'Mixed',
                # European populations  
                'British', 'Irish', 'White', 'Any other white background', 'White and Asian',
                # Middle Eastern/Other
                'Any other ethnic group', 'Other ethnic group',
                # Asian populations
                'Indian', 'Pakistani', 'Bangladeshi', 'Any other Asian background',
                'Asian or Asian British', 'Chinese',
                # Unknown
                'Do not know', 'Prefer not to answer', 'Other'
            ]
        else:  # HGDP
            return [
                'AFR', 'AMR', 'EUR', 'EAS', 'SAS', 'OCE', 'MID'
            ]
            
    def sort_by_geography(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Sort DataFrame by geographic ordering."""
        geographic_order = self._get_geographic_order(self.dataset_type)
        available_pops = set(df[group_col].unique())
        
        # Create ordered list
        ordered_pops = []
        for pop in geographic_order:
            if pop in available_pops:
                ordered_pops.append(pop)
                
        # Add any missing populations
        for pop in available_pops:
            if pop not in ordered_pops:
                ordered_pops.append(pop)
                logger.warning(f"Population '{pop}' not in predefined order, adding at end")
                
        # Sort DataFrame
        sorted_data = pd.DataFrame()
        for pop in ordered_pops:
            pop_data = df[df[group_col] == pop]
            # Sort within population by component columns (integer-named columns)
            component_cols = [col for col in df.columns if isinstance(col, int)]
            component_cols = sorted(component_cols)
            pop_data = pop_data.sort_values(component_cols)
            sorted_data = pd.concat([sorted_data, pop_data], axis=0)
            
        return sorted_data
        
    def create_multi_k_plot(self, admixture_data: Dict[int, pd.DataFrame], 
                           output_path: str, colors: Optional[List[str]] = None,
                           subsample_per_group: int = 300) -> str:
        """
        Create multi-K admixture plot.
        
        Args:
            admixture_data: Dictionary mapping K to DataFrames
            output_path: Path to save plot
            colors: Color palette for components
            subsample_per_group: Maximum samples per population group
            
        Returns:
            Path to saved plot
        """
        if not admixture_data:
            raise ValueError("No admixture data provided")
            
        # Default colors
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
        k_values = sorted(admixture_data.keys())
        n_plots = len(k_values)
        
        fig, axes = plt.subplots(nrows=n_plots, figsize=(15, n_plots * 3), sharex=True)
        if n_plots == 1:
            axes = [axes]
            
        for i, k in enumerate(k_values):
            df = admixture_data[k].copy()
            
            # Determine grouping column by actual column names
            # Column structure: sample_id, component0, component1, ..., population, super_population/ancestry
            columns = list(df.columns)
            logger.debug(f"DataFrame has {len(columns)} columns for K={k}")
            logger.debug(f"Column names: {columns}")
            logger.debug(f"Sample of data:\n{df.head(2)}")
            
            # Use the actual column name, not positional index
            if self.dataset_type == "UKBB":
                if 'self_described_ancestry' in columns:
                    group_col = 'self_described_ancestry'
                elif 'Population' in columns:
                    group_col = 'Population'
                else:
                    logger.error(f"No suitable grouping column found in {columns}")
                    continue
            else:  # HGDP
                # Look for HGDP column names
                if 'super_population' in columns:
                    group_col = 'super_population'
                elif 'population_name' in columns:
                    group_col = 'population_name'
                else:
                    logger.error(f"No suitable grouping column found in {columns}")
                    continue
                
            logger.debug(f"Using group_col = '{group_col}'")
                
            # Check that the column exists
            if group_col not in df.columns:
                logger.error(f"Group column '{group_col}' not found in DataFrame columns: {columns}")
                continue
                
            # Subsample if needed
            if subsample_per_group:
                df = self._subsample_populations(df, group_col, subsample_per_group)
                
            # Sort geographically
            df = self.sort_by_geography(df, group_col)
            
            # Plot
            self.plot_admixture_barplot(df, group_col, ax=axes[i], colors=colors)
            axes[i].set_ylabel(f'K={k}', fontsize=12)
            axes[i].set_yticks([])
            
            # Remove x-axis labels except for last plot
            if i < n_plots - 1:
                axes[i].set_xticks([])
                axes[i].set_xticklabels([])
                
        fig.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved multi-K plot to {output_path}")
        return output_path
        
    def _subsample_populations(self, df: pd.DataFrame, group_col: str, 
                              max_per_group: int) -> pd.DataFrame:
        """Subsample populations to maximum number per group."""
        subsampled_groups = []
        
        for group_name, group_data in df.groupby(group_col):
            if len(group_data) > max_per_group:
                sampled_data = group_data.sample(n=max_per_group, random_state=42)
                logger.info(f"Subsampled {group_name}: {len(group_data)} → {len(sampled_data)}")
            else:
                sampled_data = group_data
                
            subsampled_groups.append(sampled_data)
            
        return pd.concat(subsampled_groups, axis=0)


def run_admixture_pipeline(admixture_dir: str, metadata_path: str,
                          output_dir: str, dataset_type: str = "UKBB",
                          k_range: Tuple[int, int] = (2, 10),
                          create_plots: bool = True,
                          samples_train: Optional[str] = None,
                          samples_test: Optional[str] = None,
                          q_train_prefix: str = "neuralAdmixture",
                          q_test_prefix: str = "random_data_unseen") -> Dict[str, any]:
    """
    Run complete admixture processing pipeline.

    Args:
        admixture_dir: Directory with neural admixture outputs
        metadata_path: Path to metadata CSV
        output_dir: Output directory for processed files and plots
        dataset_type: 'UKBB' or 'HGDP'
        k_range: Range of K values (min, max)
        create_plots: Whether to create visualization plots
        samples_train: Path to samples.txt (optional)
        samples_test: Path to samples_unseen.txt (optional)
        q_train_prefix: Prefix for training Q files
        q_test_prefix: Prefix for test Q files

    Returns:
        Dictionary with pipeline results
    """
    logger.info(f"Starting admixture pipeline for {dataset_type}")
    
    # Initialize processor
    processor = AdmixtureProcessor(dataset_type, k_range, q_train_prefix, q_test_prefix)
    
    # Validate files
    if not processor.validate_files(admixture_dir, metadata_path, samples_train, samples_test):
        raise ValueError("File validation failed")
        
    # Process data
    processed_data = processor.process_admixture_data()
    if not processed_data:
        raise ValueError("Data processing failed")
        
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = dataset_type
    saved_files = processor.save_processed_data(str(output_path), prefix)
    
    results = {
        'dataset_type': dataset_type,
        'k_values': list(processed_data.keys()),
        'processed_files': saved_files,
        'output_dir': str(output_path)
    }
    
    # Create visualizations
    if create_plots:
        visualizer = AdmixtureVisualizer(dataset_type)
        
        # Create multi-K plot
        plot_path = output_path / f"{dataset_type}_admixture_plot.png"
        visualizer.create_multi_k_plot(processed_data, str(plot_path))
        results['plot_path'] = str(plot_path)
        
    logger.info(f"Pipeline completed successfully. Output: {output_dir}")
    return results