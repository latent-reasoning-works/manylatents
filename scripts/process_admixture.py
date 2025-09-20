#!/usr/bin/env python3
"""
Command-line script to process neural admixture outputs.

This script converts neural admixture outputs to ManyLatents format and generates
quality control visualizations.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from manylatents.utils.admixture_pipeline import run_admixture_pipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process neural admixture outputs for ManyLatents framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process UKBB data with default K range (2-10)
  python process_admixture.py \\
    --admixture-dir data/UKBB/admixture/neural_admixture_gpus_UKB.random2.5k_EUR_1.6.1 \\
    --metadata data/UKBB/UKBB_metadata.csv \\
    --output-dir data/UKBB/admixture/neural_admixture_gpus_UKB.random2.5k_EUR_1.6.1 \\
    --dataset-type UKBB

  # Process HGDP data with custom K range
  python process_admixture.py \\
    --admixture-dir data/HGDP+1KGP/admixture/neural_outputs \\
    --metadata data/HGDP+1KGP/gnomad_derived_metadata_with_filtered_sampleids.csv \\
    --output-dir data/HGDP+1KGP/admixture/neural_outputs \\
    --dataset-type HGDP \\
    --k-min 2 --k-max 8

  # Process with custom samples files
  python process_admixture.py \\
    --admixture-dir data/UKBB/admixture/neural_admixture_gpus_UKB.random5k_EUR_1.6.1 \\
    --metadata data/UKBB/UKBB_metadata.csv \\
    --output-dir data/UKBB/admixture/neural_admixture_gpus_UKB.random5k_EUR_1.6.1 \\
    --dataset-type UKBB \\
    --samples-train data/UKBB/admixture/neural_admixture_gpus_UKB.random5k_EUR_1.6.1/samples.txt \\
    --samples-test data/UKBB/admixture/neural_admixture_gpus_UKB.random5k_EUR_1.6.1/samples_unseen.txt
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--admixture-dir',
        type=str,
        required=True,
        help='Directory containing neural admixture outputs (neuralAdmixture.K.Q files)'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed files and plots'
    )
    
    # Optional arguments
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['UKBB', 'HGDP'],
        default='UKBB',
        help='Dataset type (default: UKBB)'
    )
    
    parser.add_argument(
        '--k-min',
        type=int,
        default=2,
        help='Minimum K value to process (default: 2)'
    )
    
    parser.add_argument(
        '--k-max',
        type=int,
        default=10,
        help='Maximum K value to process (default: 10)'
    )
    
    parser.add_argument(
        '--samples-train',
        type=str,
        help='Path to samples.txt (training samples). Auto-detected if not provided.'
    )
    
    parser.add_argument(
        '--samples-test',
        type=str,
        help='Path to samples_unseen.txt (test samples). Auto-detected if not provided.'
    )

    parser.add_argument(
        '--q-train-prefix',
        type=str,
        default='neuralAdmixture',
        help='Prefix for training Q files (default: neuralAdmixture). Files expected: {prefix}.{k}.Q'
    )

    parser.add_argument(
        '--q-test-prefix',
        type=str,
        default='random_data_unseen',
        help='Prefix for test Q files (default: random_data_unseen). Files expected: {prefix}.{k}.Q'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    admixture_path = Path(args.admixture_dir)
    if not admixture_path.exists():
        logger.error(f"Admixture directory does not exist: {args.admixture_dir}")
        return 1
        
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error(f"Metadata file does not exist: {args.metadata}")
        return 1
        
    if args.k_min > args.k_max:
        logger.error(f"k-min ({args.k_min}) must be <= k-max ({args.k_max})")
        return 1
        
    # Run pipeline
    try:
        results = run_admixture_pipeline(
            admixture_dir=args.admixture_dir,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            k_range=(args.k_min, args.k_max),
            create_plots=not args.no_plots,
            samples_train=args.samples_train,
            samples_test=args.samples_test,
            q_train_prefix=args.q_train_prefix,
            q_test_prefix=args.q_test_prefix
        )
        
        # Print summary
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Dataset type: {results['dataset_type']}")
        logger.info(f"K values processed: {results['k_values']}")
        logger.info(f"Output directory: {results['output_dir']}")
        logger.info(f"Processed files: {len(results['processed_files'])}")
        
        if 'plot_path' in results:
            logger.info(f"Visualization plot: {results['plot_path']}")
            
        logger.info("\nProcessed files:")
        for file_path in results['processed_files']:
            logger.info(f"  - {file_path}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())