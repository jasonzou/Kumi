"""
Enhanced Step 3: CSV Merge

This is an enhanced version of step3_merge.py with:
- Configuration integration
- State management and validation
- Better progress tracking
- Error handling
- Metrics collection

Can be used standalone or as part of the enhanced pipeline.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from state.state_manager import PipelineState


class EnhancedStep3:
    """Enhanced CSV merger with state management"""

    def __init__(self, config: dict):
        """
        Initialize enhanced step3

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.step3_config = config.get('step3', {})
        self.validation_config = config.get('validation', {})
        self.progress_config = config.get('progress', {})

    def get_unique_column_name(self, existing_columns: List[str], base_name: str = "file_name") -> str:
        """
        Generate unique column name

        Args:
            existing_columns: List of existing column names
            base_name: Base name for the column

        Returns:
            Unique column name
        """
        if base_name not in existing_columns:
            return base_name

        counter = 1
        while True:
            new_name = f"{base_name}{counter}"
            if new_name not in existing_columns:
                return new_name
            counter += 1

    def validate_csv(self, filepath: str) -> bool:
        """
        Validate CSV file structure

        Args:
            filepath: Path to CSV file

        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')

            if df.empty:
                print(f"WARNING: Empty CSV file {filepath}")
                return False

            # Check for required columns based on merge strategy
            merge_strategy = self.step3_config.get('merge_strategy', 'concat')

            if merge_strategy == 'concat':
                # For concat, we want to keep all columns
                # Just check that we have at least one column
                if len(df.columns) == 0:
                    print(f"ERROR: No columns in {filepath}")
                    return False
            else:
                # For union, check that we have common columns
                pass

            return True

        except Exception as e:
            print(f"ERROR: Validation failed for {filepath}: {e}")
            return False

    def merge_files(
        self,
        input_folder: str,
        output_folder: str,
        output_filename: str,
        state: PipelineState
    ) -> bool:
        """
        Merge all CSV files in input folder

        Args:
            input_folder: Input folder with CSV files
            output_folder: Output folder for merged file
            output_filename: Name of output file
            state: Pipeline state manager

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all CSV files
            csv_files = list(Path(input_folder).glob("*.csv"))

            if not csv_files:
                print(f"ERROR: No CSV files found in {input_folder}")
                return False

            print(f"Found {len(csv_files)} CSV files to merge")

            # Read and validate all files
            all_dataframes = []
            file_stats = []

            for csv_file in tqdm(csv_files, desc="Reading CSV files"):
                try:
                    filepath = str(csv_file)
                    df = pd.read_csv(filepath, encoding='utf-8-sig')

                    # Validate file
                    if not self.validate_csv(filepath):
                        print(f"WARNING: Skipping invalid file {csv_file.name}")
                        continue

                    # Record file info
                    file_stats.append({
                        "file": csv_file.name,
                        "rows": len(df),
                        "columns": len(df.columns)
                    })

                    # Track source file if configured
                    if self.step3_config.get('track_sources', True):
                        source_column = self.step3_config.get('source_column', 'source_file')
                        df[source_column] = csv_file.stem

                    all_dataframes.append(df)

                except Exception as e:
                    print(f"ERROR: Failed to read {csv_file.name}: {e}")
                    continue

            if not all_dataframes:
                print("ERROR: No valid CSV files to merge")
                return False

            # Merge all DataFrames
            print(f"\nMerging {len(all_dataframes)} files...")

            merge_strategy = self.step3_config.get('merge_strategy', 'concat')

            if merge_strategy == 'concat':
                # Keep all columns
                merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
            else:
                # Only keep common columns (union)
                merged_df = pd.concat(all_dataframes, ignore_index=True, sort=True)

            # Ensure file_name column is first if track_sources is enabled
            if self.step3_config.get('track_sources', True):
                source_column = self.step3_config.get('source_column', 'source_file')

                # Get unique column name
                unique_col = self.get_unique_column_name(
                    list(merged_df.columns),
                    source_column
                )

                # Rename if needed
                if unique_col != source_column:
                    merged_df.rename(columns={source_column: unique_col}, inplace=True)

                # Move to first position
                cols = merged_df.columns.tolist()
                cols.remove(unique_col)
                cols = [unique_col] + cols
                merged_df = merged_df[cols]

            # Create output directory
            os.makedirs(output_folder, exist_ok=True)

            # Save merged file
            output_filepath = os.path.join(output_folder, output_filename)
            merged_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

            # Print summary
            print(f"\n{'='*60}")
            print("Merge Summary")
            print(f"{'='*60}")
            print(f"Files merged: {len(file_stats)}")
            print(f"Total rows: {len(merged_df)}")
            print(f"Total columns: {len(merged_df.columns)}")
            print(f"Output file: {output_filepath}")
            print(f"{'='*60}\n")

            # Print column list
            print("Columns in merged file:")
            for i, col in enumerate(merged_df.columns, 1):
                print(f"  {i}. {col}")
            print()

            return True

        except Exception as e:
            print(f"ERROR: Merge failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(
        self,
        input_folder: str,
        output_folder: str,
        output_filename: str,
        state: PipelineState,
        preview: bool = False
    ) -> dict:
        """
        Run step 3 merging

        Args:
            input_folder: Input folder with QA CSV files
            output_folder: Output folder for merged file
            output_filename: Name of output file
            state: Pipeline state manager
            preview: If True, don't actually process files

        Returns:
            Dictionary with statistics
        """
        stats = {
            "files_found": 0,
            "files_valid": 0,
            "files_invalid": 0,
            "files_merged": 0,
            "total_rows": 0,
            "total_columns": 0,
            "errors": []
        }

        # Get all CSV files
        csv_files = list(Path(input_folder).glob("*.csv"))
        stats["files_found"] = len(csv_files)

        if not csv_files:
            print(f"WARNING: No CSV files found in {input_folder}")
            return stats

        print(f"\n{'='*60}")
        print("Step 3: CSV Merge")
        print(f"{'='*60}")
        print(f"Total CSV files found: {stats['files_found']}")
        print(f"{'='*60}\n")

        if preview:
            print("PREVIEW MODE: No files will be processed")
            for csv_file in csv_files:
                print(f"  Would merge: {csv_file.name}")
            return stats

        # Check if all step2 files are complete
        files_ready = True
        for csv_file in csv_files:
            # Check state
            original_file = csv_file.stem + '.md'  # or .txt
            file_status = state.get_file_status(original_file)

            if not file_status.get('step2_completed'):
                print(f"WARNING: {original_file} not complete for step2, skipping in merge")
                files_ready = False

        if not files_ready:
            print("ERROR: Not all files are ready for merge")
            stats["errors"].append("Not all files completed step2")
            return stats

        # Perform merge
        success = self.merge_files(input_folder, output_folder, output_filename, state)

        if success:
            # Get stats from output file
            output_filepath = os.path.join(output_folder, output_filename)
            if os.path.exists(output_filepath):
                merged_df = pd.read_csv(output_filepath, encoding='utf-8-sig')
                stats["files_merged"] = len(csv_files)
                stats["files_valid"] = len(csv_files)
                stats["total_rows"] = len(merged_df)
                stats["total_columns"] = len(merged_df.columns)

                # Update state for all files
                for csv_file in csv_files:
                    original_file = csv_file.stem + '.md'
                    state.update_file(
                        filepath=original_file,
                        checksum=state.calculate_checksum(original_file),
                        status="complete",
                        step_completed=3
                    )
        else:
            stats["files_invalid"] = len(csv_files)
            stats["errors"].append("Merge operation failed")

        print(f"\n{'='*60}")
        print("Step 3 Complete")
        print(f"{'='*60}")
        print(f"Files found: {stats['files_found']}")
        print(f"Files merged: {stats['files_merged']}")
        print(f"Total rows: {stats['total_rows']}")
        print(f"Total columns: {stats['total_columns']}")
        print(f"{'='*60}\n")

        return stats


def main():
    """Main entry point for standalone execution"""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Initialize state
    state_file = config.get('state', {}).get('file', './state/pipeline_state.json')
    state = PipelineState(state_file)

    # Get paths
    input_folder = os.path.join(config.get('output', {}).get('folder', './output'), 'QA')
    output_folder = config.get('output', {}).get('folder', './output')
    output_filename = config.get('pipeline', {}).get('data_name', 'dataset') + '.csv'

    # Create output subfolder for step3
    data_name = config.get('pipeline', {}).get('data_name', 'dataset')
    input_folder = os.path.join(input_folder, data_name)
    output_folder = os.path.join(output_folder, 'merged')

    # Run step3
    step3 = EnhancedStep3(config)
    stats = step3.run(input_folder, output_folder, output_filename, state, preview=config.get('pipeline', {}).get('preview', False))

    return stats


if __name__ == "__main__":
    main()
