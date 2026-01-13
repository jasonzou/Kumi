"""
Enhanced Step 2: CSV to QA Generation

This is an enhanced version of step2_chunk2qa.py with:
- Configuration integration
- State management and resume capability
- Validation and error handling
- Progress tracking
- Metrics collection
- Better retry logic for API calls

Can be used standalone or as part of the enhanced pipeline.
"""

import os
import sys
import pandas as pd
import re
import time
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from state.state_manager import PipelineState
from prompts import SYS_ED_TEMPLATE, ED_TEMPLATE


class EnhancedStep2:
    """Enhanced CSV to QA generator with state management"""

    def __init__(self, config: dict):
        """
        Initialize enhanced step2

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.step2_config = config.get('step2', {})
        self.validation_config = config.get('validation', {})
        self.progress_config = config.get('progress', {})

        # Compile QA pattern
        self.qa_pattern = re.compile(r"-?\s*question: (.*?)\s*-?\s*answer: (.+)", re.DOTALL)

        # Statistics
        self.stats = {
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "api_calls_made": 0,
            "api_retries": 0,
            "total_tokens": 0
        }

    def generate_with_retry(
        self,
        prompt: str,
        system: str = None,
        max_retries: int = None
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Generate QA with retry logic

        Args:
            prompt: User prompt
            system: System prompt
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (response, usage_info)
        """
        max_retries = max_retries or self.step2_config.get('api_retries', 3)
        retry_delay = self.step2_config.get('api_retry_delay', 5)
        retry_backoff = self.step2_config.get('api_retry_backoff', 2)

        for attempt in range(max_retries + 1):
            try:
                response, usage_info = self._generate(
                    prompt=prompt,
                    system=system,
                    temperature=self.step2_config.get('temperature', 0.7)
                )

                self.stats["api_calls_made"] += 1
                if usage_info and 'total_tokens' in usage_info:
                    self.stats["total_tokens"] += usage_info['total_tokens']

                return response, usage_info

            except Exception as e:
                if attempt < max_retries:
                    self.stats["api_retries"] += 1
                    wait_time = retry_delay * (retry_backoff ** attempt)
                    print(f"  API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  API call failed after {max_retries + 1} attempts: {e}")
                    return None, None

        return None, None

    def _generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> Tuple[str, Optional[dict]]:
        """
        Generate response using API (placeholder - integrate with actual API)

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens

        Returns:
            Tuple of (response, usage_info)
        """
        # Import API function
        from step2_chunk2qa import generate

        response, usage_info = generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response, usage_info

    def process_row(
        self,
        index: int,
        row: pd.Series,
        filename: str,
        rounds: int = None
    ) -> Tuple[int, dict]:
        """
        Process a single row to generate QA

        Args:
            index: Row index
            row: DataFrame row
            filename: Source filename
            rounds: Number of generation rounds

        Returns:
            Tuple of (index, row_data)
        """
        rounds = rounds or self.step2_config.get('rounds', 3)

        text = row['Text_pure']
        if not isinstance(text, str):
            print(f"Warning: Row {index} in {filename} has non-string text")
            row_data = row.to_dict()
            row_data['Question'] = ''
            row_data['Answer'] = ''
            return index, row_data

        best_question = ''
        best_answer = ''
        best_length = 0

        # Multi-round generation
        for round_num in range(rounds):
            try:
                sys_prompt = SYS_ED_TEMPLATE
                prompt = ED_TEMPLATE.format(text=text)

                response, usage_info = self.generate_with_retry(
                    prompt=prompt,
                    system=sys_prompt
                )

                if response:
                    match = self.qa_pattern.search(response)
                    if match:
                        question = match.group(1).strip()
                        answer = match.group(2).strip()
                        length = len(question) + len(answer)

                        if length > best_length:
                            best_question = question
                            best_answer = answer
                            best_length = length

            except Exception as e:
                print(f"  Error in round {round_num + 1}: {e}")
                continue

        # Build return data
        row_data = row.to_dict()
        row_data['Question'] = best_question
        row_data['Answer'] = best_answer

        self.stats["rows_processed"] += 1
        if best_question and best_answer:
            self.stats["rows_succeeded"] += 1
        else:
            self.stats["rows_failed"] += 1

        return index, row_data

    def validate_qa_pair(self, question: str, answer: str) -> bool:
        """
        Validate a QA pair

        Args:
            question: Question string
            answer: Answer string

        Returns:
            True if valid, False otherwise
        """
        if not question or not answer:
            return False

        if len(question.strip()) < 5:
            return False

        if len(answer.strip()) < 10:
            return False

        # Check for reasonable length ratio
        if len(answer) / len(question) > 50:
            return False

        return True

    def run(
        self,
        input_folder: str,
        output_folder: str,
        state: PipelineState,
        preview: bool = False
    ) -> dict:
        """
        Run step 2 processing

        Args:
            input_folder: Input folder path (CSV files from step1)
            output_folder: Output folder path
            state: Pipeline state manager
            preview: If True, don't actually process files

        Returns:
            Dictionary with statistics
        """
        stats = {
            "files_processed": 0,
            "files_succeeded": 0,
            "files_failed": 0,
            "files_skipped": 0,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "qa_pairs_generated": 0,
            "api_calls_made": 0,
            "api_retries": 0,
            "errors": []
        }

        # Get all CSV files
        csv_files = list(Path(input_folder).glob("*.csv"))

        if not csv_files:
            print(f"WARNING: No CSV files found in {input_folder}")
            return stats

        # Filter files based on state
        files_to_process = []
        for csv_file in csv_files:
            # Check if step1 is complete for this file
            original_file = str(csv_file.with_suffix('.md'))  # or .txt
            file_status = state.get_file_status(original_file)

            if file_status.get('step1_completed'):
                # Check if step2 is already complete
                if not file_status.get('step2_completed'):
                    files_to_process.append(str(csv_file))

        if not files_to_process:
            print("All files already processed for Step 2")
            return stats

        print(f"\n{'='*60}")
        print("Step 2: CSV to QA Generation")
        print(f"{'='*60}")
        print(f"Total CSV files found: {len(csv_files)}")
        print(f"Files to process: {len(files_to_process)}")
        print(f"{'='*60}\n")

        if preview:
            print("PREVIEW MODE: No files will be processed")
            for filename in files_to_process:
                print(f"  Would process: {Path(filename).name}")
            return stats

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Process each file
        max_workers = self.step2_config.get('max_workers', 5)

        for csv_file_path in files_to_process:
            filename = Path(csv_file_path).name
            print(f"\nProcessing: {filename}")

            try:
                # Load input CSV
                df_input = pd.read_csv(csv_file_path, encoding='utf-8-sig')

                # Check if output file exists
                output_file_path = os.path.join(output_folder, filename)

                if os.path.exists(output_file_path):
                    # Load existing output
                    df_output = pd.read_csv(output_file_path, encoding='utf-8-sig')

                    # Find rows that need processing
                    rows_to_process = []
                    for index, row in df_input.iterrows():
                        # Check if this row already has QA
                        if index < len(df_output):
                            existing_q = df_output.iloc[index].get('Question', '')
                            existing_a = df_output.iloc[index].get('Answer', '')

                            if pd.notna(existing_q) and pd.notna(existing_a) and existing_q and existing_a:
                                # Row already processed
                                continue

                        rows_to_process.append((index, row))

                else:
                    # No existing output, process all rows
                    rows_to_process = [(index, row) for index, row in df_input.iterrows()]
                    df_output = pd.DataFrame(columns=['Question', 'Answer'])

                if not rows_to_process:
                    print(f"  All rows already processed")
                    stats["files_succeeded"] += 1
                    continue

                print(f"  Processing {len(rows_to_process)} rows with {max_workers} workers")

                # Reset stats for this file
                file_stats = {
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 0
                }

                # Process rows with threading
                results = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(
                            self.process_row,
                            index,
                            row,
                            filename,
                            self.step2_config.get('rounds', 3)
                        ): index
                        for index, row in rows_to_process
                    }

                    for future in tqdm(
                        as_completed(future_to_index),
                        total=len(future_to_index),
                        desc=f"  Processing",
                        unit="row",
                        disable=not self.progress_config.get('enabled', True)
                    ):
                        try:
                            index, row_data = future.result()
                            results[index] = row_data
                            file_stats["processed"] += 1

                            if row_data.get('Question') and row_data.get('Answer'):
                                file_stats["succeeded"] += 1
                            else:
                                file_stats["failed"] += 1

                        except Exception as e:
                            index = future_to_index[future]
                            print(f"  Error processing row {index}: {e}")
                            file_stats["failed"] += 1
                            results[index] = df_input.loc[index].to_dict()
                            results[index]['Question'] = ''
                            results[index]['Answer'] = ''

                # Update output DataFrame
                for index in sorted(results.keys()):
                    row_data = results[index]

                    if index < len(df_output):
                        df_output.iloc[index] = row_data
                    else:
                        # Add new row
                        df_output = pd.concat([df_output, pd.DataFrame([row_data])], ignore_index=True)

                # Save output
                df_output.to_csv(output_file_path, index=False, encoding='utf-8-sig')

                # Update global stats
                stats["files_processed"] += 1
                stats["files_succeeded"] += 1
                stats["rows_processed"] += file_stats["processed"]
                stats["rows_succeeded"] += file_stats["succeeded"]
                stats["rows_failed"] += file_stats["failed"]
                stats["qa_pairs_generated"] += file_stats["succeeded"]

                print(f"  ✓ Completed: {file_stats['succeeded']}/{file_stats['processed']} rows succeeded")

                # Update state for this file
                original_file = Path(csv_file_path).stem + '.md'  # or .txt
                state.update_file(
                    filepath=original_file,
                    checksum=state.calculate_checksum(original_file),
                    status="complete",
                    step_completed=2
                )

            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                print(f"  ✗ {error_msg}")
                stats["files_failed"] += 1
                stats["errors"].append(error_msg)

                # Update state with error
                original_file = Path(csv_file_path).stem + '.md'
                state.add_error(original_file, step=2, error=str(e))

        print(f"\n{'='*60}")
        print("Step 2 Complete")
        print(f"{'='*60}")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files succeeded: {stats['files_succeeded']}")
        print(f"Files failed: {stats['files_failed']}")
        print(f"Rows processed: {stats['rows_processed']}")
        print(f"QA pairs generated: {stats['qa_pairs_generated']}")
        print(f"API calls made: {self.stats['api_calls_made']}")
        print(f"API retries: {self.stats['api_retries']}")
        print(f"{'='*60}\n")

        # Update state with global stats
        stats.update(self.stats)

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
    input_folder = config.get('output', {}).get('folder', './output')
    output_folder = os.path.join(config.get('output', {}).get('folder', './output'), 'QA')

    # Create output subfolder for step2
    data_name = config.get('pipeline', {}).get('data_name', 'dataset')
    input_folder = os.path.join(input_folder, data_name)
    output_folder = os.path.join(output_folder, data_name)

    # Run step2
    step2 = EnhancedStep2(config)
    stats = step2.run(input_folder, output_folder, state, preview=config.get('pipeline', {}).get('preview', False))

    return stats


if __name__ == "__main__":
    main()
