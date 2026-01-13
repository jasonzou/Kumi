"""
MD2QA Enhanced Pipeline

A unified, configurable pipeline for converting Markdown files to QA datasets with:
- Resume capability
- Incremental processing
- Validation and error recovery
- Preview mode
- Rich metrics and reporting

Usage:
    python enhanced_pipeline.py
    python enhanced_pipeline.py --config my_config.yaml
    python enhanced_pipeline.py --preview
    python enhanced_pipeline.py --step 1,2
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from state.state_manager import PipelineState
from metrics import MetricsCollector, ProgressTracker
from enhanced_step1 import EnhancedStep1
from enhanced_step2 import EnhancedStep2
from enhanced_step3 import EnhancedStep3


class EnhancedPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline

        Args:
            config_path: Path to configuration file
            config_overrides: Dictionary of configuration overrides
        """
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(
            config_path=config_path,
            overrides=config_overrides
        )

        # Initialize components
        state_file = self.config.get('state', {}).get('file', './state/pipeline_state.json')
        self.state = PipelineState(state_file)

        self.metrics = MetricsCollector(self.config)
        self.progress = ProgressTracker(self.config)

        # Setup logging
        self._setup_logging()

        # Step executors
        self.step1 = EnhancedStep1(self.config)
        self.step2 = EnhancedStep2(self.config)
        self.step3 = EnhancedStep3(self.config)

    def _setup_logging(self):
        """Setup logging configuration"""
        logging_config = self.config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_file = logging_config.get('file', 'logs/pipeline.log')
        log_console = logging_config.get('console', True)
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create log directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout) if log_console else logging.NullHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced Pipeline initialized")
        self.logger.debug(f"Configuration sources: {self.config_manager.get_sources()}")

    def _get_data_name(self) -> str:
        """Get data name from config or use default"""
        return self.config.get('pipeline', {}).get('data_name', 'dataset')

    def _get_paths(self) -> Dict[str, str]:
        """Get input/output paths based on configuration"""
        input_folder = self.config.get('input', {}).get('folder', './input')
        output_folder = self.config.get('output', {}).get('folder', './output')

        data_name = self._get_data_name()

        paths = {
            'input': os.path.join(input_folder, data_name),
            'step1_output': os.path.join(output_folder, data_name),
            'step2_input': os.path.join(output_folder, data_name),
            'step2_output': os.path.join(output_folder, 'QA', data_name),
            'step3_input': os.path.join(output_folder, 'QA', data_name),
            'step3_output': os.path.join(output_folder, 'merged'),
            'merged_filename': f"{data_name}.csv"
        }

        return paths

    def _validate_environment(self) -> bool:
        """
        Validate environment before running

        Returns:
            True if valid, False otherwise
        """
        self.logger.info("Validating environment...")

        # Check input folder
        paths = self._get_paths()
        if not os.path.exists(paths['input']):
            self.logger.error(f"Input folder not found: {paths['input']}")
            return False

        # Check output folder permissions
        try:
            os.makedirs(paths['step1_output'], exist_ok=True)
            os.makedirs(paths['step2_output'], exist_ok=True)
            os.makedirs(paths['step3_output'], exist_ok=True)
        except Exception as e:
            self.logger.error(f"Cannot create output folders: {e}")
            return False

        # Check state folder
        state_file = self.config.get('state', {}).get('file', './state/pipeline_state.json')
        state_path = Path(state_file)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Cannot create state folder: {e}")
            return False

        self.logger.info("Environment validation passed")
        return True

    def _print_preview(self):
        """Print preview of what will be processed"""
        print("\n" + "=" * 60)
        print("PREVIEW MODE - No changes will be made")
        print("=" * 60)

        # Configuration
        print("\nConfiguration:")
        print(f"  Steps: {self.config.get('pipeline', {}).get('steps', [])}")
        print(f"  Resume: {self.config.get('pipeline', {}).get('resume', False)}")

        paths = self._get_paths()
        print(f"\nInput folder: {paths['input']}")

        # Count files
        input_path = Path(paths['input'])
        if input_path.exists():
            all_files = []
            for ext in self.config.get('step1', {}).get('extensions', ['.md', '.txt']):
                all_files.extend(input_path.rglob(f"*{ext}"))

            print(f"  Total files found: {len(all_files)}")

            # Check state for each file
            new_files = []
            unchanged_files = []
            modified_files = []

            for filepath in all_files:
                file_info = self.state.get_file_status(str(filepath))
                if file_info.get('status') == 'not_processed':
                    new_files.append(filepath.name)
                elif self.state.is_file_unchanged(str(filepath)):
                    unchanged_files.append(filepath.name)
                else:
                    modified_files.append(filepath.name)

            print(f"  New files: {len(new_files)}")
            print(f"  Unchanged (will skip): {len(unchanged_files)}")
            print(f"  Modified (will re-process): {len(modified_files)}")

        # Output paths
        print(f"\nOutput paths:")
        print(f"  Step 1: {paths['step1_output']}")
        print(f"  Step 2: {paths['step2_output']}")
        print(f"  Step 3: {paths['step3_output']}")
        print(f"  Final: {os.path.join(paths['step3_output'], paths['merged_filename'])}")

        # Estimate time
        print(f"\nEstimated time: This is a preview, actual time depends on file sizes and API speed")

        print("\n" + "=" * 60)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline

        Returns:
            Dictionary with execution results
        """
        self.logger.info("Starting Enhanced Pipeline")

        # Check preview mode
        preview = self.config.get('pipeline', {}).get('preview', False)
        if preview:
            self._print_preview()
            return {"preview": True}

        # Validate environment
        if not self._validate_environment():
            self.logger.error("Environment validation failed")
            return {"success": False, "error": "Environment validation failed"}

        # Get configuration
        steps = self.config.get('pipeline', {}).get('steps', [1, 2, 3])
        resume = self.config.get('pipeline', {}).get('resume', True)

        # Start pipeline in state
        self.state.start_pipeline(steps)

        # Track overall performance
        overall_perf = self.metrics.start_operation("overall")

        results = {
            "success": True,
            "steps": {},
            "errors": []
        }

        paths = self._get_paths()

        # Execute steps
        for step in steps:
            step_name = f"step{step}"
            step_perf = self.metrics.start_operation(step_name)

            self.logger.info(f"Starting {step_name}")

            try:
                if step == 1:
                    # Step 1: Markdown to CSV
                    stats = self.step1.run(
                        input_folder=paths['input'],
                        output_folder=paths['step1_output'],
                        state=self.state,
                        preview=False
                    )

                    self.state.complete_step(1)
                    self.metrics.finish_operation(step_name, stats.get('files_processed', 0))

                    results["steps"][1] = stats

                elif step == 2:
                    # Step 2: CSV to QA
                    stats = self.step2.run(
                        input_folder=paths['step2_input'],
                        output_folder=paths['step2_output'],
                        state=self.state,
                        preview=False
                    )

                    self.state.complete_step(2)
                    self.metrics.finish_operation(step_name, stats.get('rows_processed', 0))

                    results["steps"][2] = stats

                elif step == 3:
                    # Step 3: Merge
                    stats = self.step3.run(
                        input_folder=paths['step3_input'],
                        output_folder=paths['step3_output'],
                        output_filename=paths['merged_filename'],
                        state=self.state,
                        preview=False
                    )

                    self.state.complete_step(3)
                    self.metrics.finish_operation(step_name, stats.get('files_merged', 0))

                    results["steps"][3] = stats

                self.logger.info(f"Completed {step_name}")

            except Exception as e:
                self.logger.error(f"Error in {step_name}: {e}", exc_info=True)
                results["success"] = False
                results["errors"].append(f"{step_name}: {str(e)}")

                # Record error in metrics
                self.metrics.record_error(
                    error_type="step_execution",
                    step=step,
                    message=str(e),
                    recovered=False
                )

                # Don't continue on fatal error
                if not resume:
                    break

        # Complete overall operation
        overall_perf.finish()

        # Complete pipeline in state
        if results["success"]:
            self.state.complete_pipeline()
        else:
            self.state.fail_pipeline()

        # Generate reports
        self._generate_reports(results)

        self.logger.info("Pipeline execution completed")

        return results

    def _generate_reports(self, results: Dict[str, Any]):
        """Generate metrics and summary reports"""
        # Save metrics
        metrics_config = self.config.get('metrics', {})
        if metrics_config.get('enabled', True):
            # Save JSON metrics
            metrics_file = metrics_config.get('output', 'reports/metrics.json')
            self.metrics.save_to_file(metrics_file)
            self.logger.info(f"Metrics saved to {metrics_file}")

            # Save summary report
            summary_file = metrics_config.get('summary_report', 'reports/summary.txt')
            self.metrics.generate_summary_report(summary_file)
            self.logger.info(f"Summary report saved to {summary_file}")

            # Print to console
            self.metrics.print_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MD2QA Enhanced Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python enhanced_pipeline.py

  # Run with custom configuration
  python enhanced_pipeline.py --config my_config.yaml

  # Preview what would be processed
  python enhanced_pipeline.py --preview

  # Run only specific steps
  python enhanced_pipeline.py --steps 1,2

  # Override configuration values
  python enhanced_pipeline.py --step1.chunk-size 500 --step2.max-workers 10
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated list of steps to run (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview mode: show what would be processed without making changes'
    )

    parser.add_argument(
        '--data-name',
        type=str,
        help='Name for the dataset (used for folders and output files)'
    )

    parser.add_argument(
        '--input-folder',
        type=str,
        help='Input folder path'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        help='Output folder path'
    )

    # Parse known args for nested config overrides
    args, unknown_args = parser.parse_known_args()

    # Build config overrides from command-line
    config_overrides = {}

    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(',')]
        config_overrides['pipeline.steps'] = steps

    if args.preview:
        config_overrides['pipeline.preview'] = True

    if args.data_name:
        config_overrides['pipeline.data_name'] = args.data_name

    if args.input_folder:
        config_overrides['input.folder'] = args.input_folder

    if args.output_folder:
        config_overrides['output.folder'] = args.output_folder

    # Parse nested overrides (e.g., --step1.chunk-size 500)
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg[2:].replace('-', '.')
            # Next arg should be the value
            if unknown_args.index(arg) + 1 < len(unknown_args):
                value = unknown_args[unknown_args.index(arg) + 1]
                config_overrides[key] = value

    # Initialize and run pipeline
    try:
        pipeline = EnhancedPipeline(
            config_path=args.config,
            config_overrides=config_overrides
        )

        results = pipeline.run()

        # Exit with appropriate code
        if results.get('preview'):
            sys.exit(0)
        elif results.get('success'):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
