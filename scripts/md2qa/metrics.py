"""
Metrics Collection and Reporting for MD2QA Enhanced Pipeline

Collects and reports:
- Performance metrics (timing, throughput)
- Quality metrics (validation results, QA pair quality)
- Resource metrics (memory, CPU, API calls)
- Error analytics
"""

import time
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Performance metrics for a step or operation"""
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    items_processed: int = 0
    items_per_second: Optional[float] = None

    def finish(self, items_processed: int = 0):
        """Mark operation as complete and calculate metrics"""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.items_processed = items_processed or self.items_processed

        if self.duration_seconds > 0:
            self.items_per_second = self.items_processed / self.duration_seconds


@dataclass
class QualityMetrics:
    """Quality metrics for outputs"""
    total_items: int = 0
    valid_items: int = 0
    invalid_items: int = 0
    validation_pass_rate: float = 0.0
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

    def calculate_pass_rate(self):
        """Calculate validation pass rate"""
        if self.total_items > 0:
            self.validation_pass_rate = (self.valid_items / self.total_items) * 100
        else:
            self.validation_pass_rate = 0.0


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_calls: int = 0
    api_calls_made: int = 0
    api_retries: int = 0


@dataclass
class ErrorMetrics:
    """Error tracking and analytics"""
    total_errors: int = 0
    recovered_errors: int = 0
    fatal_errors: int = 0
    errors_by_type: Dict[str, int] = None
    errors_by_step: Dict[int, int] = None
    error_details: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors_by_type is None:
            self.errors_by_type = {}
        if self.errors_by_step is None:
            self.errors_by_step = {}
        if self.error_details is None:
            self.error_details = []


class MetricsCollector:
    """Collects and aggregates metrics during pipeline execution"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics collector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get('metrics', {})
        self.resources_config = config.get('resources', {})

        # Initialize metrics
        self.performance = {}
        self.quality = {}
        self.resources = ResourceMetrics()
        self.errors = ErrorMetrics()

        # Track process for resource monitoring
        self.process = psutil.Process()

        # Memory tracking
        self.memory_samples = []

    def start_operation(self, name: str) -> PerformanceMetrics:
        """
        Start tracking an operation

        Args:
            name: Operation name

        Returns:
            PerformanceMetrics object
        """
        perf = PerformanceMetrics(start_time=time.time())
        self.performance[name] = perf
        return perf

    def finish_operation(self, name: str, items_processed: int = 0):
        """
        Finish tracking an operation

        Args:
            name: Operation name
            items_processed: Number of items processed
        """
        if name in self.performance:
            self.performance[name].finish(items_processed)

    def record_validation_result(self, step_name: str, total: int, valid: int, invalid: int, errors: List[str] = None):
        """
        Record validation results

        Args:
            step_name: Name of step
            total: Total items
            valid: Valid items
            invalid: Invalid items
            errors: List of error messages
        """
        if step_name not in self.quality:
            self.quality[step_name] = QualityMetrics()

        quality = self.quality[step_name]
        quality.total_items = total
        quality.valid_items = valid
        quality.invalid_items = invalid

        if errors:
            quality.validation_errors.extend(errors)

        quality.calculate_pass_rate()

    def record_error(self, error_type: str, step: int, message: str, recovered: bool = False):
        """
        Record an error

        Args:
            error_type: Type/category of error
            step: Step number where error occurred
            message: Error message
            recovered: Whether error was recovered
        """
        self.errors.total_errors += 1

        if recovered:
            self.errors.recovered_errors += 1
        else:
            self.errors.fatal_errors += 1

        # Track by type
        self.errors.errors_by_type[error_type] = self.errors.errors_by_type.get(error_type, 0) + 1

        # Track by step
        self.errors.errors_by_step[step] = self.errors.errors_by_step.get(step, 0) + 1

        # Track detail
        self.errors.error_details.append({
            "type": error_type,
            "step": step,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "recovered": recovered
        })

    def update_resource_usage(self):
        """Update current resource usage"""
        try:
            # Memory
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_samples.append(memory_mb)

            # Peak memory
            if memory_mb > self.resources.peak_memory_mb:
                self.resources.peak_memory_mb = memory_mb

            # CPU
            cpu_percent = self.process.cpu_percent()
            if cpu_percent > self.resources.peak_cpu_percent:
                self.resources.peak_cpu_percent = cpu_percent

        except Exception:
            pass  # Ignore errors in resource tracking

    def calculate_final_metrics(self):
        """Calculate final aggregated metrics"""
        # Average memory
        if self.memory_samples:
            self.resources.avg_memory_mb = sum(self.memory_samples) / len(self.memory_samples)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary

        Returns:
            Dictionary representation of all metrics
        """
        # Calculate final metrics
        self.calculate_final_metrics()

        # Build output
        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "performance": {},
            "quality": {},
            "resources": asdict(self.resources),
            "errors": asdict(self.errors)
        }

        # Add performance metrics
        for name, perf in self.performance.items():
            output["performance"][name] = asdict(perf)

        # Add quality metrics
        for name, quality in self.quality.items():
            output["quality"][name] = asdict(quality)

        return output

    def save_to_file(self, filepath: str):
        """
        Save metrics to JSON file

        Args:
            filepath: Path to output file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print metrics summary to console"""
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)

        # Performance
        if self.performance:
            print("\nPerformance:")
            for name, perf in self.performance.items():
                duration = perf.duration_seconds or 0
                items = perf.items_processed or 0
                rate = perf.items_per_second or 0

                print(f"  {name}:")
                print(f"    Duration: {duration:.2f}s")
                print(f"    Items processed: {items}")
                print(f"    Throughput: {rate:.2f} items/sec")

        # Quality
        if self.quality:
            print("\nQuality:")
            for name, quality in self.quality.items():
                print(f"  {name}:")
                print(f"    Total items: {quality.total_items}")
                print(f"    Valid: {quality.valid_items}")
                print(f"    Invalid: {quality.invalid_items}")
                print(f"    Pass rate: {quality.validation_pass_rate:.1f}%")

        # Resources
        print("\nResource Usage:")
        print(f"  Peak memory: {self.resources.peak_memory_mb:.2f} MB")
        print(f"  Avg memory: {self.resources.avg_memory_mb:.2f} MB")
        print(f"  Peak CPU: {self.resources.peak_cpu_percent:.1f}%")
        print(f"  API calls: {self.resources.api_calls_made}")
        print(f"  API retries: {self.resources.api_retries}")

        # Errors
        print("\nErrors:")
        print(f"  Total: {self.errors.total_errors}")
        print(f"  Recovered: {self.errors.recovered_errors}")
        print(f"  Fatal: {self.errors.fatal_errors}")

        if self.errors.errors_by_type:
            print("  By type:")
            for error_type, count in self.errors.errors_by_type.items():
                print(f"    {error_type}: {count}")

        if self.errors.errors_by_step:
            print("  By step:")
            for step, count in self.errors.errors_by_step.items():
                print(f"    Step {step}: {count}")

        print("=" * 60 + "\n")

    def generate_summary_report(self, filepath: str):
        """
        Generate human-readable summary report

        Args:
            filepath: Path to output file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("MD2QA PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")

            # Performance section
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 60 + "\n")
            for name, perf in self.performance.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Duration: {perf.duration_seconds:.2f}s\n")
                f.write(f"  Items: {perf.items_processed}\n")
                if perf.items_per_second:
                    f.write(f"  Throughput: {perf.items_per_second:.2f} items/sec\n")

            # Quality section
            f.write("\n\nQUALITY METRICS\n")
            f.write("-" * 60 + "\n")
            for name, quality in self.quality.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Total items: {quality.total_items}\n")
                f.write(f"  Valid: {quality.valid_items}\n")
                f.write(f"  Invalid: {quality.invalid_items}\n")
                f.write(f"  Pass rate: {quality.validation_pass_rate:.1f}%\n")

                if quality.validation_errors:
                    f.write(f"  Errors: {len(quality.validation_errors)}\n")

            # Resource section
            f.write("\n\nRESOURCE USAGE\n")
            f.write("-" * 60 + "\n")
            f.write(f"Peak memory: {self.resources.peak_memory_mb:.2f} MB\n")
            f.write(f"Average memory: {self.resources.avg_memory_mb:.2f} MB\n")
            f.write(f"Peak CPU: {self.resources.peak_cpu_percent:.1f}%\n")
            f.write(f"API calls: {self.resources.api_calls_made}\n")
            f.write(f"API retries: {self.resources.api_retries}\n")

            # Error section
            f.write("\n\nERROR SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total errors: {self.errors.total_errors}\n")
            f.write(f"Recovered: {self.errors.recovered_errors}\n")
            f.write(f"Fatal: {self.errors.fatal_errors}\n")

            if self.errors.errors_by_type:
                f.write("\nBy type:\n")
                for error_type, count in self.errors.errors_by_type.items():
                    f.write(f"  {error_type}: {count}\n")

            if self.errors.errors_by_step:
                f.write("\nBy step:\n")
                for step, count in self.errors.errors_by_step.items():
                    f.write(f"  Step {step}: {count}\n")

            # Error details
            if self.errors.error_details:
                f.write("\n\nERROR DETAILS\n")
                f.write("-" * 60 + "\n")
                for error in self.errors.error_details[-10:]:  # Last 10 errors
                    f.write(f"\n[{error['timestamp']}] Step {error['step']}: {error['type']}\n")
                    f.write(f"  Message: {error['message']}\n")
                    f.write(f"  Recovered: {error['recovered']}\n")

            f.write("\n" + "=" * 60 + "\n")


class ProgressTracker:
    """Tracks and displays progress across pipeline steps"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize progress tracker

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.progress_config = config.get('progress', {})

        # Trackers
        self.trackers = {}

    def create_tracker(self, name: str, total: int, unit: str = "item", description: str = ""):
        """
        Create a progress tracker

        Args:
            name: Tracker name
            total: Total number of items
            unit: Unit name (item, file, etc.)
            description: Description to display
        """
        if self.progress_config.get('enabled', True):
            self.trackers[name] = tqdm(
                total=total,
                unit=unit,
                desc=description or name,
                position=self.progress_config.get('position', 0),
                leave=True
            )
        else:
            # Create dummy tracker
            class DummyTracker:
                def update(self, n=1): pass
                def close(self): pass
                def set_postfix(self, **kwargs): pass

            self.trackers[name] = DummyTracker()

    def update(self, name: str, n: int = 1, **kwargs):
        """
        Update progress tracker

        Args:
            name: Tracker name
            n: Number of items to advance
            **kwargs: Additional info for postfix
        """
        if name in self.trackers:
            self.trackers[name].update(n)
            if kwargs:
                self.trackers[name].set_postfix(**kwargs)

    def close_tracker(self, name: str):
        """Close a progress tracker"""
        if name in self.trackers:
            self.trackers[name].close()

    def close_all(self):
        """Close all trackers"""
        for tracker in self.trackers.values():
            tracker.close()
