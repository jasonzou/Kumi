# MD2QA Enhanced Pipeline

A unified, configurable pipeline for converting Markdown files to QA datasets with resume capability, incremental processing, validation, and rich reporting.

## Features

- **Unified Pipeline**: Single command to run all steps
- **Resume Capability**: Continue from where you left off after interruption
- **Incremental Processing**: Only process new or modified files
- **Preview Mode**: See what would be processed without making changes
- **Rich Configuration**: YAML-based configuration with environment variable support
- **Progress Tracking**: Real-time progress bars and ETA estimates
- **Metrics & Reporting**: Performance, quality, and resource usage metrics
- **Error Recovery**: Automatic retry with exponential backoff
- **Validation**: Built-in validation for all outputs

## Quick Start

### 1. Basic Usage

```bash
# Run with default configuration
python enhanced_pipeline.py

# Run with preview mode (see what would be processed)
python enhanced_pipeline.py --preview

# Use a custom configuration file
python enhanced_pipeline.py --config config/my_config.yaml
```

### 2. Folder Structure

```
project/
├── input/
│   └── my_dataset/
│       ├── file1.md
│       ├── file2.md
│       └── ...
├── output/
│   ├── my_dataset/          # Step 1 output (CSV files)
│   ├── QA/
│   │   └── my_dataset/      # Step 2 output (QA pairs)
│   └── merged/              # Step 3 output (final dataset)
├── state/
│   └── pipeline_state.json # Resume state
├── logs/
│   ├── pipeline.log
│   └── errors.log
└── reports/
    ├── metrics.json
    └── summary.txt
```

### 3. Configuration

Create a `config.yaml` file:

```yaml
pipeline:
  data_name: "my_dataset"
  steps: [1, 2, 3]
  resume: true

input:
  folder: "./input"

output:
  folder: "./output"

step2:
  rounds: 3
  max_workers: 5
```

Or override on command line:

```bash
python enhanced_pipeline.py \
  --step1.chunk-size 500 \
  --step2.rounds 5 \
  --step2.max-workers 10
```

## Step-by-Step Guide

### Step 1: Markdown to CSV

Converts Markdown files into structured CSV chunks.

**Input**: `.md`, `.txt` files
**Output**: CSV with columns: `Text`, `Text_pure`, `Img_url`, `Position`

**Key Settings**:
- `chunk_size`: Size of text chunks (characters)
- `min_chunk_size`: Minimum chunk size (smaller chunks are merged)
- `handle_images`: Extract image URLs

### Step 2: CSV to QA Generation

Generates question-answer pairs from text chunks using LLM.

**Input**: CSV files from Step 1
**Output**: CSV with additional columns: `Question`, `Answer`

**Key Settings**:
- `rounds`: Number of generation attempts (higher = better quality)
- `max_workers`: Concurrent API calls
- `api_retries`: Retry attempts for failed API calls

### Step 3: Merge

Merges all QA CSV files into a single dataset.

**Input**: QA CSV files from Step 2
**Output**: Single merged CSV file

**Key Settings**:
- `merge_strategy`: `concat` (all columns) or `union` (common columns)
- `track_sources`: Track source file for each row

## Advanced Features

### Resume Capability

The pipeline automatically saves state after each file. If interrupted:

```bash
# Resume from last checkpoint
python enhanced_pipeline.py
```

State is saved in `state/pipeline_state.json` and includes:
- Processed files and their checksums
- Step completion status
- Error logs
- Checkpoints for resumption

### Incremental Processing

Only new or modified files are processed:

```bash
# Pipeline automatically detects:
# - New files: Process them
# - Modified files: Re-process them
# - Unchanged files: Skip them
```

### Preview Mode

See what would be processed without making changes:

```bash
python enhanced_pipeline.py --preview
```

Shows:
- Files that would be processed
- Estimated operations
- Configuration being used

### Metrics and Reporting

Detailed metrics are collected and saved:

**JSON Metrics** (`reports/metrics.json`):
```json
{
  "performance": {
    "step1": {
      "duration_seconds": 120.5,
      "items_processed": 45,
      "items_per_second": 0.37
    }
  },
  "quality": {
    "step2": {
      "total_items": 1250,
      "valid_items": 1230,
      "validation_pass_rate": 98.4
    }
  },
  "resources": {
    "peak_memory_mb": 512.3,
    "api_calls_made": 3750,
    "api_retries": 16
  }
}
```

**Summary Report** (`reports/summary.txt`):
```
MD2QA PIPELINE EXECUTION SUMMARY
============================================================

PERFORMANCE METRICS
------------------------------------------------------------

step1:
  Duration: 120.50s
  Items: 45
  Throughput: 0.37 items/sec

QUALITY METRICS
------------------------------------------------------------

step2:
  Total items: 1250
  Valid: 1230
  Invalid: 20
  Pass rate: 98.4%

...
```

### Error Handling

Automatic retry with exponential backoff:

```yaml
validation:
  retry_failed: true
  max_retries: 3
  retry_delay: 5
  retry_backoff: 2
```

Errors are categorized:
- **Fatal**: Config errors, permissions → Stop pipeline
- **Recoverable**: API errors, I/O issues → Retry with backoff
- **Skip-able**: Bad input files → Log and continue
- **Partial**: Some rows fail → Save good ones, mark incomplete

## Configuration Reference

### Pipeline Settings

```yaml
pipeline:
  data_name: "dataset_name"      # Dataset name
  steps: [1, 2, 3]               # Steps to run
  resume: true                    # Enable resume
  preview: false                  # Preview mode
```

### Input/Output

```yaml
input:
  folder: "./input"              # Input folder
  pattern: "**/*"                # File pattern
  exclude: ["*.tmp"]             # Exclude patterns
  extensions: [".md", ".txt"]    # File extensions

output:
  folder: "./output"             # Output folder
  create_subfolders: true        # Create subfolders per step
  backup_existing: false         # Backup existing files
```

### Step 1 Settings

```yaml
step1:
  chunk_size: 350                # Text chunk size
  min_chunk_size: 100            # Minimum chunk size
  output_format: "csv"           # csv, json, parquet
  handle_images: true            # Extract image URLs
```

### Step 2 Settings

```yaml
step2:
  rounds: 3                      # Generation rounds
  max_workers: 5                 # Concurrent API calls
  model: "model_name"            # LLM model
  temperature: 0.7               # Generation temperature
  api_retries: 3                 # API retry attempts
  api_retry_delay: 5             # Retry delay (seconds)
```

### Step 3 Settings

```yaml
step3:
  merge_strategy: "concat"        # concat or union
  track_sources: true            # Track source files
  source_column: "source_file"   # Source file column name
```

### Validation

```yaml
validation:
  check_outputs: true            # Validate outputs
  retry_failed: true             # Retry failed ops
  max_retries: 3                 # Max retry attempts
  retry_delay: 5                 # Retry delay
  retry_backoff: 2               # Backoff multiplier
  strictness: "normal"           # strict, normal, relaxed
```

### Logging

```yaml
logging:
  level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
  file: "logs/pipeline.log"      # Log file
  console: true                  # Log to console
  format: "%(asime)s - ..."      # Log format
```

### Metrics

```yaml
metrics:
  enabled: true                  # Enable metrics
  output: "reports/metrics.json" # JSON output
  summary_report: "reports/summary.txt"  # Summary report
  performance: true              # Collect performance
  quality: true                  # Collect quality
  resources: true                # Collect resources
```

## Command-Line Options

```bash
python enhanced_pipeline.py [options]

Options:
  --config PATH              Configuration file path
  --steps 1,2,3             Steps to run
  --preview                 Preview mode
  --data-name NAME          Dataset name
  --input-folder PATH        Input folder
  --output-folder PATH       Output folder
  --step1.chunk-size N      Step 1 chunk size
  --step2.rounds N          Step 2 rounds
  --step2.max-workers N     Step 2 workers
```

## Environment Variables

Configuration can also be set via environment variables:

```bash
export MD2QA_PIPELINE_DATA_NAME="my_dataset"
export MD2QA_STEP1_CHUNK_SIZE=500
export MD2QA_STEP2_MAX_WORKERS=10
export MD2QA_LOGGING_LEVEL="DEBUG"
```

## Troubleshooting

### Out of Memory

Reduce memory usage:

```yaml
resources:
  max_memory: 1024           # Reduce from default 2048
  max_cores: 2               # Limit CPU cores
```

### API Rate Limits

Increase retry settings:

```yaml
step2:
  api_retries: 5             # More retries
  api_retry_delay: 10         # Longer delay
  api_retry_backoff: 3        # Slower backoff
```

### Slow Processing

Optimize for speed:

```yaml
step1:
  chunk_size: 500            # Larger chunks = fewer chunks

step2:
  max_workers: 10            # More concurrent calls
  rounds: 2                  # Fewer generation rounds
```

### Check State

View pipeline state:

```bash
cat state/pipeline_state.json | jq .
```

### Reset State

Reset to start fresh:

```python
from state.state_manager import PipelineState

state = PipelineState("./state/pipeline_state.json")
state.reset()
```

## Migration from Old Scripts

The enhanced pipeline replaces:
- `step1_md2csv.py`
- `step2_chunk2qa.py`
- `step3_merge.py`
- `pipeline.py`

**Old way**:
```bash
python step1_md2csv.py
python step2_chunk2qa.py
python step3_merge.py
```

**New way**:
```bash
python enhanced_pipeline.py
```

## API Reference

### Python API

```python
from enhanced_pipeline import EnhancedPipeline
from config.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("my_config.yaml")

# Create pipeline
pipeline = EnhancedPipeline(config_overrides={
    "pipeline.steps": [1, 2],
    "step1.chunk_size": 500
})

# Run
results = pipeline.run()
```

### Standalone Steps

```python
from enhanced_step1 import EnhancedStep1
from enhanced_step2 import EnhancedStep2
from enhanced_step3 import EnhancedStep3

# Run individual steps
step1 = EnhancedStep1(config)
stats = step1.run(input_folder, output_folder, state)

step2 = EnhancedStep2(config)
stats = step2.run(input_folder, output_folder, state)

step3 = EnhancedStep3(config)
stats = step3.run(input_folder, output_folder, output_filename, state)
```

## Contributing

Contributions are welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

This project is licensed under the MIT License.
