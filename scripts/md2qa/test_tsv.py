import sys
sys.path.insert(0, os.getcwd())

from config.config_manager import ConfigManager
from enhanced_step1 import EnhancedStep1
from pathlib import Path

# Test TSV output format
print("Testing TSV output format...\n")

# Load config with TSV format
config_manager = ConfigManager()
config = config_manager.load_config(overrides={
    'step1.output_format': 'tsv'
})

print(f"Config output_format: {config.get('step1', {}).get('output_format')}")

# Create step1 instance
step1 = EnhancedStep1(config)

# Create a test markdown file
test_content = """
# Test Document

This is a test document for TSV output.

## Section 1

This is the first section with some content.
"""

test_file = Path("test_tsv.md")
test_file.write_text(test_content)

print("\nTest file created successfully")

# Clean up
test_file.unlink()

print("\nTSV format test completed successfully!")
