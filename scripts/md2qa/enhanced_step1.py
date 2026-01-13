"""
Enhanced Step 1: Markdown to CSV Conversion

This is an enhanced version of step1_md2csv.py with:
- Configuration integration
- State management and resume capability
- Validation and error handling
- Progress tracking
- Metrics collection

Can be used standalone or as part of the enhanced pipeline.
"""

import os
import sys
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from state.state_manager import PipelineState


class EnhancedStep1:
    """Enhanced Markdown to CSV converter with state management and multiple chunking strategies"""

    def __init__(self, config: dict):
        """
        Initialize enhanced step1

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.step1_config = config.get('step1', {})
        self.output_config = config.get('output', {})
        self.validation_config = config.get('validation', {})
        self.progress_config = config.get('progress', {})

        # Get chunking configuration
        self.chunking_strategy = self.step1_config.get('chunking_strategy', 'recursive')
        self.chunk_size = self.step1_config.get('chunk_size', 350)
        self.chunk_overlap = self.step1_config.get('chunk_overlap', 0)
        self.min_chunk_size = self.step1_config.get('min_chunk_size', 100)

        # Initialize text splitter based on strategy
        self._init_text_splitter()

    def _init_text_splitter(self):
        """Initialize text splitter based on configured strategy"""
        if self.chunking_strategy == 'recursive':
            # Recursive character splitting (original behavior)
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n", "\n", " ", ".", ",", "،", "،", "。", "、", "।",
                    "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'semantic':
            # Semantic splitting by headers and logical breaks
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n\n",  # Triple newline (major section break)
                    "\n\n",    # Double newline (paragraph break)
                    "\n",      # Single newline (line break)
                    "\r\n\r\n",  # Windows paragraph break
                    "\r\n",    # Windows line break
                    " ", ".", ",", "।",  # Character-level (as fallback)
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'markdown':
            # Markdown-aware splitting
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",     # Paragraph break
                    "\n",       # Line break
                    "\n#",      # Header 1
                    "\n##",     # Header 2
                    "\n###",    # Header 3
                    "\n####",   # Header 4
                    "\n#####",  # Header 5
                    "\n######",  # Header 6
                    "\n- ",     # Bullet list
                    "\n* ",     # Bullet list
                    "\n1. ",    # Numbered list
                    "```",      # Code block
                    " ", ".", ",", "।",  # Character-level (as fallback)
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'sentence':
            # Sentence-based splitting
            # Note: For sentence splitting, we'd ideally use nltk or spacy
            # But keeping it simple with punctuation-based splitting
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",     # Paragraph break
                    "。",        # Chinese period
                    ". ",       # English period with space
                    "! ",       # Exclamation
                    "? ",       # Question mark
                    "\n",       # Line break
                    "; ",       # Semicolon
                    ": ",       # Colon
                    ",",        # Comma (as fallback)
                    " "         # Space (as last resort)
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'large_context':
            # Optimized for large context windows (40k tokens)
            # Creates fewer, larger chunks with better coherence
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n\n",  # Major section break
                    "\n\n",    # Paragraph break
                    "\n",      # Line break
                    "\r\n\r\n", # Windows paragraph
                    "\r\n",    # Windows line
                    "\n#",     # Header 1
                    "\n##",    # Header 2
                    "\n###",   # Header 3
                    "\n- ",    # List item
                    "\n* ",    # Bullet
                    " ",       # Space (rarely used due to large chunk size)
                    ".",       # Period (very rare)
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'hierarchical':
            # Hierarchical splitting that preserves document structure
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n\n\n",  # Section break (h1)
                    "\n\n\n",    # Subsection break (h2)
                    "\n\n",      # Paragraph break
                    "\n",        # Line break
                    "\n####",   # H4
                    "\n###",     # H3
                    "\n##",      # H2
                    "\n#",       # H1
                    "\n- ",      # List item
                    "\n* ",      # Bullet
                    "\n1. ",     # Numbered list
                    "```",        # Code block
                    " ",         # Space (last resort)
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'overlap':
            # Overlap-based splitting to maintain context across chunks
            # Uses smaller chunk size with overlap to preserve context
            chunk_size = max(self.chunk_size // 2, 100)  # Use half size for better overlap
            chunk_overlap = max(self.chunk_overlap, chunk_size // 4)  # Ensure meaningful overlap

            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",     # Paragraph break
                    "\n",       # Line break
                    "\n#",      # Header
                    "\n##",     # Sub-header
                    "\n###",    # Sub-sub-header
                    "\n- ",     # Bullet
                    "\n* ",     # Bullet
                    " ",        # Word boundary
                ],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.chunking_strategy == 'content_aware':
            # Content-type aware splitting (text, code, lists, etc.)
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n\n",   # Section break
                    "\n\n",     # Paragraph
                    "\n```",    # Start of code block
                    "```\n",    # End of code block
                    "\n#",      # Header
                    "\n##",     # Sub-header
                    "\n- ",     # Bullet
                    "\n* ",     # Bullet
                    "\n1. ",    # Numbered list
                    "\n",       # Line
                    " ",        # Word
                ],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            # Default fallback
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ".", ",", "।"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

    def _get_chunking_info(self) -> dict:
        """
        Get information about current chunking strategy

        Returns:
            Dictionary with chunking details
        """
        return {
            "strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "description": self._get_strategy_description(self.chunking_strategy)
        }

    def _get_strategy_description(self, strategy: str) -> str:
        """
        Get description of chunking strategy

        Args:
            strategy: Strategy name

        Returns:
            Description string
        """
        descriptions = {
            'recursive': 'Standard recursive character splitting with multiple separator levels',
            'semantic': 'Semantic splitting by headers and logical content breaks',
            'markdown': 'Markdown-aware splitting that preserves document structure',
            'sentence': 'Sentence-based splitting optimized for natural language',
            'large_context': 'Optimized for large context windows (40k tokens) with better coherence',
            'hierarchical': 'Hierarchical splitting that preserves document hierarchy',
            'overlap': 'Overlap-based splitting to maintain context across chunks',
            'content_aware': 'Content-type aware splitting (text, code, lists, etc.)'
        }
        return descriptions.get(strategy, 'Unknown strategy')

    def validate_chunking_config(self) -> bool:
        """
        Validate chunking configuration

        Returns:
            True if valid, False otherwise
        """
        valid_strategies = [
            'recursive', 'semantic', 'markdown', 'sentence',
            'large_context', 'hierarchical', 'overlap', 'content_aware'
        ]

        if self.chunking_strategy not in valid_strategies:
            print(f"ERROR: Invalid chunking strategy '{self.chunking_strategy}'")
            print(f"Valid strategies: {', '.join(valid_strategies)}")
            return False

        if self.chunk_size <= 0:
            print(f"ERROR: chunk_size must be positive, got {self.chunk_size}")
            return False

        if self.chunk_overlap < 0:
            print(f"ERROR: chunk_overlap must be non-negative, got {self.chunk_overlap}")
            return False

        if self.min_chunk_size < 0:
            print(f"ERROR: min_chunk_size must be non-negative, got {self.min_chunk_size}")
            return False

        if self.chunk_overlap >= self.chunk_size:
            print(f"WARNING: chunk_overlap ({self.chunk_overlap}) >= chunk_size ({self.chunk_size})")
            print("This may cause infinite loops or poor chunking quality")

        return True

    def print_chunking_info(self):
        """Print information about current chunking configuration"""
        info = self._get_chunking_info()
        print("\n" + "="*60)
        print("CHUNKING CONFIGURATION")
        print("="*60)
        print(f"Strategy: {info['strategy']}")
        print(f"Description: {info['description']}")
        print(f"Chunk size: {info['chunk_size']} characters")
        print(f"Chunk overlap: {info['chunk_overlap']} characters")
        print(f"Min chunk size: {info['min_chunk_size']} characters")

        # Calculate effective chunk size
        effective_size = info['chunk_size'] - info['chunk_overlap']
        if effective_size > 0:
            print(f"Effective size: {effective_size} characters (after overlap)")
        print("="*60 + "\n")

    def merge_small_chunks(self, chunks, min_size):
        """Merge small chunks (same logic as original)"""
        if not chunks or min_size <= 0:
            return chunks

        has_small_chunks = True
        merged_chunks = [chunk.page_content for chunk in chunks]
        max_iterations = 100
        iteration_count = 0

        while has_small_chunks:
            iteration_count += 1

            if iteration_count > max_iterations:
                print(f"Warning: Merge iteration exceeded {max_iterations} times, stopping merge.")
                break

            has_small_chunks = False
            new_merged_chunks = []
            i = 0
            previous_chunks_count = len(merged_chunks)

            while i < len(merged_chunks):
                current_chunk = merged_chunks[i]
                current_length = len(current_chunk)

                if current_length >= min_size:
                    new_merged_chunks.append(current_chunk)
                    i += 1
                else:
                    if i < len(merged_chunks) - 1:
                        next_chunk = merged_chunks[i + 1]
                        merged_chunk = current_chunk + next_chunk
                        new_merged_chunks.append(merged_chunk)
                        has_small_chunks = True
                        i += 2
                    else:
                        if new_merged_chunks:
                            new_merged_chunks[-1] = new_merged_chunks[-1] + current_chunk
                        else:
                            new_merged_chunks.append(current_chunk)
                            print(f"Warning: File content length ({current_length}) is less than min chunk size ({min_size}), keeping as is.")
                        i += 1

            merged_chunks = new_merged_chunks

            if len(merged_chunks) == previous_chunks_count:
                small_chunks_exist = any(len(chunk) < min_size for chunk in merged_chunks)
                if small_chunks_exist:
                    small_count = sum(1 for chunk in merged_chunks if len(chunk) < min_size)
                    print(f"Warning: {small_count} chunks are still smaller than min chunk size ({min_size}), but cannot continue merging.")
                break

        return merged_chunks

    def process_file(self, filepath: str, output_folder: str, state: PipelineState) -> bool:
        """
        Process a single markdown file

        Args:
            filepath: Path to input file
            output_folder: Output folder path
            state: Pipeline state manager

        Returns:
            True if successful, False otherwise
        """
        filename = Path(filepath).name

        try:
            # Read file content
            with open(filepath, encoding='utf-8-sig') as f:
                file_content = f.read()

            # Preprocess
            file_content = re.sub(r'\*{3,}', '**', file_content)
            file_content = re.sub(r'\-{4,}', '---', file_content)
            file_content = re.sub(r'\={4,}', '===', file_content)

            # Handle images and links
            img_counter = 1
            text_counter = defaultdict(int)
            img_urls_per_line = []
            replacement_dict = {}

            def replace_img(match, img_counter, replacement_dict):
                key = f"<img{img_counter}>"
                replacement_dict[key] = match.group(0)
                return key, img_counter + 1, match.group(1)

            def replace_text(match, text_counter, replacement_dict):
                text = match.group(1)
                text_counter[text] += 1
                key = f"<{text}{text_counter[text] if text_counter[text] > 1 else ''}>"
                replacement_dict[key] = match.group(0)
                return key

            # Replace images
            img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
            img_matches = list(img_pattern.finditer(file_content))
            for match in img_matches:
                replacement, img_counter, img_url = replace_img(match, img_counter, replacement_dict)
                file_content = file_content.replace(match.group(0), replacement)
                img_urls_per_line.append(img_url)

            # Replace links
            text_pattern = re.compile(r'\[(.*?)\]\(.*?\)')
            text_matches = list(text_pattern.finditer(file_content))
            for match in text_matches:
                replacement = replace_text(match, text_counter, replacement_dict)
                file_content = file_content.replace(match.group(0), replacement)

            # Split text
            texts = self.text_splitter.create_documents([file_content])

            # Merge small chunks
            merged_texts = self.merge_small_chunks(
                texts,
                self.step1_config.get('min_chunk_size', 100)
            )

            # Create DataFrame
            df = pd.DataFrame(merged_texts, columns=['Text'])

            # Restore replacements
            def restore_replacement(text):
                for key, value in replacement_dict.items():
                    text = text.replace(key, value)
                return text

            df['Text_pure'] = df['Text']

            def extract_img_urls(text):
                img_keys = re.findall(r'<img\d+>', text)
                img_urls = []
                for key in img_keys:
                    if key in replacement_dict:
                        url_match = re.search(r'\((.*?)\)', replacement_dict[key])
                        if url_match:
                            img_urls.append(url_match.group(1))
                return ';'.join(img_urls)

            df['Img_url'] = df['Text_pure'].apply(extract_img_urls)
            df['Text'] = df['Text'].apply(restore_replacement)

            # Save to file (CSV or TSV based on config)
            output_format = self.step1_config.get('output_format', 'tsv')
            output_filename = f"{os.path.splitext(filename)[0]}.{output_format}"
            output_filepath = os.path.join(output_folder, output_filename)
            os.makedirs(output_folder, exist_ok=True)

            # Save with appropriate separator
            if output_format.lower() == 'tsv':
                df.to_csv(output_filepath, index=False, encoding='utf-8-sig', sep='\t')
            else:
                df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

            # Read and process position hierarchy
            if output_format.lower() == 'tsv':
                df = pd.read_csv(output_filepath, encoding='utf-8-sig', sep='\t')
            else:
                df = pd.read_csv(output_filepath, encoding='utf-8-sig')

            initial_position = os.path.splitext(filename)[0]
            position_stack = [initial_position]
            priority_stack = [-1]

            df['Position'] = ""

            last_position = ""
            position_counter = 0

            # Markdown priority
            markdown_priority = {
                r'^# ': 0, r'^## ': 1, r'^### ': 2, r'^#### ': 3,
                r'^##### ': 4, r'^###### ': 5, r'^\d+\. ': 6,
                r'^- ': 6, r'^\* ': 6, r'^\*\*.*\*\*$': 6
            }

            special_syntax = {
                r'^\s*=+\s*$': 0,
                r'^\s*-+\s*$': 1
            }

            for idx, row in df.iterrows():
                text = row['Text']
                sentences = re.split(r'\n+', text)

                current_position = ' > '.join(position_stack)

                if current_position == last_position:
                    position_counter += 1
                    if position_counter == 2:
                        df.at[idx - 1, 'Position'] += f" (part 1)"
                    current_position += f" (part {position_counter})"
                else:
                    last_position = current_position
                    position_counter = 1

                df.at[idx, 'Position'] = current_position

                previous_sentence = ""
                for sentence in sentences:
                    current_priority = None

                    for grammar, priority in markdown_priority.items():
                        if re.match(grammar, sentence.strip()):
                            current_priority = priority
                            break

                    for special_grammar, special_priority in special_syntax.items():
                        if re.match(special_grammar, sentence.strip()):
                            current_priority = special_priority
                            sentence = previous_sentence
                            break

                    if current_priority is not None:
                        while priority_stack and priority_stack[-1] >= current_priority:
                            position_stack.pop()
                            priority_stack.pop()

                        position_stack.append(sentence.strip())
                        priority_stack.append(current_priority)

                    previous_sentence = sentence

            # Save final CSV
            df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

            # Calculate checksum and update state
            checksum = state.calculate_checksum(filepath)
            state.update_file(
                filepath=filepath,
                checksum=checksum,
                status="complete",
                step_completed=1,
                output_files=[output_filename]
            )

            return True

        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"ERROR: {error_msg}")

            # Update state with error
            state.add_error(filepath, step=1, error=str(e))
            state.update_file(
                filepath=filepath,
                checksum=state.calculate_checksum(filepath) if os.path.exists(filepath) else "",
                status="failed",
                error=str(e)
            )

            return False

    def validate_output(self, filepath: str) -> bool:
        """
        Validate output CSV file

        Args:
            filepath: Path to output CSV file

        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')

            required_columns = ['Text', 'Text_pure', 'Img_url', 'Position']

            for col in required_columns:
                if col not in df.columns:
                    print(f"ERROR: Missing required column '{col}' in {filepath}")
                    return False

            if len(df) == 0:
                print(f"ERROR: Empty CSV file {filepath}")
                return False

            return True

        except Exception as e:
            print(f"ERROR: Validation failed for {filepath}: {e}")
            return False

    def run(
        self,
        input_folder: str,
        output_folder: str,
        state: PipelineState,
        preview: bool = False
    ) -> dict:
        """
        Run step 1 processing

        Args:
            input_folder: Input folder path
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
            "chunks_created": 0,
            "errors": []
        }

        # Get all files to process
        input_path = Path(input_folder)
        files = []

        for ext in self.step1_config.get('extensions', ['.md', '.txt']):
            files.extend(input_path.rglob(f"*{ext}"))

        files = [str(f) for f in files]

        if not files:
            print(f"WARNING: No files found in {input_folder}")
            return stats

        # Filter files based on state
        files_to_process = state.get_files_to_process(files, step=1)

        if not files_to_process:
            print("All files already processed for Step 1")
            return stats

        print(f"\n{'='*60}")
        print("Step 1: Markdown to CSV Conversion")
        print(f"{'='*60}")
        print(f"Total files found: {len(files)}")
        print(f"Files to process: {len(files_to_process)}")
        print(f"Files skipped (unchanged): {len(files) - len(files_to_process)}")
        print(f"{'='*60}\n")

        if preview:
            print("PREVIEW MODE: No files will be processed")
            for filename in files_to_process:
                print(f"  Would process: {Path(filename).name}")
            return stats

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Process files
        show_progress = self.progress_config.get('enabled', True)
        unit = self.progress_config.get('unit', 'file')

        for filepath in tqdm(
            files_to_process,
            desc="Processing files",
            unit=unit,
            disable=not show_progress,
            position=self.progress_config.get('position', 0)
        ):
            filename = Path(filepath).name
            stats["files_processed"] += 1

            print(f"\nProcessing: {filename}")

            success = self.process_file(filepath, output_folder, state)

            if success:
                # Validate output
                output_filename = f"{Path(filepath).stem}.csv"
                output_filepath = os.path.join(output_folder, output_filename)

                if self.validation_config.get('check_outputs', True):
                    if self.validate_output(output_filepath):
                        stats["files_succeeded"] += 1
                        print(f"✓ {filename} completed successfully")
                    else:
                        stats["files_failed"] += 1
                        stats["errors"].append(f"Validation failed for {filename}")
                        print(f"✗ {filename} validation failed")
                else:
                    stats["files_succeeded"] += 1
            else:
                stats["files_failed"] += 1
                stats["errors"].append(f"Processing failed for {filename}")

        # Count chunks created
        for csv_file in Path(output_folder).glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                stats["chunks_created"] += len(df)
            except:
                pass

        print(f"\n{'='*60}")
        print("Step 1 Complete")
        print(f"{'='*60}")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files succeeded: {stats['files_succeeded']}")
        print(f"Files failed: {stats['files_failed']}")
        print(f"Total chunks created: {stats['chunks_created']}")
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
    input_folder = config.get('input', {}).get('folder', './input')
    output_folder = config.get('output', {}).get('folder', './output')

    # Create output subfolder for step1
    output_folder = os.path.join(output_folder, config.get('pipeline', {}).get('data_name', 'dataset'))

    # Run step1
    step1 = EnhancedStep1(config)
    stats = step1.run(input_folder, output_folder, state, preview=config.get('pipeline', {}).get('preview', False))

    return stats


if __name__ == "__main__":
    main()
