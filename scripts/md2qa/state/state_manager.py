"""
Pipeline State Manager for MD2QA Enhanced Pipeline

Manages pipeline execution state to enable:
- Resume capability after interruption
- Incremental processing (skip unchanged files)
- Progress tracking
- Error recovery

State file structure:
{
  "version": "1.0",
  "pipeline": {
    "last_run": "2024-01-13T10:30:00Z",
    "completed_steps": [1],
    "current_step": 2,
    "status": "running|completed|failed|interrupted"
  },
  "files": {
    "input": {
      "file1.md": {
        "checksum": "abc123",
        "processed_at": "2024-01-13T10:25:00Z",
        "status": "complete|incomplete|failed",
        "error": null,
        "step1_completed": true,
        "step2_completed": true,
        "step3_completed": true,
        "output_files": ["file1.csv"]
      }
    }
  },
  "checkpoints": {
    "step2": {
      "last_file_index": 45,
      "last_processed_row": "2024-01-13T10:29:00Z",
      "completed_files": ["file1.md", "file2.md"]
    }
  },
  "errors": [
    {
      "file": "file3.md",
      "step": 2,
      "error": "API rate limit",
      "timestamp": "2024-01-13T10:30:00Z",
      "retries": 3
    }
  ]
}
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class PipelineState:
    """Manages pipeline execution state for resume and incremental processing"""

    def __init__(self, state_file: str):
        """
        Initialize state manager

        Args:
            state_file: Path to state file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self._state = {
            "version": "1.0",
            "pipeline": {
                "last_run": None,
                "completed_steps": [],
                "current_step": None,
                "status": "initialized"
            },
            "files": {
                "input": {}
            },
            "checkpoints": {},
            "errors": []
        }

        self._load()

    def _load(self):
        """Load state from file if it exists"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load state file: {e}")
                print("Starting with fresh state")

    def _save(self, backup: bool = True):
        """Save state to file"""
        # Create backup if file exists
        if backup and self.state_file.exists():
            backup_file = self.state_file.with_suffix('.json.backup')
            shutil.copy2(self.state_file, backup_file)

        # Atomic write
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Move to final location
        temp_file.replace(self.state_file)

    def start_pipeline(self, steps: List[int]):
        """Mark pipeline as started"""
        self._state["pipeline"]["last_run"] = datetime.utcnow().isoformat() + "Z"
        self._state["pipeline"]["completed_steps"] = []
        self._state["pipeline"]["current_step"] = steps[0] if steps else None
        self._state["pipeline"]["status"] = "running"
        self._save()

    def complete_step(self, step: int):
        """Mark a step as completed"""
        if step not in self._state["pipeline"]["completed_steps"]:
            self._state["pipeline"]["completed_steps"].append(step)
        self._save()

    def set_current_step(self, step: int):
        """Set current active step"""
        self._state["pipeline"]["current_step"] = step
        self._save()

    def complete_pipeline(self):
        """Mark pipeline as completed"""
        self._state["pipeline"]["status"] = "completed"
        self._save()

    def interrupt_pipeline(self):
        """Mark pipeline as interrupted"""
        self._state["pipeline"]["status"] = "interrupted"
        self._save()

    def fail_pipeline(self):
        """Mark pipeline as failed"""
        self._state["pipeline"]["status"] = "failed"
        self._save()

    def get_file_status(self, filepath: str) -> Dict[str, Any]:
        """Get status of a file"""
        filepath = Path(filepath).name
        return self._state["files"]["input"].get(filepath, {
            "status": "not_processed",
            "step1_completed": False,
            "step2_completed": False,
            "step3_completed": False
        })

    def update_file(
        self,
        filepath: str,
        checksum: str,
        status: str,
        step_completed: Optional[int] = None,
        output_files: Optional[List[str]] = None,
        error: Optional[str] = None
    ):
        """Update file status"""
        filepath = Path(filepath).name

        if filepath not in self._state["files"]["input"]:
            self._state["files"]["input"][filepath] = {
                "checksum": None,
                "processed_at": None,
                "status": "not_processed",
                "error": None,
                "step1_completed": False,
                "step2_completed": False,
                "step3_completed": False,
                "output_files": []
            }

        file_info = self._state["files"]["input"][filepath]
        file_info["checksum"] = checksum
        file_info["processed_at"] = datetime.utcnow().isoformat() + "Z"
        file_info["status"] = status

        if step_completed:
            if step_completed == 1:
                file_info["step1_completed"] = True
            elif step_completed == 2:
                file_info["step2_completed"] = True
            elif step_completed == 3:
                file_info["step3_completed"] = True

        if output_files:
            file_info["output_files"] = output_files

        if error:
            file_info["error"] = error

        self._save()

    def calculate_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def is_file_unchanged(self, filepath: str) -> bool:
        """Check if file has unchanged checksum"""
        filepath = Path(filepath).name
        if filepath not in self._state["files"]["input"]:
            return False

        current_checksum = self.calculate_checksum(filepath)
        stored_checksum = self._state["files"]["input"][filepath].get("checksum")

        return current_checksum == stored_checksum

    def get_files_to_process(self, filepaths: List[str], step: int) -> List[str]:
        """
        Get list of files that need processing for a given step

        Args:
            filepaths: List of all file paths
            step: Step number (1, 2, or 3)

        Returns:
            List of files that need processing
        """
        files_to_process = []

        for filepath in filepaths:
            filepath = Path(filepath).name
            file_info = self._state["files"]["input"].get(filepath)

            if not file_info:
                # File not in state, needs processing
                files_to_process.append(filepath)
            elif file_info.get("status") == "failed":
                # Failed file, retry
                files_to_process.append(filepath)
            elif file_info.get("status") == "incomplete":
                # Incomplete file, resume
                files_to_process.append(filepath)
            else:
                # Check if specific step needs processing
                if step == 1 and not file_info.get("step1_completed"):
                    files_to_process.append(filepath)
                elif step == 2 and not file_info.get("step2_completed"):
                    files_to_process.append(filepath)
                elif step == 3 and not file_info.get("step3_completed"):
                    files_to_process.append(filepath)

        return files_to_process

    def set_checkpoint(self, step: int, checkpoint_data: Dict[str, Any]):
        """Save checkpoint for a step"""
        self._state["checkpoints"][f"step{step}"] = checkpoint_data
        self._save()

    def get_checkpoint(self, step: int) -> Optional[Dict[str, Any]]:
        """Get checkpoint for a step"""
        return self._state["checkpoints"].get(f"step{step}")

    def add_error(self, filepath: str, step: int, error: str, retries: int = 0):
        """Add error to state"""
        self._state["errors"].append({
            "file": Path(filepath).name,
            "step": step,
            "error": error,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "retries": retries
        })
        self._save()

    def get_errors(self, filepath: Optional[str] = None, step: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get filtered list of errors"""
        errors = self._state["errors"]

        if filepath:
            errors = [e for e in errors if e["file"] == filepath]

        if step:
            errors = [e for e in errors if e["step"] == step]

        return errors

    def clear_errors(self, filepath: Optional[str] = None, step: Optional[int] = None):
        """Clear errors for a file or step"""
        if filepath:
            self._state["errors"] = [
                e for e in self._state["errors"]
                if e["file"] != filepath
            ]
        elif step:
            self._state["errors"] = [
                e for e in self._state["errors"]
                if e["step"] != step
            ]
        else:
            self._state["errors"] = []

        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        files = self._state["files"]["input"]

        stats = {
            "total_files": len(files),
            "completed_files": 0,
            "incomplete_files": 0,
            "failed_files": 0,
            "not_processed_files": 0,
            "by_step": {
                1: {"completed": 0, "pending": 0},
                2: {"completed": 0, "pending": 0},
                3: {"completed": 0, "pending": 0}
            },
            "errors": len(self._state["errors"])
        }

        for file_info in files.values():
            status = file_info.get("status", "not_processed")

            if status == "complete":
                stats["completed_files"] += 1
            elif status == "incomplete":
                stats["incomplete_files"] += 1
            elif status == "failed":
                stats["failed_files"] += 1
            else:
                stats["not_processed_files"] += 1

            # By step
            for step in [1, 2, 3]:
                if file_info.get(f"step{step}_completed"):
                    stats["by_step"][step]["completed"] += 1
                else:
                    stats["by_step"][step]["pending"] += 1

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Return state as dictionary"""
        return self._state.copy()

    def export(self, filepath: str):
        """Export state to file for backup/analysis"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def reset(self):
        """Reset state (use with caution!)"""
        self._state = {
            "version": "1.0",
            "pipeline": {
                "last_run": None,
                "completed_steps": [],
                "current_step": None,
                "status": "initialized"
            },
            "files": {
                "input": {}
            },
            "checkpoints": {},
            "errors": []
        }
        self._save()
