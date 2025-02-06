"""Module for loading and managing evaluation datasets."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class LogDataset:
    """Represents a single log dataset with raw logs and ground truth."""
    name: str
    system: str  # e.g., 'Apache', 'Hadoop'
    dataset_type: str  # 'loghub_2k' or 'logpub'
    raw_logs: List[str]
    ground_truth_templates: Dict[int, str]  # log_id -> template
    ground_truth_parameters: Dict[int, Dict[str, str]]  # log_id -> {param_name: value}
    
    @property
    def size(self) -> int:
        """Return number of logs in dataset."""
        return len(self.raw_logs)

class DatasetLoader:
    """Handles loading of evaluation datasets."""
    
    def __init__(self, base_dir: str = "data/eval_datasets"):
        """Initialize dataset loader.
        
        Args:
            base_dir: Base directory containing evaluation datasets.
        """
        self.base_dir = Path(base_dir)
        self.loghub_dir = self.base_dir / "loghub_2k"
        self.logpub_dir = self.base_dir / "logpub"
        
        # Validate directories exist
        if not self.loghub_dir.exists():
            raise ValueError(f"Loghub-2k directory not found at {self.loghub_dir}")
        if not self.logpub_dir.exists():
            raise ValueError(f"LogPub directory not found at {self.logpub_dir}")
    
    def _load_structured_logs(self, file_path: Path) -> Tuple[List[str], Dict[int, Dict[str, str]]]:
        """Load structured logs from CSV file.
        
        Args:
            file_path: Path to *_structured.csv file.
            
        Returns:
            Tuple of (raw_logs, parameters_dict)
        """
        df = pd.read_csv(file_path)
        
        # Extract raw logs and parameters
        raw_logs = df['Content'].tolist()
        parameters = {}
        
        # Process each row to extract parameters
        for idx, row in df.iterrows():
            param_dict = {}
            for col in df.columns:
                if col.startswith('ParameterList'):
                    value = row[col]
                    if pd.notna(value):
                        param_dict[f"param_{col.split('.')[-1]}"] = str(value)
            parameters[idx] = param_dict
        
        return raw_logs, parameters
    
    def _load_templates(self, file_path: Path) -> Dict[int, str]:
        """Load ground truth templates from CSV file.
        
        Args:
            file_path: Path to *_templates.csv file.
            
        Returns:
            Dictionary mapping log_id to template
        """
        df = pd.read_csv(file_path)
        return dict(enumerate(df['EventTemplate'].tolist()))
    
    def load_dataset(self, system: str, dataset_type: str = "loghub_2k") -> LogDataset:
        """Load a specific dataset.
        
        Args:
            system: System name (e.g., 'Apache', 'Hadoop')
            dataset_type: Either 'loghub_2k' or 'logpub'
            
        Returns:
            LogDataset object containing raw logs and ground truth
        """
        base = self.loghub_dir if dataset_type == "loghub_2k" else self.logpub_dir
        system_dir = base / system
        
        if not system_dir.exists():
            raise ValueError(f"System directory not found: {system_dir}")
        
        # Find structured and template files
        structured_file = next(system_dir.glob("*.log_structured.csv"), None)
        template_file = next(system_dir.glob("*.log_templates.csv"), None)
        
        if not structured_file or not template_file:
            raise ValueError(f"Required files not found in {system_dir}")
        
        # Load data
        raw_logs, parameters = self._load_structured_logs(structured_file)
        templates = self._load_templates(template_file)
        
        return LogDataset(
            name=f"{system}_{dataset_type}",
            system=system,
            dataset_type=dataset_type,
            raw_logs=raw_logs,
            ground_truth_templates=templates,
            ground_truth_parameters=parameters
        )
    
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List available datasets in the evaluation directory.
        
        Returns:
            Dictionary mapping dataset type to list of available systems
        """
        available = {
            "loghub_2k": [],
            "logpub": []
        }
        
        # Check Loghub-2k
        if self.loghub_dir.exists():
            available["loghub_2k"] = [
                d.name for d in self.loghub_dir.iterdir() 
                if d.is_dir() and any(d.glob("*.log_structured.csv"))
            ]
        
        # Check LogPub
        if self.logpub_dir.exists():
            available["logpub"] = [
                d.name for d in self.logpub_dir.iterdir() 
                if d.is_dir() and any(d.glob("*.log_structured.csv"))
            ]
        
        return available

# Default test datasets for initial development
DEFAULT_TEST_DATASETS = {
    "loghub_2k": ["Apache", "Hadoop", "Linux", "Zookeeper"],
    "logpub": []  # Add default LogPub datasets here
} 