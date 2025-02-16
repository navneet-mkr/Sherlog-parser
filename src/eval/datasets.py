"""Module for loading and managing evaluation datasets."""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass

class DatasetNotFoundError(Exception):
    """Raised when dataset files are not found."""
    pass

class LogTemplate(BaseModel):
    """Model for a log template with metadata."""
    template: str
    frequency: int = Field(default=1)
    parameters: Dict[str, str] = Field(default_factory=dict)

    @field_validator('template')
    @classmethod
    def template_not_empty(cls, v: str) -> str:
        """Validate template is not empty."""
        if not v.strip():
            raise ValueError("Template cannot be empty")
        return v

@dataclass
class LogDataset:
    """Represents a single log dataset with raw logs and ground truth."""
    name: str
    system: str  # e.g., 'Apache', 'Hadoop'
    dataset_type: str  # 'loghub_2k' or 'logpub'
    raw_logs: List[str]
    ground_truth_templates: Dict[int, str]  # log_id -> template
    ground_truth_parameters: Dict[int, Dict[str, str]]  # log_id -> {param_name: value}
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Return number of logs in dataset."""
        return len(self.raw_logs)
    
    def validate(self) -> None:
        """Validate dataset integrity."""
        # Check for empty dataset
        if not self.raw_logs:
            raise DatasetValidationError(f"Dataset {self.name} has no logs")
        
        # Check template coverage
        missing_templates = set(range(len(self.raw_logs))) - set(self.ground_truth_templates.keys())
        if missing_templates:
            raise DatasetValidationError(
                f"Dataset {self.name} is missing templates for log IDs: {missing_templates}"
            )
        
        # Check for empty templates
        empty_templates = [
            log_id for log_id, template in self.ground_truth_templates.items()
            if not template.strip()
        ]
        if empty_templates:
            raise DatasetValidationError(
                f"Dataset {self.name} has empty templates for log IDs: {empty_templates}"
            )
        
        # Validate parameter consistency
        for log_id in self.ground_truth_templates:
            if log_id not in self.ground_truth_parameters:
                logger.warning(
                    f"Dataset {self.name}: Log ID {log_id} has no parameters"
                )

class DatasetLoader:
    """Loader for evaluation datasets."""
    
    REQUIRED_COLUMNS = {
        'structured': {'Content'},  # Minimum required columns
        'templates': {'EventTemplate'}
    }
    
    def __init__(self, base_dir: str = "data/eval_datasets"):
        """Initialize dataset loader.
        
        Args:
            base_dir: Base directory containing evaluation datasets.
        """
        self.base_dir = Path(base_dir)
        self.loghub_dir = self.base_dir / "loghub_2k"
        self.logpub_dir = self.base_dir / "logpub"
        
        # Validate directories exist
        if not self.base_dir.exists():
            raise DatasetNotFoundError(f"Base directory not found at {self.base_dir}")
    
    def _validate_csv_file(self, file_path: Path, required_columns: Set[str]) -> None:
        """Validate CSV file exists and has required columns.
        
        Args:
            file_path: Path to CSV file
            required_columns: Set of required column names
            
        Raises:
            DatasetValidationError: If validation fails
        """
        if not file_path.exists():
            raise DatasetNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, nrows=0)  # Just read headers
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                raise DatasetValidationError(
                    f"File {file_path} missing required columns: {missing_cols}"
                )
        except Exception as e:
            raise DatasetValidationError(f"Error reading {file_path}: {str(e)}")
    
    def _load_structured_logs(self, file_path: Path) -> Tuple[List[str], Dict[int, Dict[str, str]], Dict[int, str]]:
        """Load structured logs from CSV file.
        
        Args:
            file_path: Path to *_structured.csv file.
            
        Returns:
            Tuple of (raw_logs, parameters_dict, event_ids)
            
        Raises:
            DatasetValidationError: If file format is invalid
        """
        try:
            # Validate file
            required_columns = {'Content', 'EventId'}
            self._validate_csv_file(file_path, required_columns)
            
            # Read data
            df = pd.read_csv(file_path)
            
            # Extract raw logs and event IDs
            raw_logs = df['Content'].tolist()
            event_ids = df['EventId'].tolist()
            
            # Extract parameters (if any)
            parameters = {}
            for idx, row in df.iterrows():
                param_dict = {}
                # Look for parameter columns
                for col in df.columns:
                    if col.startswith('ParameterList'):
                        value = row[col]
                        if pd.notna(value):
                            param_dict[f"param_{col.split('.')[-1]}"] = str(value)
                parameters[idx] = param_dict
            
            return raw_logs, parameters, dict(enumerate(event_ids))
        
        except Exception as e:
            raise DatasetValidationError(f"Error loading structured logs: {str(e)}")
    
    def _load_templates(self, file_path: Path) -> Dict[str, str]:
        """Load ground truth templates from CSV file.
        
        Args:
            file_path: Path to *_templates.csv file.
            
        Returns:
            Dictionary mapping EventId to template
            
        Raises:
            DatasetValidationError: If file format is invalid
        """
        try:
            # Validate file
            required_columns = {'EventId', 'EventTemplate'}
            self._validate_csv_file(file_path, required_columns)
            
            # Read data
            df = pd.read_csv(file_path)
            
            # Convert to dictionary mapping EventId to template
            templates = dict(zip(df['EventId'], df['EventTemplate']))
            
            # Validate templates
            empty_templates = [
                event_id for event_id, template in templates.items()
                if not str(template).strip()
            ]
            if empty_templates:
                raise DatasetValidationError(
                    f"Empty templates found for EventIds: {empty_templates}"
                )
            
            return templates
        
        except Exception as e:
            raise DatasetValidationError(f"Error loading templates: {str(e)}")
    
    def load_logs(self, system: str, dataset_type: str) -> pd.DataFrame:
        """Load logs for a specific system and dataset type.
        
        Args:
            system: System name (e.g., 'Apache')
            dataset_type: Dataset type (e.g., 'loghub_2k')
            
        Returns:
            DataFrame with logs and their templates
        """
        structured_log_path = os.path.join(
            self.base_dir,
            dataset_type,
            system,
            f"{system}_{dataset_type}.log_structured.csv"
        )
        
        if not os.path.exists(structured_log_path):
            raise FileNotFoundError(f"Structured log file not found: {structured_log_path}")
            
        df = pd.read_csv(structured_log_path)
        required_columns = {'LineId', 'Content', 'EventTemplate'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Structured log file missing required columns: {required_columns}")
            
        return df

    def load_dataset(self, system: str, dataset_type: str) -> LogDataset:
        """Load a complete dataset including logs and templates.
        
        Args:
            system: System name (e.g., 'Apache')
            dataset_type: Dataset type (e.g., 'loghub_2k')
            
        Returns:
            LogDataset object containing logs and templates
        """
        logs_df = self.load_logs(system, dataset_type)
        
        # Extract unique templates and their IDs
        templates_dict = dict(zip(logs_df['LineId'], logs_df['EventTemplate']))
        
        return LogDataset(
            name=f"{system}_{dataset_type}",
            system=system,
            dataset_type=dataset_type,
            raw_logs=logs_df['Content'].tolist(),
            ground_truth_templates=templates_dict,
            ground_truth_parameters={},  # Empty dict since we don't need parameters for evaluation
            metadata={
                "total_logs": len(logs_df),
                "unique_templates": len(set(logs_df['EventTemplate']))
            }
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
        
        # Helper function to check directory
        def check_dir(path: Path, dataset_type: str) -> None:
            if path.exists():
                available[dataset_type] = [
                    d.name for d in path.iterdir() 
                    if d.is_dir() and 
                    any(d.glob("*.log_structured.csv")) and
                    any(d.glob("*.log_templates.csv"))
                ]
        
        # Check both directories
        check_dir(self.loghub_dir, "loghub_2k")
        check_dir(self.logpub_dir, "logpub")
        
        return available
    
    def load_templates(self, system: str, dataset_type: str) -> pd.DataFrame:
        """Load ground truth templates.
        
        Args:
            system: System name (e.g., 'Apache')
            dataset_type: Dataset type (e.g., 'loghub_2k')
            
        Returns:
            DataFrame with 'EventTemplate' column containing templates
        """
        template_file = self.base_dir / f"{dataset_type}" / system / f"{system}_2k.log_templates.csv"
        return pd.read_csv(template_file)

# Default test datasets for initial development
DEFAULT_TEST_DATASETS = {
    "loghub_2k": ["Apache", "Hadoop", "Linux", "Zookeeper"],
    "logpub": []  # Add default LogPub datasets here
} 