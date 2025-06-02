"""
Data management features for Milvus operations including data validation, cleaning, and transformation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
import json
import re
from pathlib import Path

class DataValidationConfig(BaseModel):
    """Configuration for data validation."""
    required_fields: List[str] = Field(default_factory=list, description="Required fields")
    field_types: Dict[str, str] = Field(default_factory=dict, description="Field type mapping (str names)")
    value_ranges: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="Value ranges for fields")
    patterns: Dict[str, str] = Field(default_factory=dict, description="Regex patterns")
    custom_validators: Dict[str, Callable] = Field(default_factory=dict, description="Custom validators")

class DataCleaningConfig(BaseModel):
    """Configuration for data cleaning."""
    remove_duplicates: bool = Field(True, description="Whether to remove duplicates")
    fill_missing_values: bool = Field(True, description="Whether to fill missing values")
    fill_value: Any = Field(None, description="Value to fill missing fields with")
    normalize_numeric_fields: bool = Field(False, description="Whether to normalize numeric fields")
    remove_outliers: bool = Field(False, description="Whether to remove outliers")
    outlier_threshold: float = Field(3.0, description="Standard deviations for outlier detection")

class DataTransformationConfig(BaseModel):
    """Configuration for data transformation."""
    field_mappings: Dict[str, str] = Field(default_factory=dict, description="Field name mappings")
    transformations: Dict[str, Callable] = Field(default_factory=dict, description="Field transformations")
    vector_normalization: bool = Field(True, description="Whether to normalize vectors")
    metadata_extraction: Dict[str, str] = Field(default_factory=dict, description="Metadata extraction rules")

class DataManager:
    """Manager for data operations."""
    
    def __init__(
        self,
        validation_config: Optional[DataValidationConfig] = None,
        cleaning_config: Optional[DataCleaningConfig] = None,
        transformation_config: Optional[DataTransformationConfig] = None
    ):
        """
        Initialize data manager.
        
        Args:
            validation_config: Optional validation configuration
            cleaning_config: Optional cleaning configuration
            transformation_config: Optional transformation configuration
        """
        self.validation_config = validation_config or DataValidationConfig()
        self.cleaning_config = cleaning_config or DataCleaningConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.logger = logging.getLogger(__name__)
        
    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate data according to configuration.
        
        Args:
            data: List of data records
            
        Returns:
            Tuple of (valid records, error messages)
        """
        valid_records = []
        errors = []
        
        for i, record in enumerate(data):
            try:
                # Check required fields
                for field in self.validation_config.required_fields:
                    if field not in record:
                        raise ValueError(f"Missing required field: {field}")
                        
                # Check field types
                for field, expected_type in self.validation_config.field_types.items():
                    if field in record:
                        if not isinstance(record[field], eval(expected_type)):
                            raise ValueError(f"Invalid type for field {field}: expected {expected_type}")
                            
                # Check value ranges
                for field, (min_value, max_value) in self.validation_config.value_ranges.items():
                    if field in record and record[field] < min_value:
                        raise ValueError(f"Value for field {field} below minimum: {min_value}")
                        
                    if field in record and record[field] > max_value:
                        raise ValueError(f"Value for field {field} above maximum: {max_value}")
                        
                # Check patterns
                for field, pattern in self.validation_config.patterns.items():
                    if field in record and not re.match(pattern, str(record[field])):
                        raise ValueError(f"Value for field {field} does not match pattern: {pattern}")
                        
                # Run custom validators
                for field, validator_func in self.validation_config.custom_validators.items():
                    if field in record and not validator_func(record[field]):
                        raise ValueError(f"Custom validation failed for field {field}")
                        
                valid_records.append(record)
                
            except Exception as e:
                errors.append(f"Record {i}: {str(e)}")
                
        return valid_records, errors
        
    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean data according to configuration.
        
        Args:
            data: List of data records
            
        Returns:
            Cleaned data records
        """
        if not data:
            return []
            
        # Remove duplicates
        if self.cleaning_config.remove_duplicates:
            data = self._remove_duplicates(data)
            
        # Fill missing values
        if self.cleaning_config.fill_missing_values:
            data = self._fill_missing_values(data)
            
        # Normalize numeric fields
        if self.cleaning_config.normalize_numeric_fields:
            data = self._normalize_numeric_fields(data)
            
        # Remove outliers
        if self.cleaning_config.remove_outliers:
            data = self._remove_outliers(data)
            
        return data
        
    def transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform data according to configuration."""
        transformed_data = []
        
        for record in data:
            transformed_record = record.copy()
            
            # Apply field mappings
            if self.transformation_config.field_mappings:
                for old_field, new_field in self.transformation_config.field_mappings.items():
                    if old_field in transformed_record:
                        transformed_record[new_field] = transformed_record.pop(old_field)
            
            # Apply transformations
            if self.transformation_config.transformations:
                for field, transform_func in self.transformation_config.transformations.items():
                    if field in transformed_record:
                        transformed_record[field] = transform_func(transformed_record[field])
            
            # Normalize vectors if configured
            if self.transformation_config.vector_normalization and "vector" in transformed_record:
                vector = transformed_record["vector"]
                if isinstance(vector, (list, np.ndarray)):
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        transformed_record["normalized_vector"] = (vector / norm).tolist()
            
            # Extract metadata if configured
            if self.transformation_config.metadata_extraction:
                transformed_record["metadata"] = {}
                for field, rule in self.transformation_config.metadata_extraction.items():
                    if field in transformed_record:
                        transformed_record["metadata"][field] = rule(transformed_record[field])
            
            transformed_data.append(transformed_record)
        
        return transformed_data
        
    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate records.
        
        Args:
            data: List of data records
            
        Returns:
            List of unique records
        """
        seen = set()
        unique_data = []
        
        for record in data:
            # Convert record to tuple of items for hashing
            record_tuple = tuple(sorted(record.items()))
            
            if record_tuple not in seen:
                seen.add(record_tuple)
                unique_data.append(record)
                
        return unique_data
        
    def _fill_missing_values(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fill missing values in records.
        
        Args:
            data: List of data records
            
        Returns:
            List of records with filled values
        """
        filled_data = []
        
        for record in data:
            filled_record = self._fill_missing_values(record)
            filled_data.append(filled_record)
            
        return filled_data
        
    def _fill_missing_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values in a record."""
        filled_record = record.copy()
        if self.cleaning_config.fill_missing_values:
            for field in self.cleaning_config.required_fields:
                if field not in filled_record or filled_record[field] is None:
                    if self.cleaning_config.fill_value is not None:
                        filled_record[field] = self.cleaning_config.fill_value
                    else:
                        # If fill_value is None, use a default value based on field type
                        if isinstance(filled_record.get(field), (int, float)):
                            filled_record[field] = 0
                        elif isinstance(filled_record.get(field), str):
                            filled_record[field] = ""
                        elif isinstance(filled_record.get(field), list):
                            filled_record[field] = []
                        elif isinstance(filled_record.get(field), dict):
                            filled_record[field] = {}
                        else:
                            filled_record[field] = None
        return filled_record
        
    def _normalize_numeric_fields(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize numeric fields.
        
        Args:
            data: List of data records
            
        Returns:
            List of records with normalized fields
        """
        normalized_data = []
        
        # Calculate statistics for each numeric field
        stats = {}
        for field, field_type in self.validation_config.field_types.items():
            if field_type in ("int", "float"):
                values = [record[field] for record in data if field in record]
                if values:
                    stats[field] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
                    
        # Normalize fields
        for record in data:
            normalized_record = record.copy()
            
            for field, stat in stats.items():
                if field in normalized_record:
                    value = normalized_record[field]
                    normalized_record[field] = (value - stat["mean"]) / stat["std"]
                    
            normalized_data.append(normalized_record)
            
        return normalized_data
        
    def _remove_outliers(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove outlier records.
        
        Args:
            data: List of data records
            
        Returns:
            List of records without outliers
        """
        if not data:
            return []
            
        # Calculate statistics for each numeric field
        stats = {}
        for field, field_type in self.validation_config.field_types.items():
            if field_type in ("int", "float"):
                values = [record[field] for record in data if field in record]
                if values:
                    stats[field] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
                    
        # Remove outliers
        filtered_data = []
        for record in data:
            is_outlier = False
            
            for field, stat in stats.items():
                if field in record:
                    value = record[field]
                    z_score = abs((value - stat["mean"]) / stat["std"])
                    if z_score > self.cleaning_config.outlier_threshold:
                        is_outlier = True
                        break
                        
            if not is_outlier:
                filtered_data.append(record)
                
        return filtered_data
        
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            return (vector / norm).tolist()
        return vector.tolist()
        
    def _extract_metadata(self, value: Any, rule: str) -> Any:
        """
        Extract metadata according to rule.
        
        Args:
            value: Value to extract from
            rule: Extraction rule
            
        Returns:
            Extracted metadata
        """
        if rule == "length" and isinstance(value, (str, list, dict)):
            return len(value)
        elif rule == "type":
            return type(value).__name__
        elif rule == "is_empty":
            return not bool(value)
        elif rule == "is_numeric":
            return isinstance(value, (int, float))
        elif rule == "is_string":
            return isinstance(value, str)
        else:
            return value
            
    def export_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Export data to file.
        
        Args:
            data: List of data records
            file_path: Path to export file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def import_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Import data from file.
        
        Args:
            file_path: Path to import file
            
        Returns:
            List of imported records
        """
        with open(file_path, "r") as f:
            return json.load(f) 