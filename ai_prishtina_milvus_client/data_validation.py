"""
Data validation utilities for Milvus with async support.
"""

import re
from typing import Any, Dict, List, Optional, Union, Awaitable, Callable, Tuple
import asyncio
from pydantic import BaseModel, Field, validator

from .exceptions import ValidationError


class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    required_fields: List[str] = Field(default_factory=list, description="Required fields")
    field_types: Dict[str, str] = Field(default_factory=dict, description="Field type mapping")
    value_ranges: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="Value ranges")
    patterns: Dict[str, str] = Field(default_factory=dict, description="Regex patterns")
    custom_validators: Dict[str, Callable] = Field(default_factory=dict, description="Custom validators")
    validate_vectors: bool = Field(True, description="Validate vector fields")
    vector_dimensions: Optional[int] = Field(None, description="Expected vector dimensions")
    validate_metadata: bool = Field(True, description="Validate metadata fields")
    metadata_schema: Optional[Dict[str, str]] = Field(None, description="Metadata schema")


class VectorValidationConfig(BaseModel):
    """Configuration for vector validation."""
    expected_dim: int = Field(..., description="Expected dimension of vectors")
    normalize: bool = Field(False, description="Whether to normalize vectors")
    check_type: bool = Field(True, description="Whether to check vector type")


class DataValidator:
    """Data validation utilities for Milvus."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
    async def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate data according to configuration asynchronously."""
        try:
            valid_records = []
            errors = []
            
            for i, record in enumerate(data):
                try:
                    await self._validate_record(record)
                    valid_records.append(record)
                except ValidationError as e:
                    errors.append(f"Record {i}: {str(e)}")
                    
            return valid_records, errors
            
        except Exception as e:
            raise ValidationError(f"Data validation failed: {str(e)}")
            
    async def _validate_record(self, record: Dict[str, Any]) -> None:
        """Validate a single record asynchronously."""
        # Check required fields
        for field in self.config.required_fields:
            if field not in record:
                raise ValidationError(f"Missing required field: {field}")
                
        # Check field types
        for field, expected_type in self.config.field_types.items():
            if field in record:
                if not isinstance(record[field], eval(expected_type)):
                    raise ValidationError(f"Invalid type for field {field}: expected {expected_type}")
                    
        # Check value ranges
        for field, (min_value, max_value) in self.config.value_ranges.items():
            if field in record:
                if record[field] < min_value:
                    raise ValidationError(f"Value for field {field} below minimum: {min_value}")
                if record[field] > max_value:
                    raise ValidationError(f"Value for field {field} above maximum: {max_value}")
                    
        # Check patterns
        for field, pattern in self.config.patterns.items():
            if field in record and not re.match(pattern, str(record[field])):
                raise ValidationError(f"Value for field {field} does not match pattern: {pattern}")
                
        # Run custom validators
        for field, validator_func in self.config.custom_validators.items():
            if field in record:
                if asyncio.iscoroutinefunction(validator_func):
                    if not await validator_func(record[field]):
                        raise ValidationError(f"Custom validation failed for field {field}")
                else:
                    if not validator_func(record[field]):
                        raise ValidationError(f"Custom validation failed for field {field}")
                        
        # Validate vectors if enabled
        if self.config.validate_vectors:
            await self._validate_vectors(record)
            
        # Validate metadata if enabled
        if self.config.validate_metadata:
            await self._validate_metadata(record)
            
    async def _validate_vectors(self, record: Dict[str, Any]) -> None:
        """Validate vector fields asynchronously."""
        for field, value in record.items():
            if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                # Check vector dimensions
                if self.config.vector_dimensions is not None:
                    if len(value) != self.config.vector_dimensions:
                        raise ValidationError(
                            f"Vector field {field} has incorrect dimensions: "
                            f"expected {self.config.vector_dimensions}, got {len(value)}"
                        )
                        
                # Check for NaN or infinite values
                if any(not (x == x) or abs(x) == float('inf') for x in value):
                    raise ValidationError(f"Vector field {field} contains NaN or infinite values")
                    
    async def _validate_metadata(self, record: Dict[str, Any]) -> None:
        """Validate metadata fields asynchronously."""
        if "metadata" in record:
            metadata = record["metadata"]
            if not isinstance(metadata, dict):
                raise ValidationError("Metadata must be a dictionary")
                
            if self.config.metadata_schema:
                for field, expected_type in self.config.metadata_schema.items():
                    if field in metadata:
                        if not isinstance(metadata[field], eval(expected_type)):
                            raise ValidationError(
                                f"Invalid type for metadata field {field}: "
                                f"expected {expected_type}"
                            )
                            
    async def validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate collection schema asynchronously."""
        try:
            # Check required fields
            if "fields" not in schema:
                raise ValidationError("Schema must contain 'fields'")
                
            # Validate each field
            for field in schema["fields"]:
                await self._validate_field_schema(field)
                
        except Exception as e:
            raise ValidationError(f"Schema validation failed: {str(e)}")
            
    async def _validate_field_schema(self, field: Dict[str, Any]) -> None:
        """Validate a single field schema asynchronously."""
        # Check required field properties
        required_props = ["name", "dtype"]
        for prop in required_props:
            if prop not in field:
                raise ValidationError(f"Field schema missing required property: {prop}")
                
        # Validate field type
        valid_types = [
            "INT64", "VARCHAR", "FLOAT_VECTOR", "BINARY_VECTOR",
            "BOOL", "INT8", "INT16", "INT32", "FLOAT", "DOUBLE"
        ]
        if field["dtype"] not in valid_types:
            raise ValidationError(f"Invalid field type: {field['dtype']}")
            
        # Validate vector dimensions
        if field["dtype"] in ["FLOAT_VECTOR", "BINARY_VECTOR"]:
            if "dim" not in field:
                raise ValidationError(f"Vector field {field['name']} missing dimension")
            if not isinstance(field["dim"], int) or field["dim"] <= 0:
                raise ValidationError(f"Invalid dimension for vector field {field['name']}")
                
    async def validate_query(self, query: Dict[str, Any]) -> None:
        """Validate search query asynchronously."""
        try:
            # Check required fields
            if "vector" not in query:
                raise ValidationError("Query must contain 'vector'")
                
            # Validate vector
            vector = query["vector"]
            if not isinstance(vector, (list, tuple)):
                raise ValidationError("Query vector must be a list or tuple")
                
            if not all(isinstance(x, (int, float)) for x in vector):
                raise ValidationError("Query vector must contain only numbers")
                
            # Validate optional fields
            if "top_k" in query:
                if not isinstance(query["top_k"], int) or query["top_k"] <= 0:
                    raise ValidationError("top_k must be a positive integer")
                    
            if "metric_type" in query:
                valid_metrics = ["L2", "IP", "COSINE"]
                if query["metric_type"] not in valid_metrics:
                    raise ValidationError(f"Invalid metric_type: {query['metric_type']}")
                    
        except Exception as e:
            raise ValidationError(f"Query validation failed: {str(e)}") 