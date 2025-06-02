"""
Data validation and preprocessing utilities for Milvus operations.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import normalize

class VectorValidationConfig(BaseModel):
    """Configuration for vector validation."""
    expected_dim: int = Field(..., description="Expected vector dimension")
    normalize: bool = Field(False, description="Whether to normalize vectors")
    check_type: bool = Field(True, description="Whether to check vector type")
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")

class DataValidator:
    """Utility class for validating and preprocessing data."""
    
    @staticmethod
    def validate_vectors(
        vectors: List[List[float]],
        config: VectorValidationConfig
    ) -> List[List[float]]:
        """
        Validate and optionally normalize vectors.
        
        Args:
            vectors: List of vectors to validate
            config: Validation configuration
            
        Returns:
            Validated and optionally normalized vectors
            
        Raises:
            ValueError: If validation fails
        """
        if not vectors:
            raise ValueError("Empty vector list provided")
            
        # Convert to numpy array for efficient processing
        vectors_array = np.array(vectors)
        
        # Check dimensions
        if vectors_array.shape[1] != config.expected_dim:
            raise ValueError(
                f"Vector dimension mismatch. Expected {config.expected_dim}, "
                f"got {vectors_array.shape[1]}"
            )
            
        # Check type
        if config.check_type and not np.issubdtype(vectors_array.dtype, np.number):
            raise ValueError(f"Vectors must be numeric, got {vectors_array.dtype}")
            
        # Check value ranges
        if config.min_value is not None:
            if np.any(vectors_array < config.min_value):
                raise ValueError(f"Values below minimum {config.min_value} found")
                
        if config.max_value is not None:
            if np.any(vectors_array > config.max_value):
                raise ValueError(f"Values above maximum {config.max_value} found")
                
        # Normalize if requested
        if config.normalize:
            vectors_array = normalize(vectors_array)
            
        return vectors_array.tolist()
        
    @staticmethod
    def validate_metadata(
        metadata: List[Dict[str, Any]],
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, type]] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate metadata entries.
        
        Args:
            metadata: List of metadata dictionaries
            required_fields: List of required field names
            field_types: Dictionary mapping field names to expected types
            
        Returns:
            Validated metadata
            
        Raises:
            ValueError: If validation fails
        """
        if not metadata:
            raise ValueError("Empty metadata list provided")
            
        # Check required fields
        if required_fields:
            for i, entry in enumerate(metadata):
                missing = [field for field in required_fields if field not in entry]
                if missing:
                    raise ValueError(
                        f"Missing required fields {missing} in metadata entry {i}"
                    )
                    
        # Check field types
        if field_types:
            for i, entry in enumerate(metadata):
                for field, expected_type in field_types.items():
                    if field in entry and not isinstance(entry[field], expected_type):
                        raise ValueError(
                            f"Invalid type for field {field} in metadata entry {i}. "
                            f"Expected {expected_type}, got {type(entry[field])}"
                        )
                        
        return metadata
        
    @staticmethod
    def preprocess_text(
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_whitespace: bool = True
    ) -> str:
        """
        Preprocess text data.
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_whitespace: Whether to remove extra whitespace
            
        Returns:
            Preprocessed text
        """
        import re
        
        if lowercase:
            text = text.lower()
            
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
            
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        if remove_whitespace:
            text = ' '.join(text.split())
            
        return text
        
    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        target_size: Optional[tuple] = None,
        normalize: bool = True,
        convert_to_rgb: bool = True
    ) -> np.ndarray:
        """
        Preprocess image data.
        
        Args:
            image: Input image as numpy array
            target_size: Target size (height, width)
            normalize: Whether to normalize pixel values
            convert_to_rgb: Whether to convert to RGB
            
        Returns:
            Preprocessed image
        """
        from PIL import Image
        
        # Convert to PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        # Convert to RGB if needed
        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize if needed
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
        # Convert back to numpy array
        image_array = np.array(image)
        
        # Normalize if needed
        if normalize:
            image_array = image_array.astype(np.float32) / 255.0
            
        return image_array 