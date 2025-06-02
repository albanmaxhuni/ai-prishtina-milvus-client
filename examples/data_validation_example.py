from ai_prishtina_milvus_client import DataValidator, VectorValidationConfig

# Example: Validate and normalize vectors
vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
config = VectorValidationConfig(expected_dim=3, normalize=True)
validated_vectors = DataValidator.validate_vectors(vectors, config)
print("Validated vectors:", validated_vectors) 