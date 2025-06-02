from ai_prishtina_milvus_client import DataManager, DataValidationConfig

data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
validation_config = DataValidationConfig(required_fields=["id", "text"], field_types={"id": int, "text": str})
manager = DataManager()
valid_data, errors = manager.validate_data(data, validation_config)
print("Valid data:", valid_data)
print("Errors:", errors) 