"""Task 1 data processing operators."""

from src.operators.data_ops.csv_cleaner import clean_csv, validate_cleaning_result
from src.operators.data_ops.csv_profile import profile_csv
from src.operators.data_ops.data_transform import extract_fields_from_text, transform_csv
from src.operators.data_ops.datamate_client import DataMateClient
from src.operators.data_ops.json_loader import json_records_to_csv, load_json_records
from src.operators.data_ops.text_processor import extract_medical_entities, process_text

__all__ = [
    "DataMateClient",
    "clean_csv",
    "extract_fields_from_text",
    "extract_medical_entities",
    "json_records_to_csv",
    "load_json_records",
    "process_text",
    "profile_csv",
    "transform_csv",
    "validate_cleaning_result",
]
