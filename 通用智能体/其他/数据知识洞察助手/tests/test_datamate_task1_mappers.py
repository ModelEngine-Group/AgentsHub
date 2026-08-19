import pytest

from adapters.datamate_task1_mappers import ExtractMedicalFieldsMapper, NormalizeMedicalRecordMapper


def test_normalize_medical_record_mapper_strips_text_fields():
    mapper = NormalizeMedicalRecordMapper(columns=["diagnosis", "medicine"])

    result = mapper.execute(
        {
            "patient_id": "P001",
            "diagnosis": " 肺炎  ",
            "medicine": " 阿莫西林   头孢 ",
            "age": 68,
        }
    )

    assert result == {
        "patient_id": "P001",
        "diagnosis": "肺炎",
        "medicine": "阿莫西林 头孢",
        "age": 68,
    }


def test_extract_medical_fields_mapper_keeps_configured_fields():
    mapper = ExtractMedicalFieldsMapper(fields=["patient_id", "diagnosis"])

    result = mapper.execute(
        {
            "patient_id": "P001",
            "diagnosis": "肺炎",
            "medicine": "阿莫西林",
        }
    )

    assert result == {"patient_id": "P001", "diagnosis": "肺炎"}


def test_extract_medical_fields_mapper_rejects_missing_fields():
    mapper = ExtractMedicalFieldsMapper(fields=["patient_id", "diagnosis", "department"])

    with pytest.raises(KeyError, match="DataMate sample missing fields"):
        mapper.execute({"patient_id": "P001", "diagnosis": "肺炎"})
