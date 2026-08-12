from adapters.datamate_adapter import CleanTextDataMateMapper


def test_clean_text_datamate_mapper_updates_sample_text_and_keeps_metadata():
    sample = {
        "text": "  发热\n\n  咳嗽   3天。 ",
        "fileName": "record.txt",
        "fileType": "txt",
    }

    result = CleanTextDataMateMapper(remove_digits=True, remove_punctuation=True).execute(sample)

    assert result == {
        "text": "发热 咳嗽 天",
        "fileName": "record.txt",
        "fileType": "txt",
    }


def test_clean_text_datamate_mapper_requires_text_field():
    sample = {"fileName": "missing-text.txt"}

    try:
        _ = CleanTextDataMateMapper().execute(sample)
    except KeyError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError("CleanTextDataMateMapper should require sample['text']")
