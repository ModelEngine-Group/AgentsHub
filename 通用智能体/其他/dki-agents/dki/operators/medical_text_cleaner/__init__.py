# -*- coding: utf-8 -*-
from dki.dmcompat import OPERATORS

if "MedicalTextCleaner" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="MedicalTextCleaner",
        module_path="dki.operators.medical_text_cleaner.process",
    )
