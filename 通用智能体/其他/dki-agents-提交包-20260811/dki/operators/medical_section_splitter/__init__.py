# -*- coding: utf-8 -*-
from dki.dmcompat import OPERATORS

if "MedicalSectionSplitter" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="MedicalSectionSplitter",
        module_path="dki.operators.medical_section_splitter.process",
    )
