# -*- coding: utf-8 -*-
from dki.dmcompat import OPERATORS

if "MedicalEntityExtractor" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="MedicalEntityExtractor",
        module_path="dki.operators.medical_entity_extractor.process",
    )
