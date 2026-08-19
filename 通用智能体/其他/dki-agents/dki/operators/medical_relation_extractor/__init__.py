# -*- coding: utf-8 -*-
from dki.dmcompat import OPERATORS

if "MedicalRelationExtractor" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="MedicalRelationExtractor",
        module_path="dki.operators.medical_relation_extractor.process",
    )
