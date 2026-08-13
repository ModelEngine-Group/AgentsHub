# -*- coding: utf-8 -*-
from dki.dmcompat import OPERATORS

if "TripleBuilderValidator" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="TripleBuilderValidator",
        module_path="dki.operators.triple_builder_validator.process",
    )
