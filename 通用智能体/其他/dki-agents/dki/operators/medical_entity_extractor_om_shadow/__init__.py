"""Opt-in OM shadow operator package.

Lazy string-path registration (mirrors the five catalog operators) so the
pipeline runner can resolve ``MedicalEntityExtractorOmShadow`` by its formal
name without importing the nested primary implementation at catalog time.
"""

from dki.dmcompat import OPERATORS

if "MedicalEntityExtractorOmShadow" not in OPERATORS.modules:
    OPERATORS.register_module(
        module_name="MedicalEntityExtractorOmShadow",
        module_path="dki.operators.medical_entity_extractor_om_shadow.process",
    )
