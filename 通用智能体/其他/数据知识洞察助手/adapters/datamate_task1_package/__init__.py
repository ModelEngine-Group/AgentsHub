"""DataMate registration example for the task 1 operator package."""

try:
    from datamate.core.base_op import OPERATORS
except ImportError:
    OPERATORS = None

if OPERATORS is not None:
    OPERATORS.register_module(
        module_name="NormalizeMedicalRecord",
        module_path="ops.user.datamate_task1_package.process",
    )
