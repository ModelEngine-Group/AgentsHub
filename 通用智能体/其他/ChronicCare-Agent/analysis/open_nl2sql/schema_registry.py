from __future__ import annotations

from typing import Any, Dict, List, Set

SCHEMA_REGISTRY: List[Dict[str, Any]] = [
    {
        "table": "patient_profile",
        "field": "patient_id",
        "description": "患者编号",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["患者编号", "患者ID"],
    },
    {
        "table": "patient_profile",
        "field": "name",
        "description": "患者姓名",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["姓名", "名字"],
    },
    {
        "table": "patient_profile",
        "field": "gender",
        "description": "患者性别",
        "type": "string",
        "allowed_values": ["male", "female"],
        "chinese_alias": ["性别", "男", "女"],
    },
    {
        "table": "patient_profile",
        "field": "age",
        "description": "患者年龄",
        "type": "number",
        "allowed_values": [],
        "chinese_alias": ["年龄"],
    },
    {
        "table": "patient_profile",
        "field": "bmi",
        "description": "体重指数",
        "type": "number",
        "allowed_values": [],
        "chinese_alias": ["BMI", "体重指数"],
    },
    {
        "table": "patient_profile",
        "field": "disease_tags",
        "description": "患者疾病标签组合",
        "type": "string",
        "allowed_values": ["hypertension", "diabetes", "hyperlipidemia", "obesity", "hyperuricemia", "coronary_risk", "ckd_risk", "fatty_liver_risk"],
        "chinese_alias": ["疾病标签", "疾病类型", "病种", "高血压", "糖尿病", "慢病患者"],
    },
    {
        "table": "followup_plan",
        "field": "followup_date",
        "description": "计划随访日期",
        "type": "date",
        "allowed_values": [],
        "chinese_alias": ["随访日期", "未来随访", "计划日期"],
    },
    {
        "table": "followup_plan",
        "field": "priority",
        "description": "随访优先级",
        "type": "string",
        "allowed_values": ["high", "medium", "low"],
        "chinese_alias": ["优先级", "风险等级", "高风险", "中风险", "低风险"],
    },
    {
        "table": "followup_plan",
        "field": "status",
        "description": "随访状态",
        "type": "string",
        "allowed_values": ["pending", "scheduled", "completed", "cancelled"],
        "chinese_alias": ["状态", "待随访", "已计划"],
    },
    {
        "table": "patient_risk_score",
        "field": "patient_id",
        "description": "风险评分所属患者编号",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["患者编号", "患者ID"],
    },
    {
        "table": "patient_risk_score",
        "field": "risk_level",
        "description": "患者风险等级",
        "type": "string",
        "allowed_values": ["high", "medium", "low"],
        "chinese_alias": ["风险等级", "高风险", "中风险", "低风险"],
    },
    {
        "table": "patient_risk_score",
        "field": "risk_score",
        "description": "患者风险评分",
        "type": "number",
        "allowed_values": [],
        "chinese_alias": ["风险评分"],
    },
    {
        "table": "patient_risk_score",
        "field": "risk_factors",
        "description": "患者风险因素描述",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["风险因素"],
    },
    {
        "table": "lab_result",
        "field": "item_name",
        "description": "检验指标名称",
        "type": "string",
        "allowed_values": ["hba1c", "fasting_glucose", "ldl_c", "systolic_bp", "diastolic_bp", "egfr", "uacr", "uric_acid", "alt"],
        "chinese_alias": ["指标", "HbA1c", "空腹血糖", "LDL-C", "收缩压", "舒张压"],
    },
    {
        "table": "lab_result",
        "field": "item_value",
        "description": "检验数值",
        "type": "number",
        "allowed_values": [],
        "chinese_alias": ["指标值", "检验值"],
    },
    {
        "table": "lab_result",
        "field": "abnormal_flag",
        "description": "检验是否异常",
        "type": "string",
        "allowed_values": ["normal", "high", "low", "abnormal"],
        "chinese_alias": ["异常标记", "异常", "正常"],
    },
    {
        "table": "medication_record",
        "field": "drug_name",
        "description": "药物名称",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["药物", "用药", "药名"],
    },
    {
        "table": "medication_record",
        "field": "drug_category",
        "description": "药物类别",
        "type": "string",
        "allowed_values": ["antihypertensive", "glucose_lowering", "lipid_lowering", "uric_acid_lowering", "lifestyle"],
        "chinese_alias": ["药物类别", "用药类别"],
    },
    {
        "table": "risk_event",
        "field": "event_type",
        "description": "风险事件类型",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["风险事件", "事件类型"],
    },
    {
        "table": "risk_event",
        "field": "event_level",
        "description": "风险事件等级",
        "type": "string",
        "allowed_values": ["high", "medium", "low"],
        "chinese_alias": ["事件等级", "高风险事件"],
    },
    {
        "table": "graph_nodes",
        "field": "entity_type",
        "description": "图谱节点实体类型",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["实体类型", "节点类型"],
    },
    {
        "table": "graph_edges",
        "field": "relation_type",
        "description": "图谱边关系类型",
        "type": "string",
        "allowed_values": [],
        "chinese_alias": ["关系类型", "边类型"],
    },
]


def get_schema_registry() -> List[Dict[str, Any]]:
    return list(SCHEMA_REGISTRY)


def allowed_tables() -> Set[str]:
    return {item["table"] for item in SCHEMA_REGISTRY}


def fields_for_table(table: str) -> Set[str]:
    return {item["field"] for item in SCHEMA_REGISTRY if item["table"] == table}


def allowed_fields() -> Dict[str, Set[str]]:
    return {table: fields_for_table(table) for table in allowed_tables()}
