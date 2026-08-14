from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict

PROMPT_VERSION = "nl2sql_candidate_semantic_constraints"

SYSTEM_PROMPT = """你是 ChronicCare SQLite NL2SQL 候选生成器。
只能基于输入中的 schema、字段枚举字典、疾病中英文映射和统计口径约束生成一个只读 SELECT 查询。
字段枚举字典中的英文键是数据库真实存储值；用户使用中文或同义词时，必须转换为对应英文键，禁止把中文标签直接写入 SQL。
统计问题必须严格遵循 statistical_constraints，不得混淆患者人数与记录条数。
只能使用 schema 中列出的表和字段，禁止臆造表、字段或枚举值。
禁止 UPDATE/DELETE/INSERT/DROP/ALTER/CREATE/PRAGMA，禁止多语句和 UNION。
不能确定时返回 {"sql":"UNSUPPORTED","confidence":0,"reason":"原因"}。
能够生成时返回 {"sql":"SELECT ...","confidence":0到1之间的数值,"reason":"简短口径说明"}。
不要输出 Markdown、代码围栏或 JSON 之外的解释。"""


DISEASE_VALUE_DICTIONARY: Dict[str, list[str]] = {
    "hypertension": ["高血压"],
    "hyperlipidemia": ["高脂血症", "血脂异常"],
    "diabetes": ["糖尿病"],
    "obesity": ["肥胖"],
    "coronary_heart_disease": ["冠心病", "冠状动脉粥样硬化性心脏病"],
    "copd": ["慢性阻塞性肺疾病", "慢阻肺", "COPD"],
    "hyperuricemia": ["高尿酸血症"],
    "chronic_kidney_disease": ["慢性肾病", "慢性肾脏病", "CKD"],
    "osteoarthritis": ["骨关节炎"],
    "osteoporosis": ["骨质疏松"],
    "cerebrovascular_disease": ["脑血管病"],
    "atrial_fibrillation": ["心房颤动", "房颤"],
    "fatty_liver_disease": ["脂肪肝"],
    "chronic_heart_failure": ["慢性心力衰竭", "慢性心衰"],
    "gout": ["痛风"],
    "diabetic_kidney_disease": ["糖尿病肾病"],
    "obstructive_sleep_apnea": ["阻塞性睡眠呼吸暂停", "睡眠呼吸暂停", "OSA"],
    "asthma": ["哮喘"],
    "chronic_hepatitis": ["慢性肝炎"],
    "hypothyroidism": ["甲状腺功能减退", "甲减"],
}


FIELD_VALUE_DICTIONARY: Dict[str, Any] = {
    "patient_profile.gender": {"female": ["女", "女性"], "male": ["男", "男性"]},
    "patient_profile.disease_tags": {
        "storage": "以分号分隔的英文疾病代码；疾病筛选使用对应英文代码",
        "disease_code_to_labels": DISEASE_VALUE_DICTIONARY,
    },
    "lab_result.abnormal_flag": {
        "normal": ["正常"], "high": ["偏高", "升高", "高于正常"],
        "low": ["偏低", "降低", "低于正常"],
        "abnormal_semantics": "异常表示 abnormal_flag != 'normal'，即 high 或 low",
    },
    "lab_result.item_name": {
        "hba1c": ["HbA1c", "糖化血红蛋白"], "fasting_glucose": ["空腹血糖"],
        "systolic_bp": ["收缩压"], "diastolic_bp": ["舒张压"],
        "ldl_c": ["LDL-C", "低密度脂蛋白胆固醇"], "hdl_c": ["HDL-C", "高密度脂蛋白胆固醇"],
        "total_cholesterol": ["总胆固醇"], "triglyceride": ["甘油三酯"],
        "creatinine": ["肌酐"], "egfr": ["eGFR", "估算肾小球滤过率"],
        "uric_acid": ["尿酸"], "uacr": ["UACR", "尿白蛋白肌酐比"],
        "alt": ["ALT", "丙氨酸氨基转移酶"], "ast": ["AST", "天门冬氨酸氨基转移酶"],
        "fev1_ratio": ["FEV1比值", "肺功能比值"], "bmi": ["BMI", "体重指数"],
    },
    "followup_plan.status": {"scheduled": ["已计划", "待执行", "待随访", "未完成"], "completed": ["已完成", "完成"]},
    "followup_plan.priority": {"high": ["高", "高优先级"], "medium": ["中", "中优先级"], "low": ["低", "低优先级"]},
    "followup_plan.plan_type": {"lab_followup": ["检验随访"], "routine_followup": ["常规随访"]},
    "lifestyle_record.smoking_status": {"yes": ["吸烟", "现吸烟", "吸烟者"], "no": ["不吸烟", "非吸烟者"]},
    "lifestyle_record.salt_intake_level": {"high": ["高盐", "盐摄入高"], "medium": ["中等盐摄入"], "low": ["低盐", "盐摄入低"]},
    "medication_record.drug_category": {
        "lifestyle": ["生活方式干预"], "glucose_lowering": ["降糖药"],
        "antihypertensive": ["降压药"], "lipid_lowering": ["调脂药", "降脂药"],
        "cardiovascular": ["心血管用药"], "antiplatelet": ["抗血小板药"],
        "uric_acid_lowering": ["降尿酸药"], "liver_management": ["肝病管理用药"],
        "renal_antihypertensive": ["肾病降压药"], "thyroid_management": ["甲状腺用药"],
        "bronchodilator": ["支气管扩张剂"], "bone_health": ["骨健康用药"],
        "anticoagulant": ["抗凝药"], "joint_pain_management": ["关节疼痛管理用药"],
        "airway_management": ["气道管理用药"],
    },
    "doctor_advice.advice_type": {"risk_management": ["风险管理"], "routine_management": ["常规管理"]},
    "doctor_advice.priority": {"high": ["高", "高优先级"], "medium": ["中", "中优先级"], "low": ["低", "低优先级"]},
    "patient_risk_score.risk_level": {"high": ["高风险"], "medium": ["中风险"], "low": ["低风险"]},
    "risk_event.event_level": {"high": ["高风险", "高层级"], "medium": ["中风险", "中层级"], "low": ["低风险", "低层级"]},
}


STATISTICAL_CONSTRAINTS = [
    "问题询问人数、患者数、队列规模、覆盖患者数或某类患者有多少时，使用 COUNT(DISTINCT patient_id)，结果别名优先使用 patient_count。",
    "问题明确询问记录数、条数或次数时，使用 COUNT(*)；不要把记录数当作患者人数。",
    "按类别统计患者覆盖数时，分组后使用 COUNT(DISTINCT patient_id)，不要使用 COUNT(*)。",
    "问题只说分布或汇总且统计对象是医嘱、风险事件、随访计划、检验或用药记录时，默认统计记录数 COUNT(*)；只有明确出现患者、人数、去重人数、队列或覆盖患者时才统计 COUNT(DISTINCT patient_id)。",
    "医嘱优先级或医嘱类型分布默认按 doctor_advice 记录计数并使用 advice_count；风险事件层级分布默认按 risk_event 记录计数并使用 event_count。",
    "询问平均值时使用 AVG(CAST(目标数值字段 AS REAL))；需要展示时可用 ROUND(..., 4)。",
    "疾病筛选必须将中文疾病名转换为 disease_tags 中的英文疾病代码，并按分号分隔的完整标签匹配，例如 (';' || disease_tags || ';') LIKE '%;hypertension;%'，禁止只做可能误命中的任意子串匹配。",
    "检验异常表示 abnormal_flag != 'normal'；偏高和偏低分别使用 high 与 low。",
    "待随访、待执行或未来随访计划对应 followup_plan.status = 'scheduled'。",
    "药物类别覆盖患者数按 medication_record.drug_category 分组并统计 COUNT(DISTINCT patient_id)。",
    "问题中的医生建议、建议类型或建议优先级默认使用 doctor_advice；只有明确提到随访计划时才使用 followup_plan。",
    "问题要求一个汇总数时必须返回聚合结果，禁止用 SELECT * 或患者明细代替汇总数。",
    "枚举字段用于筛选时把中文转换为真实英文存储值；枚举字段出现在 SELECT 结果中时保留数据库英文原值，除非问题明确要求翻译，不要用 CASE 改写为中文标签。",
    "只生成一个 SELECT；允许安全的 JOIN、子查询和 WITH，但禁止 UNION、写操作和多语句。",
]


def llm_available() -> bool:
    if str(os.getenv("OPEN_SQL_LLM_ENABLED", "false")).lower() not in {"1", "true", "yes"}:
        return False
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ensure_proxy_env() -> str | None:
    existing = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    if existing:
        return existing
    proxy_url = os.getenv("OPEN_SQL_PROXY_URL") or os.getenv("CHRONICCARE_PROXY_URL")
    if not proxy_url and _can_connect("172.17.0.1", 17893):
        proxy_url = "http://172.17.0.1:17893"
    if not proxy_url:
        return None
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(key, proxy_url)
    return proxy_url


def generate_llm_sql_candidate(question: str, schema_link: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_available():
        return {"status": "skipped", "reason": "llm_unavailable", "sql": None, "confidence": 0.0}
    _ensure_proxy_env()
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = os.getenv("OPEN_SQL_LLM_MODEL", "deepseek-chat")
    schema_hint = {
        table: [field["name"] for field in meta.get("fields", [])]
        for table, meta in (catalog.get("tables") or {}).items()
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "schema": schema_hint,
                        "schema_link": schema_link,
                        "field_value_dictionary": FIELD_VALUE_DICTIONARY,
                        "statistical_constraints": STATISTICAL_CONSTRAINTS,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        cleaned = str(content).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        parsed = json.loads(match.group(0) if match else cleaned)
        usage = data.get("usage") or {}
        raw_sql = parsed.get("sql")
        if not isinstance(raw_sql, str) or not raw_sql.strip():
            return {
                "status": "failed", "reason": "llm_empty_sql", "sql": None, "confidence": 0.0,
                "model": model, "prompt_version": PROMPT_VERSION, "usage": usage,
            }
        normalized_sql = raw_sql.strip()
        if normalized_sql.upper() == "UNSUPPORTED":
            return {"status": "unsupported", "reason": parsed.get("reason", "LLM returned UNSUPPORTED"), "sql": None, "confidence": 0.0}
        return {
            "status": "success", "sql": normalized_sql,
            "confidence": float(parsed.get("confidence") or 0.0), "reason": parsed.get("reason"),
            "model": model, "endpoint_category": "OpenAI-compatible remote API",
            "temperature": 0, "prompt_version": PROMPT_VERSION, "usage": usage,
        }
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "failed", "reason": str(exc), "sql": None, "confidence": 0.0}
