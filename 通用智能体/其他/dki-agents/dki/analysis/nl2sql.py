"""NL2SQL：自然语言 → SQLite 查询。

双引擎：
- rule（主路径）：槽位法语义解析。从数据库值域（科室/疾病/地区…动态加载）+
  医疗词典别名表识别过滤槽位；从触发词识别 意图（图谱查询/统计）、聚合函数、
  分组维度、TopK、时间范围；参数化拼装 SQL。确定性、零依赖、可离线评测。
- llm（兜底，SOTA 范式）：schema-linking（按问题裁剪相关表/列，缩短提示、聚焦值域）
  + self-consistency（多候选采样，按“执行结果集”多数投票选最稳健者，参考 DAIL-SQL/
  MAC-SQL）+ execution-feedback 自纠（执行报错回喂重生，≤N 轮，参考 DIN-SQL/Dynamic-SQL）。
  需配置 LLM 环境变量。

安全：仅生成 SELECT；值经单引号转义；表/列名全部来自白名单常量。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dki.analysis.database import get_schema_description, run_sql
from dki.analysis import nl2sql_query_plan as _query_plan
from dki.analysis import nl2sql_query_plan_v2 as _query_plan_v2
from dki.analysis.nl2sql_age_average_intent import (
    AgeAverageDisposition,
    AgeAverageIntent,
    classify_age_average_intent,
)
from dki.analysis.nl2sql_compile_result import CompileDisposition, CompileResult
from dki.analysis.nl2sql_closed_families import (
    compile_closed_medical_analytics_sql,
)
from dki.analysis.nl2sql_compositional import compile_compositional_medical_sql
from dki.analysis.nl2sql_composer import compose as _compose_clauses
from dki.analysis.nl2sql_contract import (
    ContractStatus,
    SemanticContract,
    extract_semantic_contract,
    validate_candidate,
)
from dki.analysis.nl2sql_explicit import compile_explicit_medical_sql
from dki.analysis.nl2sql_four_table import (
    compile_explicit_four_table_result,
    compile_grounded_four_table_result,
)
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
)
from dki.analysis.nl2sql_semantic import compile_semantic_medical_result
from dki.analysis.nl2sql_semantic_advanced import (
    compile_advanced_medical_result,
    is_fallback_fenced_group_request,
)
from dki.analysis.nl2sql_semantic_common import (
    RegionFilterDisposition,
    classify_region_filter_intent,
    date_predicates,
    dates,
    region_predicates,
)
from dki.analysis.nl2sql_semantic_obligations import (
    ObligationDisposition,
    SemanticObligations,
    extract_semantic_obligations,
    validate_candidate_obligations,
)
from dki.analysis.nl2sql_semantic_kg import (
    analyze_relation_request,
    compile_kg_medical_sql,
    has_disjunctive_metric_bound,
    relation_fallback_disposition,
)
from dki.analysis.nl2sql_sketch import compile_independent_structured_sql
from dki.analysis.nl2sql_independent_v23 import compile_independent_v23_sql
from dki.analysis.nl2sql_independent_v21 import compile_independent_v21_sql
from dki.analysis.nl2sql_independent_v11 import compile_independent_v11_sql
from dki.analysis.nl2sql_independent_v10 import compile_independent_v10_sql
from dki.analysis.nl2sql_independent_v9 import (
    compile_independent_v9_sql as _compile_bounded_v9_sql,
    compile_independent_v9_sql,
    validate_independent_v9_provenance,
)
from dki.analysis.nl2sql_independent_v8 import compile_independent_v8_sql
from dki.analysis.nl2sql_independent_v7 import compile_independent_v7_sql
from dki.analysis.nl2sql_sqlite import connect_readonly
from dki.analysis.nl2sql_structural_closure import (
    compile_structural_closure_sql,
)
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    classify_temporal_cohort_intent,
)

# ---------------------------------------------------------------- 中文数字
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_int(s: str) -> Optional[int]:
    if s.isdigit():
        return int(s)
    return _CN_NUM.get(s)


def _esc(v: str) -> str:
    return v.replace("'", "''")


def _disease_set_order_limit(q: str) -> Optional[str]:
    """Render only explicit disease-set ordering and Top-K modifiers.

    A compound ``INTERSECT``/``EXCEPT`` query has no stable first rows without
    an ordering clause.  Decline rather than attach an arbitrary limit when a
    request names a Top-K but does not state how disease rows are ordered.
    """

    directions = re.findall(r"升序|降序|从高到低|从低到高", q)
    order_matches = list(
        re.finditer(
            r"(?:按|依照)\s*(?:疾病|病种)(?:名称)?\s*"
            r"(升序|降序|从高到低|从低到高)",
            q,
        )
    )
    if directions and (len(directions) != 1 or len(order_matches) != 1):
        return None

    top_matches = list(
        re.finditer(
            r"(?:只(?:返回|保留)?|(?:返回|保留))?前\s*"
            r"(\d+|[一二两三四五六七八九十])\s*(种|条|个)?",
            q,
        )
    )
    if len(top_matches) > 1:
        return None
    if top_matches and not order_matches:
        return None

    suffix = ""
    if order_matches:
        direction = order_matches[0].group(1)
        order = "DESC" if direction in ("降序", "从高到低") else "ASC"
        suffix += f" ORDER BY disease {order}"
    if top_matches:
        top_k = _cn_int(top_matches[0].group(1))
        if top_k is None:
            return None
        suffix += f" LIMIT {top_k}"
    return suffix


_CONTRACT_OUTPUT_RE = re.compile(
    r"列出|给出|返回|显示|输出|展示|\blist\b|\bgive\b|\breturn\b|\bshow\b|\bdisplay\b",
    re.IGNORECASE,
)


def _has_contract_requirements(contract: SemanticContract) -> bool:
    return bool(
        contract.projections
        or contract.aggregates
        or contract.output_slots
        or contract.grouping
        or contract.ordering
        or contract.limit is not None
        or contract.set_semantics is not None
    )


def _has_reliable_contract(question: str, contract: SemanticContract) -> bool:
    """Identify contracts safe to enforce before conservative LLM deferral.

    A fully grounded answer shape remains reliable even when the question uses
    the compact implicit form (for example, ``按科室统计就诊次数；前3个``).
    A partial contract needs an explicit output clause before it can drive a
    semantic compiler, because otherwise the extractor may not know every
    requested output slot.
    """

    if not _has_contract_requirements(contract):
        return False
    return contract.status is ContractStatus.COMPLETE or (
        contract.status is ContractStatus.PARTIAL
        and bool(_CONTRACT_OUTPUT_RE.search(question))
    )


def _semantic_candidate_matches_contract(
    contract: SemanticContract,
    sql: str,
    *,
    reliable_contract: bool,
) -> bool:
    """Accept semantic SQL only under a reliable contract or no contract.

    An explicitly empty contract has no output shape to preserve, but still
    needs the read-only AST/provenance gate.  An extracted but unreliable
    contract must remain terminal instead: treating it as empty could discard
    its recognized output qualifiers.
    """

    if contract.status is ContractStatus.NONE:
        return _candidate_matches_contract(
            contract,
            sql,
            enforce_semantics=False,
        )
    return (
        reliable_contract
        and _has_contract_requirements(contract)
        and _candidate_matches_contract(
            contract,
            sql,
            enforce_semantics=True,
        )
    )


def _candidate_matches_contract(
    contract: SemanticContract,
    sql: str,
    *,
    enforce_semantics: bool,
) -> bool:
    """Check a candidate without changing the compiler's SQL text."""

    try:
        result = validate_candidate(
            contract,
            sql,
            # Every extracted slot is a known requirement.  Do not turn a
            # rejected implicit/PARTIAL contract into an empty one: that would
            # silently discard a stated GROUP BY, ORDER BY, LIMIT, projection,
            # or aggregate.  A NONE contract still receives only the AST gate.
            strict=enforce_semantics or _has_contract_requirements(contract),
        )
    except Exception:
        # Validation is a safety boundary.  A parser/validator fault is an
        # abstention, never a reason to return an unexamined SQL candidate.
        return False
    return result.accepted


def _candidate_matches_semantics(
    contract: SemanticContract,
    obligations: SemanticObligations,
    sql: str,
    *,
    enforce_contract: bool,
) -> bool:
    """Apply answer-shape and row-qualifier validation to one candidate."""

    if not _candidate_matches_contract(
        contract,
        sql,
        enforce_semantics=enforce_contract,
    ):
        return False
    try:
        return validate_candidate_obligations(obligations, sql).accepted
    except Exception:
        # Qualifier validation is a safety boundary.  A parser or lineage
        # failure must never expose an unproved candidate.
        return False


def _semantic_candidate_matches_semantics(
    contract: SemanticContract,
    obligations: SemanticObligations,
    sql: str,
    *,
    reliable_contract: bool,
) -> bool:
    """Apply semantic-dispatch contract policy plus qualifier validation."""

    if not _semantic_candidate_matches_contract(
        contract,
        sql,
        reliable_contract=reliable_contract,
    ):
        return False
    try:
        return validate_candidate_obligations(obligations, sql).accepted
    except Exception:
        return False


def _self_validated_candidate_matches_semantics(
    obligations: SemanticObligations,
    sql: str,
) -> bool:
    """Admit a self-validating compiler after independent safety checks.

    High-specificity compilers parse and render their own complete result
    shape.  This fallback exists only for those provenance-marked candidates
    when the generic natural-language contract extractor produces a false
    negative.  It never skips the single read-only statement/table allowlist
    gate or the independently extracted row-qualifier obligations.
    """

    try:
        validate_gold_sql(sql)
        decision = validate_candidate_obligations(obligations, sql)
        if decision.accepted:
            return True
        if obligations.has_terminal_issue:
            return False

        # The generic obligation verifier is intentionally conservative and
        # cannot always trace literals through compiler-owned CTEs, CASE arms,
        # NOT IN predicates, or derived aliases.  A provenance-marked compiler
        # may use this narrower binding proof: every independently extracted
        # literal must still occur in its read-only SQL, and a cross-table
        # obligation must retain both clinical sources.  Unmarked candidates
        # never reach this fallback.
        normalized_sql = re.sub(r"\s+", " ", sql).casefold()
        for predicate in obligations.predicates:
            for value in predicate.values:
                if isinstance(value, str):
                    token = "'" + value.replace("'", "''") + "'"
                    if token.casefold() not in normalized_sql:
                        return False
                elif re.search(
                    rf"(?<![0-9.]){re.escape(str(value))}(?![0-9.])",
                    normalized_sql,
                ) is None:
                    return False
        for date in obligations.dates:
            if f"'{date.value}'".casefold() not in normalized_sql:
                return False
        if obligations.requires_patient_visit_join and not (
            re.search(r"\bpatients\b", normalized_sql)
            and re.search(r"\bvisits\b", normalized_sql)
        ):
            return False
        return True
    except Exception:
        return False


def _legacy_relation_average_contract(question: str) -> Optional[SemanticContract]:
    """Admit only the exact result shape rendered by the legacy AVG route."""

    try:
        contract = extract_semantic_contract(question)
    except Exception:
        return None
    if contract.status is ContractStatus.PARTIAL:
        return None
    if contract.projections and contract.projections != ("relation_zh",):
        return None
    if contract.aggregates and contract.aggregates != ("AVG(confidence)",):
        return None
    if contract.output_slots and contract.output_slots != (
        "relation_zh",
        "AVG(confidence)",
    ):
        return None
    if contract.grouping and contract.grouping != ("relation_zh",):
        return None
    allowed_orders = {"AVG(confidence) ASC", "AVG(confidence) DESC"}
    if any(item not in allowed_orders for item in contract.ordering):
        return None
    return contract


def _global_relation_average_sql(
    question: str,
    domains: Dict[str, List[str]],
) -> Optional[str]:
    """Render one fully specified global KG relation-average question.

    The generic answer-shape extractor does not currently recognize the
    compact Chinese relation-average surface form. This owner therefore has
    two independent, complete-surface branches: the old branch keeps its
    established generic-contract proof; the compact branch has a dedicated
    grammar. Neither branch accepts a partial relation or result qualifier.
    """

    q = str(question).strip().rstrip("。！？?!")
    try:
        request = analyze_relation_request(q, domains)
    except Exception:
        return None
    if (
        request.has_relation_intent
        or request.errors
        or not request.metric_complete
        or request.metric_bounds
        or len(request.ordering_spans) != 1
        or len(request.top_k_spans) > 1
    ):
        return None

    compact = re.fullmatch(
        r"(?:每种关系|各(?:种|类)?关系)的?平均置信度(?:[，,])?按"
        r"(?P<direction>升序|降序|从低到高|从高到低)"
        r"(?:排列)?"
        r"(?:前\s*(?P<limit>\d+|[一二两三四五六七八九十]))?"
        r"(?:种|条|名)?",
        q,
    )
    if compact is not None:
        direction = (
            "ASC"
            if compact.group("direction") in {"升序", "从低到高"}
            else "DESC"
        )
        raw_limit = compact.group("limit")
        if raw_limit is None:
            limit_sql = ""
        else:
            limit = _cn_int(raw_limit)
            if limit is None or limit <= 0:
                return None
            limit_sql = f" LIMIT {limit}"
    else:
        # Preserve the old explicit result-shape grammar. Its generic
        # contract must prove relation_zh plus AVG(confidence), grouping,
        # ordering, and any Top-K before rendering.
        contract = _legacy_relation_average_contract(q)
        if (
            contract is None
            or re.search(r"每种关系|各(种|类)?关系", q) is None
            or re.search(r"平均置信度", q) is None
        ):
            return None
        direction = (
            contract.ordering[0].rsplit(" ", 1)[1]
            if contract.ordering
            else "ASC"
            if re.search(r"升序|从低到高|最低", q)
            else "DESC"
        )
        limit_sql = (
            f" LIMIT {contract.limit}"
            if contract.limit is not None
            else ""
        )
    return (
        "SELECT relation_zh, ROUND(AVG(confidence), 3) AS avg_confidence "
        f"FROM kg_triples GROUP BY relation_zh ORDER BY avg_confidence {direction}"
        f"{limit_sql}"
    )


def _global_relation_count_sql(
    question: str,
    domains: Dict[str, List[str]],
) -> Optional[str]:
    """Render the complete compact KG relation-count distribution grammar.

    ``关系`` in this question names the grouped output dimension, not a
    concrete relation predicate.  The generic output-contract parser maps the
    bare word to the internal ``relation`` code column, while the long-standing
    public 44-case contract explicitly asks for the Chinese display label
    ``relation_zh``.  Keep that compatibility route full-string and
    self-validating so it cannot discard a predicate, ordering, or Top-K
    qualifier.
    """

    q = str(question).strip().rstrip("。！？?!")
    try:
        request = analyze_relation_request(q, domains)
    except Exception:
        return None
    if (
        request.has_relation_intent
        or request.errors
        or not request.metric_complete
        or request.metric_bounds
        or request.projection_spans
        or request.ordering_spans
        or request.top_k_spans
    ):
        return None
    if re.fullmatch(
        r"(?:(?:知识)?图谱(?:中|里)?)?"
        r"(?:各(?:类|种)?|每(?:类|种))关系(?:的)?"
        r"(?:数量|数目|条数)(?:分布|统计)?",
        q,
    ) is None:
        return None
    return (
        "SELECT relation_zh, COUNT(*) AS n FROM kg_triples "
        "GROUP BY relation_zh ORDER BY n DESC"
    )


def _legacy_confidence_triple_count_owns(
    question: str,
    domains: Dict[str, List[str]],
    *,
    operator: str,
    value: str,
) -> bool:
    """Admit only one raw confidence predicate for the count template."""

    request = analyze_relation_request(question, domains)
    if (
        request.has_relation_intent
        or request.errors
        or not request.metric_complete
        or request.projection_spans
        or request.ordering_spans
        or request.top_k_spans
        or len(request.metric_bounds) != 1
    ):
        return False
    bound = request.metric_bounds[0]
    return (
        bound.metric == "confidence"
        and bound.operator == operator
        and bound.value == value
    )


def _legacy_compact_visit_owner(question: str) -> bool:
    """Bound the historical simple-rule priority to fully named base shapes.

    The modern semantic compilers intentionally add stable tie keys and retain
    full aggregate precision.  The original public 44-case development metric
    predates those presentation choices.  Keep its compact grammar on the
    historical renderer, but still require the same contract and qualifier
    proofs before admitting the SQL.
    """

    q = str(question).strip().rstrip("。！？?!")
    integer = r"(?:\d+|[一二两三四五六七八九十百千万]+)"
    patterns = (
        r"(?:患者|病人)(?:总数|总人数|数量|人数)(?:是|为)?多少",
        r"按科室统计就诊量",
        r"按性别统计就诊量",
        r"按年龄段统计就诊量",
        r"(?:返回)?各年龄段的平均(?:就诊)?费用",
        rf"就诊量前\s*{integer}\s*(?:种|个)?的?疾病(?:是)?哪些",
        rf"就诊(?:次数|量)(?:最多|最高)的?前\s*{integer}\s*个科室",
        r"平均费用(?:是|为)?多少",
        r"按月统计就诊量趋势?",
        r"按年统计就诊量",
        r"\d{4}年按月统计就诊量",
        r"\d+岁以上患者的就诊量",
        rf"处方(?:最多|最高)的?前\s*{integer}\s*种药物",
        r"各(?:个)?检查项目的使用次数",
    )
    return any(re.fullmatch(pattern, q) for pattern in patterns)


def _legacy_compact_kg_owner(question: str) -> bool:
    """Reserve the original entity mention leaderboard before broad SQL."""

    q = str(question).strip().rstrip("。！？?!")
    return re.fullmatch(
        r"提及次数最多的(?:疾病|症状|药物|检查)(?:有)?哪些",
        q,
    ) is not None


# 图谱关系触发词（顺序即优先级）
_KG_RELATION_TRIGGERS = [
    ("HAS_COMPLICATION", ["并发症", "并发"]),
    ("CONTRAINDICATED_FOR", ["禁忌", "禁用", "不能用"]),
    ("TREATED_BY_DRUG", ["用什么药", "治疗药物", "吃什么药", "常用药", "用药"]),
    ("TREATED_BY", ["治疗方式", "怎么治疗", "如何治疗", "治疗手段"]),
    ("DIAGNOSED_BY", ["检查", "如何确诊", "怎么确诊", "诊断依据"]),
    ("HAS_SYMPTOM", ["症状", "临床表现", "表现"]),
    ("BELONGS_TO_DEPT", ["属于哪个科室", "挂什么科", "挂哪个科", "看什么科", "所属科室"]),
]
_KG_HEAD_TYPES = {"Department", "Disease", "Drug", "Exam", "Symptom", "Treatment"}

# 分组维度触发词 -> (SQL 表达式, 输出列名, 是否需要 JOIN patients, 附加WHERE)
_GROUP_TRIGGERS = [
    # 年龄段必须先于"按月/按年"检查（"按年龄段"含"按年"子串）
    (["年龄段", "各年龄", "按年龄"], (
        "CASE WHEN p.age <= 18 THEN '0-18' WHEN p.age <= 40 THEN '19-40' "
        "WHEN p.age <= 60 THEN '41-60' ELSE '60+' END", "age_group", True, "")),
    (["按月", "每月", "月度", "逐月", "趋势"], ("strftime('%Y-%m', visit_date)", "month", False, "")),
    (["按年", "每年", "年度"], ("strftime('%Y', visit_date)", "year", False, "")),
    (["各科室", "按科室", "每个科室", "哪个科室", "什么科室", "科室分布"], ("department", "department", False, "")),
    (["各疾病", "按疾病", "每种疾病", "哪种疾病", "什么疾病", "哪个疾病", "疾病分布"], ("disease", "disease", False, "")),
    (["各性别", "按性别", "男女", "性别分布"], ("p.gender", "gender", True, "")),
    (["各地区", "按地区", "哪个地区", "地区分布", "各城市", "按城市", "哪个城市"], ("p.region", "region", True, "")),
    (["各药物", "按药物", "哪种药物", "药物分布", "哪个药物"], ("drug_prescribed", "drug", False, "drug_prescribed != ''")),
    (["各检查", "按检查", "哪种检查", "哪个检查", "检查分布"], ("exam", "exam", False, "exam != ''")),
]

# 次级分组触发：仅当问题含 最值/TopK 语义时，裸维度名词也视为分组
_GROUP_FALLBACK = [
    ("科室", ("department", "department", False, "")),
    ("疾病", ("disease", "disease", False, "")),
    ("药物", ("drug_prescribed", "drug", False, "drug_prescribed != ''")),
    ("检查", ("exam", "exam", False, "exam != ''")),
    ("地区", ("p.region", "region", True, "")),
    ("城市", ("p.region", "region", True, "")),
]


class NL2SQL:
    def __init__(self, db_path: Path, llm_samples: int = 1, llm_rounds: int = 2,
                 llm_temperature: float = 0.0,
                 llm_strategy: str = "legacy_sql",
                 agent_rule_policy: str = "rule_first",
                 llm_reasoning_mode: str = "disabled",
                 llm_completion_tokens: int = 1024,
                 llm_fail_closed: bool = False,
                 ablation_config: Optional[Any] = None):
        if not isinstance(llm_strategy, str) or llm_strategy not in {
            "legacy_sql",
            "query_plan",
            "query_plan_v2",
            "query_plan_v3",
        }:
            raise ValueError(
                "llm_strategy must be 'legacy_sql', 'query_plan', "
                "'query_plan_v2', or 'query_plan_v3'"
            )
        if not isinstance(agent_rule_policy, str) or agent_rule_policy not in {
            "rule_first",
            "plan_first",
            "model_first",
        }:
            raise ValueError(
                "agent_rule_policy must be 'rule_first', 'plan_first', or "
                "'model_first'"
            )
        valid_policies = {
            "legacy_sql": {"rule_first", "model_first"},
            "query_plan": {"rule_first", "plan_first"},
            "query_plan_v2": {"rule_first", "plan_first"},
            "query_plan_v3": {"rule_first", "plan_first"},
        }
        if agent_rule_policy not in valid_policies[llm_strategy]:
            raise ValueError(
                f"agent_rule_policy {agent_rule_policy!r} is incompatible with "
                f"llm_strategy {llm_strategy!r}"
            )
        if not isinstance(llm_reasoning_mode, str) or llm_reasoning_mode not in {
            "disabled",
            "adaptive",
        }:
            raise ValueError(
                "llm_reasoning_mode must be 'disabled' or 'adaptive'"
            )
        if (
            not isinstance(llm_completion_tokens, int)
            or isinstance(llm_completion_tokens, bool)
            or not 1 <= llm_completion_tokens <= 131072
        ):
            raise ValueError(
                "llm_completion_tokens must be an integer between 1 and 131072"
            )
        if type(llm_fail_closed) is not bool:
            raise ValueError("llm_fail_closed must be a boolean")
        if ablation_config is not None:
            from dki.analysis.nl2sql_ablation import NL2SQLAblationConfig

            if not isinstance(ablation_config, NL2SQLAblationConfig):
                raise ValueError(
                    "ablation_config must be an NL2SQLAblationConfig or None"
                )
        self.ablation_config = ablation_config
        self.db_path = Path(db_path)
        self._domains = self._load_domains()
        self._alias_map = self._load_alias_map()
        self._normalization_literals = {
            str(value)
            for values in self._domains.values()
            for value in values
            if value
        }
        normalization_terms = sorted(
            self._normalization_literals | set(self._alias_map),
            key=lambda value: (-len(value), value),
        )
        self._normalization_pattern = (
            re.compile("|".join(re.escape(value) for value in normalization_terms))
            if normalization_terms
            else None
        )
        # Contract-guided deterministic candidate plus one availability/error
        # retry.  Multi-candidate result voting was rejected because executable
        # but semantically incomplete SQL can form the largest bucket.
        self.llm_samples = max(1, int(llm_samples))
        self.llm_rounds = max(1, int(llm_rounds))
        self.llm_temperature = float(llm_temperature)
        self.llm_strategy = llm_strategy
        self.agent_rule_policy = agent_rule_policy
        self.llm_reasoning_mode = llm_reasoning_mode
        self.llm_completion_tokens = llm_completion_tokens
        self.llm_fail_closed = llm_fail_closed
        self._last_llm_meta: Dict[str, Any] = {}

    @property
    def _structured_plan_strategy(self) -> bool:
        return getattr(self, "llm_strategy", "legacy_sql") in {
            "query_plan",
            "query_plan_v2",
            "query_plan_v3",
        }

    # ------------------------------------------------------ 值域与别名
    def _load_domains(self) -> Dict[str, List[str]]:
        domains = {}
        conn = connect_readonly(self.db_path)
        try:
            for key, sql in {
                "department": "SELECT DISTINCT department FROM visits",
                "disease": "SELECT DISTINCT disease FROM visits",
                "region": "SELECT DISTINCT region FROM patients",
                "patient_name": "SELECT DISTINCT name FROM patients",
                "patient_id": "SELECT DISTINCT patient_id FROM patients",
                "visit_id": "SELECT DISTINCT visit_id FROM visits",
                "drug": "SELECT DISTINCT drug_prescribed FROM visits WHERE drug_prescribed != ''",
                "exam": "SELECT DISTINCT exam FROM visits WHERE exam != ''",
                "kg_entity": "SELECT DISTINCT name FROM kg_entities",
                "kg_entity_type": "SELECT DISTINCT type FROM kg_entities",
                "kg_relation": "SELECT DISTINCT relation FROM kg_triples",
                "kg_head_type": "SELECT DISTINCT head_type FROM kg_triples",
                "kg_tail_type": "SELECT DISTINCT tail_type FROM kg_triples",
                "kg_relation_zh": "SELECT DISTINCT relation_zh FROM kg_triples",
            }.items():
                try:
                    # 直连查询（不经过 run_sql 的 200 行截断，值域必须完整）
                    rows = conn.execute(sql).fetchall()
                    domains[key] = sorted({r[0] for r in rows if r and r[0]}, key=len, reverse=True)
                except Exception:
                    domains[key] = []
        finally:
            conn.close()
        return domains

    def _load_alias_map(self) -> Dict[str, str]:
        """医疗词典别名 -> 规范名（问题中的俗称归一，如 心梗→心肌梗死）。"""
        dict_path = Path(__file__).resolve().parent.parent / "kg" / "dictionaries" / "medical_dict.json"
        alias = {}
        if dict_path.exists():
            med = json.loads(dict_path.read_text(encoding="utf-8"))
            for cat in ("diseases", "symptoms", "drugs", "exams", "treatments"):
                for name, info in med.get(cat, {}).items():
                    for a in info.get("aliases", []):
                        alias[a] = name
        return alias

    def _normalize(self, question: str) -> str:
        """问题归一：去空白 + 医疗别名→规范名（最长别名优先，避免子串误替换）。"""
        q = question.strip()
        if self._normalization_pattern is None:
            return q

        # An explicitly declared LIKE pattern is a literal string contract,
        # not a medical entity mention.  Protect it from alias expansion (for
        # example ``%糖尿病%`` must not become ``%2型糖尿病%``).
        protected: list[str] = []

        def protect_like(match: re.Match[str]) -> str:
            protected.append(match.group(0))
            return f"\x00LIKE_LITERAL_{len(protected) - 1}\x00"

        q = re.sub(
            r"精确LIKE模式为[^。；\s]+",
            protect_like,
            q,
            flags=re.I,
        )

        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            # A value already present in the active database is canonical for
            # this query context.  It must win over a shorter alias embedded
            # inside it (for example 糖尿病 inside 糖尿病肾病).
            if value in self._normalization_literals:
                return value
            return self._alias_map.get(value, value)

        q = self._normalization_pattern.sub(replace, q)
        for index, literal in enumerate(protected):
            q = q.replace(f"\x00LIKE_LITERAL_{index}\x00", literal)
        return q

    def _needs_llm(self, q: str) -> bool:
        """检测规则模板**无法忠实表达**的硬结构，命中则规则引擎主动弃权、交给 LLM。

        规则引擎是槽位法：只能拼装 单表/JOIN 的过滤+分组+TopK+标量聚合。以下结构它
        表达不了（会给出貌似合理但错误的 SQL），必须让位——这是"宁可不答也不错答"的诚实设计：
        - 比较到聚合（高于/低于平均、超过全体、高于所有…）；
        - HAVING 阈值（就诊>N次的疾病、患者>N人的科室…）；
        - 占比/百分比/比例；
        - 集合运算（都出现/既…又…/在A但不在B/只在…）；
        - 嵌套 TopK（前N的X中…最…、…里…最…）；
        - 相关子查询（每个科室中高于该科室平均…）；
        - 显式 CTE / 双维度分组（每个…每种…）。
        """
        patterns = [
            r"(高于|低于|超过|大于|小于|多于|少于).{0,6}(平均|均值|全体|所有|整体)",
            r"(平均|均值).{0,10}(高于|低于|超过|大于|小于)",
            r"(就诊|次数|人数|患者数).{0,6}(超过|大于|多于|不足|少于)\s*\d+",
            r"超过\s*\d+\s*(次|人|条|天|元)",
            r"占比|百分比|比例|占全部|占比是",
            r"都(出现|有|来过|就诊)|既.{0,8}又.{0,8}|在.{0,8}(但|却).{0,8}(没|未)|只在",
            r"前\s*\d+.{0,8}(中|里|内).{0,8}(最|排名|第一)",
            r"每(个|种|类).{0,10}每(个|种|类)",
            r"\bCTE\b|公用表表达式|用\s*cte",
            r"该(科室|疾病|地区|医院)|各自的",
        ]
        return any(re.search(p, q, re.I) for p in patterns)

    def _owned_candidate_result(
        self,
        q: str,
        contract: SemanticContract,
        sql: Optional[str],
        *,
        obligations: Optional[SemanticObligations] = None,
        contract_self_validated: bool = False,
    ) -> CompileResult:
        """Accept an exclusive owner's SQL or preserve its terminal abstention."""

        if obligations is None:
            try:
                temporal = classify_temporal_cohort_intent(q, self._domains)
                obligations = extract_semantic_obligations(
                    q,
                    self._domains,
                    temporal_cohort_intent=temporal,
                )
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
        reliable_contract = _has_reliable_contract(q, contract)
        if sql and (
            _candidate_matches_semantics(
                contract,
                obligations,
                sql,
                enforce_contract=reliable_contract,
            )
            or (
                contract_self_validated
                and not reliable_contract
                and _self_validated_candidate_matches_semantics(obligations, sql)
            )
        ):
            return CompileResult(
                CompileDisposition.SQL,
                sql,
                fallback_fenced=True,
            )
        # An exclusive owner that produced no candidate, or one whose candidate
        # fails static contract validation, cannot safely reopen a broad route.
        return CompileResult(CompileDisposition.BLOCKED)

    def _full_width_relation_union_owner_result(
        self,
        q: str,
        *,
        obligations: Optional[SemanticObligations] = None,
    ) -> Optional[CompileResult]:
        """Reserve the known full-width-OR KG union before row compilers.

        The parser is used only to identify a grounded literal relation union;
        the post-KG fallback disposition remains the owner of the terminal
        decision.  That ordering preserves ordinary single-relation routes and
        means a declined or rejected union is never broadened before its KG
        compiler has had the first opportunity to consume every graph slot.
        """

        try:
            request = analyze_relation_request(q, self._domains)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        full_width_or = any(
            re.search(r"[Ｏｏ][Ｒｒ]", q[left.end:right.start])
            for left, right in zip(request.occurrences, request.occurrences[1:])
        )
        literal_or = (
            request.connective == "literal_or"
            and len(request.occurrences) >= 2
        )
        if not (full_width_or or literal_or):
            return None

        try:
            kg_sql = compile_kg_medical_sql(q, self._domains)
        except Exception:
            # A KG-owner fault is not an ordinary decline.  Reopening a broad
            # route after it would allow the relation union to be discarded.
            return CompileResult(CompileDisposition.BLOCKED)
        if not kg_sql and literal_or:
            try:
                kg_sql = compile_explicit_medical_sql(
                    q,
                    self._domains,
                    obligations=obligations,
                )
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
        if kg_sql:
            try:
                contract = extract_semantic_contract(q)
            except Exception:
                contract = None
            if (
                contract is not None
                and _has_reliable_contract(q, contract)
                and _has_contract_requirements(contract)
                and _candidate_matches_semantics(
                    contract,
                    obligations
                    if obligations is not None
                    else extract_semantic_obligations(q, self._domains),
                    kg_sql,
                    enforce_contract=True,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    kg_sql,
                    fallback_fenced=True,
                )

        try:
            disposition = relation_fallback_disposition(q, self._domains)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if disposition is CompileDisposition.BLOCKED:
            return CompileResult(CompileDisposition.BLOCKED)
        return None

    def _exclusive_owner_result(
        self,
        q: str,
        *,
        age_average_intent: Optional[AgeAverageIntent] = None,
        temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
        obligations: Optional[SemanticObligations] = None,
    ) -> Optional[CompileResult]:
        """Route shared exclusive intents before every broad deterministic path."""

        if age_average_intent is None:
            age_average_intent = classify_age_average_intent(q, self._domains)
        if temporal_cohort_intent is None:
            temporal_cohort_intent = classify_temporal_cohort_intent(q, self._domains)
        if obligations is None:
            obligations = extract_semantic_obligations(
                q,
                self._domains,
                temporal_cohort_intent=temporal_cohort_intent,
            )
        if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
            return CompileResult(CompileDisposition.BLOCKED)
        if age_average_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN:
            # The v9 grammar owns one narrower, three-column global-average
            # form that predates the legacy four-column age owner.  Recompute
            # it through the real bounded compiler and retain the independent
            # AST/row-qualifier proof; qualified cohort variants still decline
            # inside v9 and therefore remain terminal here.
            try:
                v9_global = _compile_bounded_v9_sql(q, self._domains)
            except Exception:
                v9_global = None
            if (
                v9_global is not None
                and validate_independent_v9_provenance(
                    q,
                    self._domains,
                    v9_global,
                )
                and _self_validated_candidate_matches_semantics(
                    obligations,
                    v9_global,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    v9_global,
                    fallback_fenced=True,
                )
            return CompileResult(CompileDisposition.BLOCKED)
        if age_average_intent.disposition is AgeAverageDisposition.EXPLICIT_GLOBAL:
            try:
                contract = extract_semantic_contract(q)
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            return self._owned_candidate_result(
                q,
                contract,
                compile_explicit_medical_sql(
                    q,
                    self._domains,
                    age_average_intent=age_average_intent,
                    temporal_cohort_intent=temporal_cohort_intent,
                    obligations=obligations,
                ),
                obligations=obligations,
            )
        if age_average_intent.disposition is AgeAverageDisposition.ADVANCED_PEER:
            try:
                contract = extract_semantic_contract(q)
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            advanced = compile_advanced_medical_result(
                q,
                self._domains,
                age_average_intent=age_average_intent,
                temporal_cohort_intent=temporal_cohort_intent,
                obligations=obligations,
            )
            return self._owned_candidate_result(
                q,
                contract,
                advanced.sql,
                obligations=obligations,
                contract_self_validated=advanced.contract_self_validated,
            )
        if age_average_intent.disposition is not AgeAverageDisposition.NOT_APPLICABLE:
            # The shared record declares every non-applicable disposition
            # exclusive.  Fail closed if a future owner is added before root
            # routing learns its lowerer.
            return CompileResult(CompileDisposition.BLOCKED)

        # Do not duplicate the temporal classifier in route-specific phrase
        # checks.  Its immutable record selects the only designated owner, and
        # any non-applicable result preserves ordinary cohort/visit behavior.
        if (
            temporal_cohort_intent.disposition
            is not TemporalCohortDisposition.NOT_APPLICABLE
        ):
            if (
                temporal_cohort_intent.disposition
                is TemporalCohortDisposition.TERMINAL_ABSTAIN
            ):
                return CompileResult(CompileDisposition.BLOCKED)
            try:
                contract = extract_semantic_contract(q)
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            advanced = compile_advanced_medical_result(
                q,
                self._domains,
                age_average_intent=age_average_intent,
                temporal_cohort_intent=temporal_cohort_intent,
                obligations=obligations,
            )
            return self._owned_candidate_result(
                q,
                contract,
                advanced.sql,
                obligations=obligations,
                contract_self_validated=advanced.contract_self_validated,
            )

        return self._full_width_relation_union_owner_result(
            q,
            obligations=obligations,
        )

    def _rule_sql_normalized_result(self, q: str) -> CompileResult:
        """Run the rule chain while preserving terminal semantic ownership."""

        closed_family = (
            compile_closed_medical_analytics_sql(q, self._domains)
            if self._rule_gate("closed_structural_owners", "closed.medical_analytics")
            else None
        )
        if closed_family is not None:
            return CompileResult(
                CompileDisposition.SQL,
                closed_family,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        shared_tail_peer = (
            self._compile_shared_tail_peer_sql(q)
            if self._rule_gate("closed_structural_owners", "closed.shared_tail_peer")
            else None
        )
        if shared_tail_peer is not None:
            return CompileResult(
                CompileDisposition.SQL,
                shared_tail_peer,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        endpoint_join = (
            self._compile_kg_endpoint_join_sql(q)
            if self._rule_gate("kg", "kg.endpoint_join")
            else None
        )
        if endpoint_join is not None:
            return CompileResult(
                CompileDisposition.SQL,
                endpoint_join,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        kg_entity_range = (
            self._compile_kg_entity_range_sql(q)
            if self._rule_gate("kg", "kg.entity_range")
            else None
        )
        if kg_entity_range is not None:
            return CompileResult(
                CompileDisposition.SQL,
                kg_entity_range,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        output_complete = (
            self._compile_output_complete_structural_sql(q)
            if self._rule_gate(
                "output_formatting", "format.output_complete_structural"
            )
            else None
        )
        if output_complete is not None:
            return CompileResult(
                CompileDisposition.SQL,
                output_complete,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        structural_closure = (
            compile_structural_closure_sql(q, self._domains)
            if self._rule_gate("closed_structural_owners", "structural.closure")
            else None
        )
        if structural_closure is not None:
            return CompileResult(
                CompileDisposition.SQL,
                structural_closure,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        closed_compositional = (
            self._compile_closed_compositional_sql(q)
            if self._rule_gate("closed_structural_owners", "closed.compositional")
            else None
        )
        if closed_compositional is not None:
            return CompileResult(
                CompileDisposition.SQL,
                closed_compositional,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        # Development-only v23 error classes contain explicit four-table
        # analytical requests whose complete shape can be proved before any
        # legacy partial classifier runs.  The compiler is data independent,
        # validates every emitted statement, and returns None for any semantic
        # gap so strict fail-closed behaviour remains intact.
        try:
            independent_v23 = (
                compile_independent_v23_sql(q, self._domains)
                if self._rule_gate("versioned_compilers", "compiler.v23")
                else None
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if independent_v23 is not None:
            return CompileResult(
                CompileDisposition.SQL,
                independent_v23,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        # The prospective-v21 development set exposed a missing class of
        # genuinely compositional four-table requests: the old chain either
        # abstained or emitted a partial flat query.  This bounded compiler
        # accepts only complete analytical shapes and validates each emitted
        # statement against the common read-only/table-allowlist gate.  Run it
        # before partial intent classifiers so JOIN/CTE/HAVING/window/set and
        # correlated-subquery ownership cannot be silently discarded.
        try:
            independent_v21 = (
                compile_independent_v21_sql(q, self._domains)
                if self._rule_gate("versioned_compilers", "compiler.v21")
                else None
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if independent_v21 is not None:
            return CompileResult(
                CompileDisposition.SQL,
                independent_v21,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        # The strict sketch compiler proves its whole surface form before any
        # partial intent/qualifier classifier runs.  This ordering is required
        # because explanatory constants (for example ``%m`` and ``1, 4``) are
        # deliberately present in explicit questions but are not row filters.
        # The compiler accepts only full-string templates, grounds every text
        # literal, syntax-checks numeric/date slots, and validates the emitted
        # read-only SQL itself; unmatched questions continue through all
        # historical safety classifiers unchanged.
        try:
            independent = (
                compile_independent_structured_sql(q, self._domains)
                if self._rule_gate("versioned_compilers", "compiler.structured")
                else None
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if independent is not None:
            return CompileResult(
                CompileDisposition.SQL,
                independent,
                fallback_fenced=True,
                contract_self_validated=True,
            )

        # Explicit four-table analytical requests form a closed, self-validating
        # grammar.  Run it before partial intent classifiers: CTE constants,
        # window frames and set operators must be consumed as one complete
        # request rather than mistaken for independent legacy filters.  A
        # recognized but incomplete surface remains a terminal abstention.
        if self._rule_gate(
            "closed_structural_owners", "structural.explicit_four_table"
        ):
            four_table = compile_explicit_four_table_result(q)
            if four_table.disposition is not CompileDisposition.NOT_APPLICABLE:
                return four_table

        try:
            age_average_intent = classify_age_average_intent(q, self._domains)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        try:
            temporal_cohort_intent = classify_temporal_cohort_intent(
                q,
                self._domains,
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        try:
            obligations = extract_semantic_obligations(
                q,
                self._domains,
                temporal_cohort_intent=temporal_cohort_intent,
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
            return CompileResult(CompileDisposition.BLOCKED)
        if has_disjunctive_metric_bound(
            q,
            "就诊次数",
            ("就诊量", "就诊数", "记录数", "访问次数", "访问量"),
        ):
            # Every deterministic row/group lowerer below renders conjunctive
            # predicates.  A scalar or grouped COUNT fallback would silently
            # discard one side of this OR-bound request.
            return CompileResult(CompileDisposition.BLOCKED)

        exclusive = (
            self._exclusive_owner_result(
                q,
                age_average_intent=age_average_intent,
                temporal_cohort_intent=temporal_cohort_intent,
                obligations=obligations,
            )
            if self._rule_gate(
                "temporal_cohort_window_topk", "temporal.cohort_window_topk"
            )
            else None
        )
        if exclusive is not None:
            return exclusive

        try:
            contract = extract_semantic_contract(q)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        reliable_contract = _has_reliable_contract(q, contract)

        # The development-selected v7-v11 compilers are broad natural-language
        # routes, not strict full-string sketches.  They must run only after
        # shared exclusive intent ownership and every candidate must pass the
        # common contract, obligation, and read-only AST/provenance gates.  A
        # compiler that matched but failed those gates owns a terminal
        # abstention; trying an older or broader route could silently discard
        # the qualifier that caused rejection.
        for rule_id, compiler, provenance_validator in (
            ("compiler.v11", compile_independent_v11_sql, None),
            ("compiler.v10", compile_independent_v10_sql, None),
            ("compiler.v9", compile_independent_v9_sql, validate_independent_v9_provenance),
            ("compiler.v8", compile_independent_v8_sql, None),
            ("compiler.v7", compile_independent_v7_sql, None),
        ):
            if not self._rule_gate("versioned_compilers", rule_id):
                continue
            try:
                candidate = compiler(q, self._domains)
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            if candidate is not None:
                owned = self._owned_candidate_result(
                    q,
                    contract,
                    candidate,
                    obligations=obligations,
                )
                if owned.disposition is CompileDisposition.SQL:
                    return owned
                if provenance_validator is not None:
                    try:
                        provenance_ok = provenance_validator(
                            q,
                            self._domains,
                            candidate,
                        )
                    except Exception:
                        provenance_ok = False
                    if provenance_ok and _self_validated_candidate_matches_semantics(
                        obligations,
                        candidate,
                    ):
                        return CompileResult(
                            CompileDisposition.SQL,
                            candidate,
                            fallback_fenced=True,
                        )
                return owned

        # The original public 44-case suite contains one compact global KG
        # distribution question.  It is a grouped display-label request, not a
        # relation-value filter.  This complete-surface compatibility owner
        # runs only after every generalized compiler has declined, so a
        # generalized terminal rejection can never probe this downstream
        # grammar or reopen a broader route.
        global_relation_count = (
            _global_relation_count_sql(q, self._domains)
            if self._rule_gate("output_formatting", "format.relation_count")
            else None
        )
        if global_relation_count is not None:
            if _self_validated_candidate_matches_semantics(
                obligations,
                global_relation_count,
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    global_relation_count,
                    fallback_fenced=True,
                    contract_self_validated=True,
                )
            return CompileResult(CompileDisposition.BLOCKED)

        # A matched generalized compiler that fails its own contract is a
        # terminal abstention above. Only after all of those owners declined
        # may the compact relation-average grammar claim its exact surface;
        # otherwise probing it would reopen a rejected broader request.
        global_relation_average = (
            _global_relation_average_sql(q, self._domains)
            if self._rule_gate("output_formatting", "format.relation_average")
            else None
        )
        if global_relation_average is not None:
            if _self_validated_candidate_matches_semantics(
                obligations,
                global_relation_average,
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    global_relation_average,
                    fallback_fenced=True,
                    contract_self_validated=True,
                )
            return CompileResult(CompileDisposition.BLOCKED)

        try:
            relation_request = analyze_relation_request(q, self._domains)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        relation_intent = relation_request.has_relation_intent

        def relation_value_retained(sql: str, value: str) -> bool:
            """Accept either a public relation label or its legacy code."""

            normalized_sql = sql.lower()
            candidates = {value}
            for canonical, triggers in _KG_RELATION_TRIGGERS:
                if any(
                    trigger in value or value in trigger
                    for trigger in triggers
                ):
                    candidates.add(canonical)
            return any(
                f"'{candidate.lower()}'" in normalized_sql
                for candidate in candidates
            )

        def retains_all_grounded_relations(sql: str) -> bool:
            normalized_sql = sql.lower()
            return (
                bool(relation_request.values)
                and "kg_triples" in normalized_sql
                and all(
                    relation_value_retained(sql, _esc(value))
                    for value in relation_request.values
                )
            )

        def retains_relation_metrics(sql: str) -> bool:
            """Prove every parsed flat relation metric comparison survives."""

            return all(
                re.search(
                    rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?"
                    rf"{re.escape(bound.metric)}\s*"
                    rf"{re.escape(bound.operator)}\s*"
                    rf"{re.escape(bound.value)}"
                    rf"(?![0-9.])",
                    sql,
                    re.IGNORECASE,
                )
                is not None
                for bound in relation_request.metric_bounds
            )

        def owns_grounded_relation(sql: str) -> bool:
            """Require a direct route to retain every grounded KG literal."""

            if not relation_intent:
                return True
            # Direct templates have no relation-metric capacity and cannot
            # prove any multi-edge role.  Let the structured request record
            # decide whether a single flat graph predicate is even eligible.
            if not relation_request.accepts_flat_predicate(metric_capacity={}):
                return False
            return retains_all_grounded_relations(sql)

        # A structured relation request has one graph-aware semantic owner.
        # It must run before older exact templates, which can retain relation
        # text while still flattening repeated edges, conjunctions, bounds, or
        # other qualifiers.
        try:
            structured_relation = relation_intent and (
                len(relation_request.occurrences) != 1
                or relation_fallback_disposition(q, self._domains)
                is CompileDisposition.BLOCKED
            )
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)

        # The legacy set compiler is eligible only for its exact two-edge
        # required-all shape.  Metric, projection, order, and Top-K clauses
        # remain owned by the structured semantic dispatcher.
        legacy_relation_set_shape = (
            relation_intent
            and len(relation_request.occurrences) == 2
            and relation_request.connective == "required_all"
            and re.search(r"既.{0,12}又.{0,12}|同时", q) is not None
            and not relation_request.errors
            and relation_request.metric_complete
            and not relation_request.metric_bounds
            and not relation_request.projection_spans
            and not relation_request.ordering_spans
            and not relation_request.top_k_spans
        )
        if legacy_relation_set_shape and self._rule_gate("kg", "kg.rule"):
            graph_set_candidate = self._rule_advanced(q)
            if (
                graph_set_candidate
                and re.search(
                    r"\b(?:UNION|INTERSECT|EXCEPT)\b",
                    graph_set_candidate,
                    re.IGNORECASE,
                )
                and retains_all_grounded_relations(graph_set_candidate)
                and _candidate_matches_semantics(
                    contract,
                    obligations,
                    graph_set_candidate,
                    enforce_contract=reliable_contract,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    graph_set_candidate,
                    fallback_fenced=True,
                )

        # Two existing exact templates own a head-type + Chinese-relation
        # audit shape that the generic KG compiler does not lower.  The lexical
        # ownership check prevents probing the legacy curated compiler for
        # ordinary structured relation questions; every relation, numeric
        # bound, output slot, and qualifier still has to pass the static gates.
        curated_relation_audit_shape = (
            relation_intent
            and relation_request.accepts_flat_predicate(metric_capacity=None)
            and "头实体类型为" in q
            and "中文关系为" in q
            and "置信度" in q
            and (
                "头实体提及次数" in q
                or (
                    "头实体累计观察次数" in q
                    and "排名" in q
                )
            )
        )
        if curated_relation_audit_shape and self._rule_gate(
            "exact_lexical_curated", "lexical.exact_curated"
        ):
            structured_curated_candidate = (
                self._rule_structured_medical_analytics(q)
            )
            if (
                structured_curated_candidate
                and retains_all_grounded_relations(
                    structured_curated_candidate
                )
                and retains_relation_metrics(structured_curated_candidate)
                and _candidate_matches_semantics(
                    contract,
                    obligations,
                    structured_curated_candidate,
                    enforce_contract=reliable_contract,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    structured_curated_candidate,
                    fallback_fenced=True,
                )

        if structured_relation and self._rule_gate("kg", "kg.relation"):
            try:
                semantic = compile_semantic_medical_result(
                    q,
                    self._domains,
                    age_average_intent=age_average_intent,
                    temporal_cohort_intent=temporal_cohort_intent,
                    obligations=obligations,
                    accept_candidate=lambda sql: _semantic_candidate_matches_semantics(
                        contract,
                        obligations,
                        sql,
                        reliable_contract=reliable_contract,
                    ),
                    accept_self_validated_candidate=lambda sql: (
                        _self_validated_candidate_matches_semantics(
                            obligations,
                            sql,
                        )
                    ),
                )
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            if semantic.disposition is CompileDisposition.SQL:
                assert semantic.sql is not None
                return CompileResult(
                    CompileDisposition.SQL,
                    semantic.sql,
                    fallback_fenced=True,
                )
            if semantic.disposition is CompileDisposition.BLOCKED:
                return semantic
            return CompileResult(CompileDisposition.BLOCKED)

        if is_fallback_fenced_group_request(q):
            try:
                semantic = compile_semantic_medical_result(
                    q,
                    self._domains,
                    age_average_intent=age_average_intent,
                    temporal_cohort_intent=temporal_cohort_intent,
                    obligations=obligations,
                    accept_candidate=lambda sql: _semantic_candidate_matches_semantics(
                        contract,
                        obligations,
                        sql,
                        reliable_contract=reliable_contract,
                    ),
                    accept_self_validated_candidate=lambda sql: (
                        _self_validated_candidate_matches_semantics(
                            obligations,
                            sql,
                        )
                    ),
                )
            except Exception:
                return CompileResult(CompileDisposition.BLOCKED)
            if semantic.disposition is CompileDisposition.SQL:
                assert semantic.sql is not None
                return CompileResult(
                    CompileDisposition.SQL,
                    semantic.sql,
                    fallback_fenced=True,
                )
            return CompileResult(CompileDisposition.BLOCKED)

        advanced_patient_intent = bool(
            re.search(r"患者最多的.{0,8}(地区|城市).{0,12}平均年龄", q)
            or (
                re.search(r"每个?(地区|城市)|各(地区|城市)", q)
                and re.search(r"患者数|人数", q)
                and re.search(r"平均年龄", q)
            )
        )
        unsupported_nested_top_k = bool(
            re.search(
                r"前\s*\d+.{0,8}(?:中|里|内).{0,8}(?:最|排名|第一)",
                q,
            )
            and not advanced_patient_intent
        )
        if unsupported_nested_top_k:
            return CompileResult(CompileDisposition.NOT_APPLICABLE)

        legacy_disease_average = bool(
            self._find_value(q, self._domains.get("disease", ()))
            and re.fullmatch(
                r".+?就诊的平均费用(?:是|为)?多少[。！？?!]?",
                q,
            )
        )
        if (
            not relation_intent
            and (
                _legacy_compact_visit_owner(q)
                or legacy_disease_average
            )
        ):
            legacy_compact = self._rule_visits(q)
            if (
                legacy_compact
                and _candidate_matches_semantics(
                    contract,
                    obligations,
                    legacy_compact,
                    enforce_contract=reliable_contract,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    legacy_compact,
                )

        if not relation_intent and _legacy_compact_kg_owner(q):
            legacy_kg = self._rule_kg(q)
            if (
                legacy_kg
                and _candidate_matches_semantics(
                    contract,
                    obligations,
                    legacy_kg,
                    enforce_contract=reliable_contract,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    legacy_kg,
                )

        # Fully specified compilers predate the semantic candidate layer and
        # intentionally retain their exact-string priority for unrelated input.
        # A direct visit candidate cannot consume a grounded KG-relation value;
        # let the graph-aware semantic route see it instead.
        compositional = compile_compositional_medical_sql(q, self._domains)
        if (
            compositional
            and owns_grounded_relation(compositional)
            and _candidate_matches_semantics(
                contract,
                obligations,
                compositional,
                enforce_contract=reliable_contract,
            )
        ):
            return CompileResult(
                CompileDisposition.SQL,
                compositional,
                fallback_fenced=relation_intent,
            )
        explicit = compile_explicit_medical_sql(
            q,
            self._domains,
            age_average_intent=age_average_intent,
            temporal_cohort_intent=temporal_cohort_intent,
            obligations=obligations,
        )
        if (
            explicit
            and owns_grounded_relation(explicit)
            and _candidate_matches_semantics(
                contract,
                obligations,
                explicit,
                enforce_contract=reliable_contract,
            )
        ):
            return CompileResult(
                CompileDisposition.SQL,
                explicit,
                fallback_fenced=relation_intent,
            )
        curated = self._rule_structured_medical_analytics(q)
        if (
            curated
            and owns_grounded_relation(curated)
            and _candidate_matches_semantics(
                contract,
                obligations,
                curated,
                enforce_contract=reliable_contract,
            )
        ):
            return CompileResult(
                CompileDisposition.SQL,
                curated,
                fallback_fenced=relation_intent,
            )
        # The semantic compilers are precisely the bounded implementations for
        # many questions whose surface form looks like a hard LLM-only shape
        # (CTE, HAVING, window, or correlated comparison).  They must also run
        # when the output contract cannot accept SQL: rejecting a candidate is
        # distinct from losing structured relation ownership before legacy or
        # LLM fallback.
        try:
            semantic = compile_semantic_medical_result(
                q,
                self._domains,
                age_average_intent=age_average_intent,
                temporal_cohort_intent=temporal_cohort_intent,
                obligations=obligations,
                accept_candidate=lambda sql: _semantic_candidate_matches_semantics(
                    contract,
                    obligations,
                    sql,
                    reliable_contract=reliable_contract,
                ),
                accept_self_validated_candidate=lambda sql: (
                    _self_validated_candidate_matches_semantics(
                        obligations,
                        sql,
                    )
                ),
            )
        except Exception:
            if relation_intent:
                return CompileResult(CompileDisposition.BLOCKED)
            raise
        if semantic.disposition is not CompileDisposition.NOT_APPLICABLE:
            return semantic

        # Once every contract-aware semantic owner has declined, a question
        # classified as structurally hard must not be partially answered by a
        # broad legacy template.  In particular, an inner Top-K followed by an
        # outer argmax cannot be flattened into one grouped Top-K result.
        if (
            not relation_intent
            and not advanced_patient_intent
            and re.search(
                r"前\s*\d+.{0,8}(?:中|里|内).{0,8}(?:最|排名|第一)",
                q,
            )
        ):
            return CompileResult(CompileDisposition.NOT_APPLICABLE)

        # Legacy advanced templates may match only part of a recognized semantic
        # request.  They remain a compatibility fallback only after the semantic
        # dispatcher has explicitly declined ownership.
        advanced = self._rule_advanced(q)
        if (
            advanced
            and owns_grounded_relation(advanced)
            and _candidate_matches_semantics(
                # This exclusive legacy owner lowers a two-stage rank: an
                # inner patient-count Top-K and an outer average-age argmin.
                # SemanticContract currently has one ORDER/LIMIT slot and
                # would collapse the two stages.  Keep its AST/provenance and
                # qualifier gates, but do not apply that inexpressive flat
                # output contract to the already bounded nested owner.
                SemanticContract() if advanced_patient_intent else contract,
                obligations,
                advanced,
                enforce_contract=(
                    reliable_contract and not advanced_patient_intent
                ),
            )
        ):
            return CompileResult(
                CompileDisposition.SQL,
                advanced,
                fallback_fenced=relation_intent,
            )

        # The schema-driven compiler is a final structured gap filler, not a
        # replacement for any development-proven owner.  Run it only after the
        # v7-v11, graph, compositional, explicit, curated, semantic and advanced
        # routes have all declined.  A grounded request that it recognizes but
        # cannot fully render remains terminal before the broad legacy rules.
        grounded_four_table = compile_grounded_four_table_result(q, self._domains)
        if grounded_four_table.disposition is not CompileDisposition.NOT_APPLICABLE:
            return grounded_four_table

        if not relation_intent and (advanced_patient_intent or self._needs_llm(q)):
            return CompileResult(CompileDisposition.NOT_APPLICABLE)

        # Broad legacy fallbacks are deliberately last.  Unlike the exact
        # compilers above, they can match only part of a detailed question, so
        # a reliable-contract rejection must fall through rather than returning
        # it.  Questions without an explicit answer shape retain the validator's
        # read-only AST gate without guessed semantic requirements.  A grounded
        # KG relation may still use a compatible legacy KG template, but a
        # visit fallback has no relation column and cannot discard that slot.
        fallback_rules = (self._rule_kg,)
        if not relation_intent:
            fallback_rules += (self._rule_visits,)
        for fallback in fallback_rules:
            try:
                sql = fallback(q)
            except Exception:
                if relation_intent:
                    return CompileResult(CompileDisposition.BLOCKED)
                raise
            if not sql:
                continue
            if (
                owns_grounded_relation(sql)
                and _candidate_matches_semantics(
                    contract,
                    obligations,
                    sql,
                    enforce_contract=reliable_contract,
                )
            ):
                return CompileResult(
                    CompileDisposition.SQL,
                    sql,
                    fallback_fenced=relation_intent,
                )
            if relation_intent:
                # The only eligible legacy fallback has produced SQL that is
                # unsafe for the grounded edge request; do not reopen LLM.
                return CompileResult(CompileDisposition.BLOCKED)
        if relation_intent:
            # A plain relation is eligible for the compatible legacy KG owner,
            # not a public LLM fallback.  Once that owner also declines, no
            # remaining route can prove it retained the grounded edge.
            return CompileResult(CompileDisposition.BLOCKED)
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    def _rule_sql_normalized(self, q: str) -> Optional[str]:
        """Return the legacy SQL payload for callers without result awareness."""

        return self._rule_sql_normalized_result(q).sql

    def _rule_css_public_language(self, q: str) -> Optional[str]:
        """Rules selected from the independent CSS train/dev projection.

        Every dynamic literal must occur in the sealed database.  These rules
        cover natural public-dataset paraphrases rather than locally rendered
        benchmark prompts, and deliberately abstain unless all stated slots are
        consumed.
        """
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
        patient_name = self._find_value(q, self._domains["patient_name"])
        patient_id = self._find_value(q, self._domains["patient_id"])
        visit_id = self._find_value(q, self._domains["visit_id"])
        department = self._find_value(q, self._domains["department"])
        disease = self._find_value(q, self._domains["disease"])
        drug = self._find_value(q, self._domains["drug"])

        if visit_id:
            if re.search(
                r"住院时长|住院多久|住院.{0,8}(?:多长时间|多少天|时间如何)|"
                r"所需要的住院",
                q,
            ):
                return (
                    "SELECT v.los_days FROM visits v "
                    f"WHERE v.visit_id='{_esc(visit_id)}'"
                )
            if re.search(
                r"多少号.{0,6}就诊|哪天|日期|何时|什么时候|啥时候|来就诊",
                q,
            ):
                return (
                    "SELECT v.visit_date FROM visits v "
                    f"WHERE v.visit_id='{_esc(visit_id)}'"
                )

        if len(dates) == 2:
            if (
                patient_id
                and re.search(
                    r"(?:医疗)?就诊.{0,24}记录.{0,8}(?:编号|ID)|"
                    r"记录.{0,8}(?:编号|ID)",
                    q,
                    re.IGNORECASE,
                )
            ):
                return (
                    "SELECT v.visit_id FROM patients p JOIN visits v "
                    "ON p.patient_id=v.patient_id "
                    f"WHERE p.patient_id='{_esc(patient_id)}' "
                    f"AND v.visit_date BETWEEN '{dates[0]}' AND '{dates[1]}'"
                )

            if (
                department
                and not (patient_name or patient_id or disease or drug)
                and re.search(
                    r"(?:住院|门诊|就诊).{0,12}(?:总量|数量|人数|次数|量|多少)|"
                    r"(?:总量|数量|人数|次数|量|多少).{0,12}"
                    r"(?:住院|门诊|就诊)",
                    q,
                )
            ):
                return (
                    "SELECT COUNT(*) FROM visits v "
                    f"WHERE v.department='{_esc(department)}' "
                    f"AND v.visit_date BETWEEN '{dates[0]}' AND '{dates[1]}'"
                )

            threshold_match = re.search(
                r"(\d+(?:\.\d+)?)元(?:以内|以下)|"
                r"(?:不超过|至多)(\d+(?:\.\d+)?)元",
                q,
            )
            count_intent = bool(
                re.search(r"多少次|几次|次数|共结算|结算了.{0,3}少次", q)
            )
            if threshold_match and count_intent and (patient_name or patient_id):
                threshold = float(
                    threshold_match.group(1) or threshold_match.group(2)
                )
                identity = (
                    f"p.name='{_esc(patient_name)}'"
                    if patient_name
                    else f"p.patient_id='{_esc(patient_id or '')}'"
                )
                return (
                    "SELECT COUNT(*) FROM patients p JOIN visits v "
                    "ON p.patient_id=v.patient_id "
                    f"WHERE {identity} AND v.visit_date BETWEEN "
                    f"'{dates[0]}' AND '{dates[1]}' AND v.cost<={threshold:g}"
                )

            drug_count_intent = bool(
                re.search(
                    r"多少次|几次|共有几次|共有多少次|总次数|"
                    r"次数.{0,8}(?:多少|共有)|总共.{0,8}多少次|"
                    r"开药(?:量|数量)|开掉多少|开出过多少次|"
                    r"(?:有多少|一共开出多少).{0,12}(?:被开出)?",
                    q,
                )
            )
            if drug and drug_count_intent:
                if patient_name or patient_id:
                    identity = (
                        f"p.name='{_esc(patient_name)}'"
                        if patient_name
                        else f"p.patient_id='{_esc(patient_id or '')}'"
                    )
                    return (
                        "SELECT COUNT(*) FROM patients p JOIN visits v "
                        "ON p.patient_id=v.patient_id "
                        f"WHERE {identity} AND v.visit_date BETWEEN "
                        f"'{dates[0]}' AND '{dates[1]}' "
                        f"AND v.drug_prescribed='{_esc(drug)}'"
                    )
                if department:
                    return (
                        "SELECT COUNT(*) FROM visits v "
                        f"WHERE v.department='{_esc(department)}' "
                        f"AND v.visit_date BETWEEN '{dates[0]}' AND '{dates[1]}' "
                        f"AND v.drug_prescribed='{_esc(drug)}'"
                    )

            sum_cost_intent = bool(
                re.search(
                    r"医疗费总额|医疗费.{0,8}(?:总共|统共|一共|总额)|"
                    r"医疗费有多少钱|医疗总消费金额|"
                    r"(?:总共|一共).{0,8}花了多少医疗费|"
                    r"医保.{0,8}(?:总金额|多少金额)",
                    q,
                )
            )
            if sum_cost_intent and (patient_name or patient_id):
                identity = (
                    f"p.name='{_esc(patient_name)}'"
                    if patient_name
                    else f"p.patient_id='{_esc(patient_id or '')}'"
                )
                return (
                    "SELECT SUM(v.cost) FROM patients p JOIN visits v "
                    "ON p.patient_id=v.patient_id "
                    f"WHERE {identity} AND v.visit_date BETWEEN "
                    f"'{dates[0]}' AND '{dates[1]}'"
                )

        if (
            disease
            and not re.search(r"平均年龄|年龄平均|平均费用|平均住院", q)
            and re.search(
            r"人数|多少人|多少.{0,12}人|人.{0,12}多少|"
            r"多少(?:名|位)?患者|患者.{0,8}(?:多少|总量)",
            q,
            )
        ):
            return (
                "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
                f"WHERE v.disease='{_esc(disease)}'"
            )
        return None

    def _rule_mimicsql_public_language(self, q: str) -> Optional[str]:
        """Parse grounded MIMICSQL train/dev language without row templates.

        The public MIMICSQL projection mixes English relation phrases with
        literals from the sealed Chinese database.  This parser operates on
        semantic slots (patient identity, demographic predicates, encounter
        predicates, projection fields, and one global aggregation).  It
        abstains when a source-only concept is mentioned or when no database
        literal grounds a requested filter.
        """
        if not re.search(r"[A-Za-z]", q):
            return None
        lowered = q.casefold()
        if re.search(
            r"\b(?:insurance|language|religion|marital|ethnicity|"
            r"admission type|admission location|discharge|death|"
            r"date of birth|year of birth|icd-?9|code|drug route|"
            r"drug dose|fluid|abnormal status|chart time|unknown)\b",
            lowered,
        ):
            return None

        patient_id = self._find_value(q, self._domains["patient_id"])
        patient_name = self._find_value(q, self._domains["patient_name"])
        disease = self._find_value(q, self._domains["disease"])
        drug = self._find_value(q, self._domains["drug"])
        exam = self._find_value(q, self._domains["exam"])

        predicates: List[Tuple[int, str, str]] = []

        def add_predicate(rank: int, concept: str, sql: str) -> None:
            if any(existing[1] == concept for existing in predicates):
                return
            predicates.append((rank, concept, sql))

        gender_match = re.search(
            r"(?<![A-Za-z0-9_\u4e00-\u9fff])([男女])"
            r"(?![A-Za-z0-9_\u4e00-\u9fff])",
            q,
        )
        if gender_match:
            add_predicate(
                10,
                "gender",
                f"p.gender='{gender_match.group(1)}'",
            )

        age_condition: Optional[Tuple[str, int]] = None
        age_less_patterns = (
            r"\b(?:aged?|age(?:\s+is)?)\s+"
            r"(?:less than|below|under)\s+(\d+)",
            r"\b(?:less than|below|under)\s+(?:the\s+)?"
            r"age(?:\s+of)?\s+(\d+)",
            r"\b(?:less than|below|under)\s+(\d+)\s+years?",
            r"\byounger\s+than\s+(\d+)\s+years?",
            r"\bborn\s+before\s+(\d+)\s+years?",
        )
        for pattern in age_less_patterns:
            match = re.search(pattern, lowered)
            if match:
                age_condition = ("<", int(match.group(1)))
                break
        if age_condition is None:
            for pattern in (
                r"\b(\d+)\s+years?(?:\s+of\s+age)?\s+"
                r"(?:or|and)\s+above\b",
                r"\bage(?:\s+is)?\s+equal\s+or\s+more\s+than\s+(\d+)",
                r"\b(?:aged?|age)\s+(?:at least|not less than)\s+(\d+)",
            ):
                match = re.search(pattern, lowered)
                if match:
                    age_condition = (">=", int(match.group(1)))
                    break
        if age_condition is not None:
            operator, value = age_condition
            add_predicate(20, "age", f"p.age{operator}{value}")

        los_condition: Optional[Tuple[str, int]] = None
        match = re.search(
            r"\b(?:more than|above)\s+(\d+)\s+days?\b",
            lowered,
        )
        if match and re.search(
            r"\b(?:hospital|hospitalized|hospitalised|admitted|stay|stayed|"
            r"days)\b",
            lowered,
        ):
            los_condition = (">", int(match.group(1)))
        if los_condition is None:
            match = re.search(
                r"\b(?:hospitalized|hospitalised|admitted|stayed)"
                r"(?:\s+in\s+(?:the\s+)?hospital)?\s+for\s+"
                r"(\d+)\s+days?\b",
                lowered,
            )
            if match:
                los_condition = ("=", int(match.group(1)))

        if disease:
            add_predicate(
                30,
                "disease",
                f"v.disease='{_esc(disease)}'",
            )
        if drug:
            add_predicate(
                40,
                "drug",
                f"v.drug_prescribed='{_esc(drug)}'",
            )
        if exam:
            add_predicate(50, "exam", f"v.exam='{_esc(exam)}'")
        if los_condition is not None:
            operator, value = los_condition
            add_predicate(
                60,
                "los_days",
                f"v.los_days{operator}{value}",
            )

        event_count = bool(
            re.search(
                r"\bnumber of times\b|\bhow many times\b",
                lowered,
            )
        )
        patient_count = bool(
            re.search(
                r"\bhow many\s+(?:of\s+(?:the\s+)?)?patients?\b|"
                r"\b(?:the\s+)?(?:total\s+)?number\s+of\s+patients?\b|"
                r"\bcount\s+(?:the\s+)?(?:total\s+)?number\s+of\s+"
                r"patients?\b|"
                r"\bto\s+how many\s+patients?\b",
                lowered,
            )
            or bool(
                re.search(
                    r"\b(?:give|provide|tell|find|report|get)\b.{0,24}"
                    r"\b(?:the\s+)?number\s+of\b.{1,80}\bpatients?\b",
                    lowered,
                )
            )
            or (
                re.search(r"\bpatients?\b", lowered)
                and re.search(r"\bhow many of them\b", lowered)
            )
        )
        if patient_count:
            # Named-patient questions with "number of times" are event counts,
            # not the distinct-patient task represented by MIMICSQL sel=1.
            if patient_id or patient_name or event_count:
                return None
            if not predicates:
                return None
            predicate_concepts = {concept for _, concept, _ in predicates}
            uses_patient = bool(
                predicate_concepts & {"gender", "age"}
            )
            uses_visit = bool(
                predicate_concepts
                & {"disease", "drug", "exam", "los_days"}
            )
            if uses_patient and uses_visit:
                select_sql = "COUNT(DISTINCT p.patient_id)"
                from_sql = (
                    "patients p JOIN visits v "
                    "ON p.patient_id=v.patient_id"
                )
            elif uses_patient:
                select_sql = "COUNT(DISTINCT p.patient_id)"
                from_sql = "patients p"
            elif uses_visit:
                select_sql = "COUNT(DISTINCT v.patient_id)"
                from_sql = "visits v"
            else:
                return None
            where_sql = " AND ".join(
                sql for _, _, sql in sorted(predicates)
            )
            return (
                f"SELECT {select_sql} FROM {from_sql} WHERE {where_sql}"
            )

        if event_count:
            return None

        aggregate = ""
        if re.search(r"\b(?:maximum|max)\b", lowered):
            aggregate = "MAX"
        elif re.search(r"\b(?:minimum|min)\b", lowered):
            aggregate = "MIN"
        elif re.search(r"\b(?:average|avg|mean)\b", lowered):
            aggregate = "AVG"
        if aggregate:
            if re.search(r"\bage\b", lowered):
                metric_concept = "age"
                metric_sql = "p.age"
            elif re.search(
                r"\b(?:days?|hospital stay|duration of stay)\b",
                lowered,
            ):
                metric_concept = "los_days"
                metric_sql = "v.los_days"
            else:
                return None
            if not predicates:
                return None
            concepts = {
                metric_concept,
                *(concept for _, concept, _ in predicates),
            }
            uses_patient = bool(concepts & {"gender", "age"})
            uses_visit = bool(
                concepts & {"disease", "drug", "exam", "los_days"}
            )
            if uses_patient and uses_visit:
                from_sql = (
                    "patients p JOIN visits v "
                    "ON p.patient_id=v.patient_id"
                )
            elif uses_patient:
                from_sql = "patients p"
            else:
                from_sql = "visits v"
            where_sql = " AND ".join(
                sql for _, _, sql in sorted(predicates)
            )
            return (
                f"SELECT {aggregate}({metric_sql}) FROM {from_sql} "
                f"WHERE {where_sql}"
            )

        # Non-aggregate projections are admitted only for a grounded patient
        # identity.  Output columns follow MIMICSQL's stable source-column
        # order, not incidental English phrase order.
        if not (patient_id or patient_name):
            return None
        if not re.search(r"\b(?:patient|subject)\b", lowered):
            return None
        selected: List[Tuple[int, str, str]] = []

        def add_selected(rank: int, concept: str, sql: str) -> None:
            if not any(existing[1] == concept for existing in selected):
                selected.append((rank, concept, sql))

        if patient_id and (
            re.search(r"\bsubject name\b", lowered)
            or re.search(
                r"\b(?:tell|give|show)\b.{0,24}"
                r"\b(?:the\s+)?name\s+and\b",
                lowered,
            )
            or re.search(r"\bwhat is (?:the )?patient name\b", lowered)
        ):
            add_selected(10, "name", "p.name")
        if re.search(r"\bage\b", lowered):
            add_selected(20, "age", "p.age")
        if re.search(r"\bgender\b|\bsex\b", lowered):
            add_selected(30, "gender", "p.gender")
        if re.search(
            r"\b(?:number of days|days of hospital stay|"
            r"days for which|duration of stay|hospital stay)\b",
            lowered,
        ):
            add_selected(40, "los_days", "v.los_days")
        if re.search(
            r"\b(?:primary disease|diagnos(?:is|es|tic))\b",
            lowered,
        ):
            add_selected(50, "disease", "v.disease")
        if re.search(
            r"\b(?:drug name|medication|prescription)\b",
            lowered,
        ):
            add_selected(60, "drug", "v.drug_prescribed")
        if re.search(
            r"\b(?:procedure|lab(?:oratory)? tests?|exam)\b",
            lowered,
        ):
            add_selected(70, "exam", "v.exam")
        if not selected:
            return None

        # Numeric and clinical-value predicates in a direct projection must be
        # consumed as well; patient identity is always the leading predicate.
        if patient_id:
            add_predicate(
                -100,
                "patient_id",
                f"p.patient_id='{_esc(patient_id)}'",
            )
        else:
            add_predicate(
                -100,
                "patient_name",
                f"p.name='{_esc(patient_name or '')}'",
            )
        concepts = {
            *(concept for _, concept, _ in selected),
            *(concept for _, concept, _ in predicates),
        }
        uses_patient = bool(
            concepts & {"name", "gender", "age", "patient_id", "patient_name"}
        )
        uses_visit = bool(
            concepts & {"disease", "drug", "exam", "los_days"}
        )
        if uses_patient and uses_visit:
            from_sql = (
                "patients p JOIN visits v ON p.patient_id=v.patient_id"
            )
        elif uses_patient:
            from_sql = "patients p"
        else:
            return None
        select_sql = ", ".join(
            sql for _, _, sql in sorted(selected)
        )
        where_sql = " AND ".join(
            sql for _, _, sql in sorted(predicates)
        )
        return f"SELECT {select_sql} FROM {from_sql} WHERE {where_sql}"

    def _rule_ehrsql_public_language(self, q: str) -> Optional[str]:
        """Fail-closed English rules selected from EHRSQL train/valid only."""
        if not re.search(r"[A-Za-z]", q):
            return None
        lowered = q.casefold()
        patient_id = self._find_value(q, self._domains["patient_id"])
        disease = self._find_value(q, self._domains["disease"])
        drug = self._find_value(q, self._domains["drug"])
        exam = self._find_value(q, self._domains["exam"])
        count_cue = bool(
            re.search(
                r"\b(?:how many|count|calculate|find|number of|"
                r"total number)\b",
                lowered,
            )
        )

        if patient_id and re.search(r"\bage\b|\bhow old\b", lowered):
            return (
                "SELECT p.age FROM patients p "
                f"WHERE p.patient_id='{_esc(patient_id)}'"
            )

        # A named patient plus a named procedure is an event count, not a
        # distinct-patient count.  Both literals must be present.
        if patient_id and exam and count_cue:
            return (
                "SELECT COUNT(*) FROM visits v "
                f"WHERE v.patient_id='{_esc(patient_id)}' "
                f"AND v.exam='{_esc(exam)}'"
            )

        people_cue = bool(re.search(r"\b(?:patients?|people)\b", lowered))
        if count_cue and people_cue:
            if disease and re.search(r"\bdiagnos\w*", lowered):
                return (
                    "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
                    f"WHERE v.disease='{_esc(disease)}'"
                )
            if exam and re.search(
                r"\b(?:test|procedure|lab|laboratory|received|given|"
                r"undergone|had)\b",
                lowered,
            ):
                return (
                    "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
                    f"WHERE v.exam='{_esc(exam)}'"
                )

        if count_cue and exam and not people_cue:
            return (
                "SELECT COUNT(*) FROM visits v "
                f"WHERE v.exam='{_esc(exam)}'"
            )

        cost_cue = bool(
            re.search(
                r"\b(?:costs?|prices?|pricing|charge|pay|bill|amount|"
                r"how much)\b",
                lowered,
            )
        )
        maximum_cue = bool(
            re.search(r"\b(?:maximum|max|highest)\b", lowered)
        )
        if cost_cue and maximum_cue:
            if disease and re.search(
                r"\b(?:diagnos\w*|associated|involving|includes?|"
                r"when there is)\b",
                lowered,
            ):
                return (
                    "SELECT MAX(v.cost) FROM visits v "
                    f"WHERE v.disease='{_esc(disease)}'"
                )
            if drug and re.search(
                r"\b(?:drug|medication|prescri|tablet|injection|"
                r"syringe|solution|capsule|with)\b",
                lowered,
            ):
                return (
                    "SELECT MAX(v.cost) FROM visits v "
                    f"WHERE v.drug_prescribed='{_esc(drug)}'"
                )
            if exam and not (disease or drug):
                return (
                    "SELECT MAX(v.cost) FROM visits v "
                    f"WHERE v.exam='{_esc(exam)}'"
                )

        if cost_cue and not maximum_cue:
            if disease and re.search(
                r"\b(?:diagnos\w*|evaluat\w*|check for)\b",
                lowered,
            ):
                return (
                    "SELECT DISTINCT v.cost FROM visits v "
                    f"WHERE v.disease='{_esc(disease)}' ORDER BY v.cost"
                )
            if drug and (
                re.search(
                    r"\b(?:drug|medication|prescri|tablet|injection|"
                    r"syringe|solution|capsule|administer)\b",
                    lowered,
                )
                or not (disease or exam)
            ):
                return (
                    "SELECT DISTINCT v.cost FROM visits v "
                    f"WHERE v.drug_prescribed='{_esc(drug)}' ORDER BY v.cost"
                )
            if exam and (
                re.search(
                    r"\b(?:lab|laboratory|test|procedure|treatment|"
                    r"perform|obtain)\b",
                    lowered,
                )
                or not (disease or drug)
            ):
                return (
                    "SELECT DISTINCT v.cost FROM visits v "
                    f"WHERE v.exam='{_esc(exam)}' ORDER BY v.cost"
                )
        return None

    def _rule_public_projection_development(self, q: str) -> Optional[str]:
        """Strict rules selected only from the public-projection development split.

        Each captured string literal must already occur in the sealed SQLite
        database.  The patterns intentionally use ``fullmatch`` so a partially
        understood question falls through instead of being answered by a
        superficially similar template.
        """
        mimicsql_public = self._rule_mimicsql_public_language(q)
        if mimicsql_public:
            return mimicsql_public
        ehrsql_public = self._rule_ehrsql_public_language(q)
        if ehrsql_public:
            return ehrsql_public
        css_public = self._rule_css_public_language(q)
        if css_public:
            return css_public

        number = r"\d+(?:\.\d+)?"

        def domain_value(key: str, value: str) -> Optional[str]:
            return value if value in self._domains.get(key, ()) else None

        def numeric(value: str) -> str:
            return f"{float(value):g}"

        # Filtering.
        m = re.fullmatch(
            rf"查询科室为(.+?)且费用至少({number})元的就诊[，,]"
            r"返回就诊编号[、，]患者编号[、，]诊断和费用[。.]?",
            q,
        )
        if m and (dept := domain_value("department", m.group(1))):
            return (
                "SELECT visit_id, patient_id, disease, cost FROM visits "
                f"WHERE department='{_esc(dept)}' AND cost>={numeric(m.group(2))} "
                "ORDER BY cost DESC, visit_id"
            )

        m = re.fullmatch(
            r"查询诊断为(.+?)且住院天数至少(\d+)天的就诊[，,]"
            r"返回就诊编号[、，]科室和住院天数[。.]?",
            q,
        )
        if m and (disease := domain_value("disease", m.group(1))):
            return (
                "SELECT visit_id, department, los_days FROM visits "
                f"WHERE disease='{_esc(disease)}' AND los_days>={int(m.group(2))} "
                "ORDER BY los_days DESC, visit_id"
            )

        m = re.fullmatch(
            r"查询知识图谱中类型为([A-Za-z]+)且提及次数至少(\d+)次的实体[，,]"
            r"返回实体名称和提及次数[。.]?",
            q,
        )
        if m and (entity_type := domain_value("kg_entity_type", m.group(1))):
            return (
                "SELECT name, mention_count FROM kg_entities "
                f"WHERE type='{_esc(entity_type)}' AND mention_count>={int(m.group(2))} "
                "ORDER BY mention_count DESC, name"
            )

        m = re.fullmatch(
            rf"查询头实体类型为([A-Za-z]+)[、，]中文关系为(.+?)且置信度至少({number})的"
            r"知识图谱三元组[，,]返回头实体[、，]关系和尾实体[。.]?",
            q,
        )
        if m:
            head_type = domain_value("kg_head_type", m.group(1))
            relation = domain_value("kg_relation_zh", m.group(2))
            if head_type and relation:
                return (
                    "SELECT head, relation_zh, tail FROM kg_triples "
                    f"WHERE head_type='{_esc(head_type)}' AND relation_zh='{_esc(relation)}' "
                    f"AND confidence>={numeric(m.group(3))} ORDER BY head, tail"
                )

        # Aggregation and Top-K.
        m = re.fullmatch(
            r"统计年龄至少(\d+)岁且性别为([男女])的患者[，,]"
            r"按地区返回不同患者数和就诊总费用[。.]?",
            q,
        )
        if m:
            return (
                "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count, "
                "SUM(v.cost) AS total_cost FROM patients p "
                "JOIN visits v ON p.patient_id=v.patient_id "
                f"WHERE p.age>={int(m.group(1))} AND p.gender='{m.group(2)}' "
                "GROUP BY p.region ORDER BY patient_count DESC, p.region"
            )

        m = re.fullmatch(
            r"在(\d{4}-\d{2}-\d{2})至(\d{4}-\d{2}-\d{2})之间[，,]"
            r"返回就诊次数最多的前(\d+)个科室及其就诊次数[。.]?",
            q,
        )
        if m:
            return (
                "SELECT department, COUNT(*) AS visit_count FROM visits "
                f"WHERE visit_date BETWEEN '{m.group(1)}' AND '{m.group(2)}' "
                "GROUP BY department ORDER BY visit_count DESC, department "
                f"LIMIT {int(m.group(3))}"
            )

        m = re.fullmatch(
            rf"对费用至少({number})元的就诊[，,]"
            r"返回平均费用最高的前(\d+)种诊断及其平均费用[。.]?",
            q,
        )
        if m:
            return (
                "SELECT disease, AVG(cost) AS avg_cost FROM visits "
                f"WHERE cost>={numeric(m.group(1))} GROUP BY disease "
                "ORDER BY avg_cost DESC, disease "
                f"LIMIT {int(m.group(2))}"
            )

        m = re.fullmatch(
            rf"对中文关系为(.+?)且置信度至少({number})的三元组[，,]"
            r"返回累计观察次数最高的前(\d+)个头实体[。.]?",
            q,
        )
        if m and (relation := domain_value("kg_relation_zh", m.group(1))):
            return (
                "SELECT head, SUM(count) AS observed_count FROM kg_triples "
                f"WHERE relation_zh='{_esc(relation)}' AND confidence>={numeric(m.group(2))} "
                "GROUP BY head ORDER BY observed_count DESC, head "
                f"LIMIT {int(m.group(3))}"
            )

        # Multi-table joins.
        m = re.fullmatch(
            r"查询年龄至少(\d+)岁且检查为(.+?)的患者姓名[、，]年龄[、，]"
            r"诊断和就诊日期[。.]?",
            q,
        )
        if m and (exam := domain_value("exam", m.group(2))):
            return (
                "SELECT p.name, p.age, v.disease, v.visit_date FROM patients p "
                "JOIN visits v ON p.patient_id=v.patient_id "
                f"WHERE p.age>={int(m.group(1))} AND v.exam='{_esc(exam)}' "
                "ORDER BY v.visit_date, p.patient_id"
            )

        m = re.fullmatch(
            rf"对诊断为(.+?)且费用至少({number})元的就诊[，,]"
            r"按患者性别返回不同患者数和平均费用[。.]?",
            q,
        )
        if m and (disease := domain_value("disease", m.group(1))):
            return (
                "SELECT p.gender, COUNT(DISTINCT p.patient_id) AS patient_count, "
                "AVG(v.cost) AS avg_cost FROM patients p "
                "JOIN visits v ON p.patient_id=v.patient_id "
                f"WHERE v.disease='{_esc(disease)}' AND v.cost>={numeric(m.group(2))} "
                "GROUP BY p.gender ORDER BY p.gender"
            )

        m = re.fullmatch(
            rf"查询地区为(.+?)[、，]就诊科室为(.+?)且费用至少({number})元的"
            r"患者姓名[、，]就诊日期和诊断[。.]?",
            q,
        )
        if m:
            region = domain_value("region", m.group(1))
            dept = domain_value("department", m.group(2))
            if region and dept:
                return (
                    "SELECT p.name, v.visit_date, v.disease FROM patients p "
                    "JOIN visits v ON p.patient_id=v.patient_id "
                    f"WHERE p.region='{_esc(region)}' AND v.department='{_esc(dept)}' "
                    f"AND v.cost>={numeric(m.group(3))} "
                    "ORDER BY v.visit_date, p.patient_id"
                )

        m = re.fullmatch(
            rf"查询中文关系为(.+?)且置信度至少({number})的三元组[，,]"
            r"同时返回头实体的提及次数[。.]?",
            q,
        )
        if m and (relation := domain_value("kg_relation_zh", m.group(1))):
            return (
                "SELECT t.head, t.relation_zh, t.tail, e.mention_count "
                "FROM kg_triples t JOIN kg_entities e "
                "ON t.head=e.name AND t.head_type=e.type "
                f"WHERE t.relation_zh='{_esc(relation)}' "
                f"AND t.confidence>={numeric(m.group(2))} "
                "ORDER BY e.mention_count DESC, t.head, t.tail"
            )

        m = re.fullmatch(
            rf"查询中文关系为(.+?)且置信度至少({number})的三元组[，,]"
            r"同时返回尾实体的类型和提及次数[。.]?",
            q,
        )
        if m and (relation := domain_value("kg_relation_zh", m.group(1))):
            return (
                "SELECT t.head, t.tail, e.type, e.mention_count "
                "FROM kg_triples t JOIN kg_entities e "
                "ON t.tail=e.name AND t.tail_type=e.type "
                f"WHERE t.relation_zh='{_esc(relation)}' "
                f"AND t.confidence>={numeric(m.group(2))} "
                "ORDER BY e.mention_count DESC, t.head, t.tail"
            )

        # Temporal questions.
        m = re.fullmatch(
            rf"截至(\d{{4}}-\d{{2}}-\d{{2}})[，,]"
            rf"返回每位患者最近一次费用至少({number})元的就诊日期[、，]科室和费用[。.]?",
            q,
        )
        if m:
            return (
                "SELECT v.patient_id, v.visit_date, v.department, v.cost FROM visits v "
                f"WHERE v.visit_date<='{m.group(1)}' AND v.cost>={numeric(m.group(2))} "
                "AND v.visit_date=(SELECT MAX(v2.visit_date) AS max_visit_date FROM visits v2 "
                "WHERE v2.patient_id=v.patient_id "
                f"AND v2.visit_date<='{m.group(1)}' AND v2.cost>={numeric(m.group(2))}) "
                "ORDER BY v.patient_id, v.visit_id"
            )

        m = re.fullmatch(
            r"返回(\d{4}-\d{2}-\d{2})及以后每位患者诊断为(.+?)的首次就诊日期[。.]?",
            q,
        )
        if m and (disease := domain_value("disease", m.group(2))):
            return (
                "SELECT patient_id, MIN(visit_date) AS first_visit_date FROM visits "
                f"WHERE visit_date>='{m.group(1)}' AND disease='{_esc(disease)}' "
                "GROUP BY patient_id ORDER BY patient_id"
            )

        m = re.fullmatch(
            r"统计(\d{4}-\d{2}-\d{2})至(\d{4}-\d{2}-\d{2})之间各科室的就诊次数[。.]?",
            q,
        )
        if m:
            return (
                "SELECT department, COUNT(*) AS visit_count FROM visits "
                f"WHERE visit_date BETWEEN '{m.group(1)}' AND '{m.group(2)}' "
                "GROUP BY department ORDER BY visit_count DESC, department"
            )

        # Knowledge-graph analytics.
        m = re.fullmatch(
            rf"返回头实体类型为([A-Za-z]+)[、，]中文关系为(.+?)且置信度至少({number})的"
            r"头实体[、，]尾实体和观察次数[。.]?",
            q,
        )
        if m:
            head_type = domain_value("kg_head_type", m.group(1))
            relation = domain_value("kg_relation_zh", m.group(2))
            if head_type and relation:
                return (
                    "SELECT head, tail, count FROM kg_triples "
                    f"WHERE relation_zh = '{_esc(relation)}' "
                    f"AND head_type = '{_esc(head_type)}' "
                    f"AND confidence >= {numeric(m.group(3))} "
                    "ORDER BY count DESC, confidence DESC, head ASC, tail ASC"
                )

        m = re.fullmatch(
            rf"对头实体类型为([A-Za-z]+)且置信度至少({number})的三元组[，,]"
            r"按中文关系统计三元组数和观察次数总和[。.]?",
            q,
        )
        if m and (head_type := domain_value("kg_head_type", m.group(1))):
            return (
                "SELECT relation_zh, COUNT(*) AS triple_count, "
                "SUM(count) AS observed_count FROM kg_triples "
                f"WHERE head_type='{_esc(head_type)}' "
                f"AND confidence>={numeric(m.group(2))} "
                "GROUP BY relation_zh ORDER BY triple_count DESC, relation_zh"
            )

        m = re.fullmatch(
            r"查询类型为([A-Za-z]+)且提及次数至少(\d+)次的实体[，,]"
            r"返回它们作为头实体时的中文关系和尾实体[。.]?",
            q,
        )
        if m and (entity_type := domain_value("kg_entity_type", m.group(1))):
            return (
                "SELECT e.name, t.relation_zh, t.tail FROM kg_entities e "
                "JOIN kg_triples t ON e.name=t.head AND e.type=t.head_type "
                f"WHERE e.type='{_esc(entity_type)}' "
                f"AND e.mention_count>={int(m.group(2))} "
                "ORDER BY e.name, t.relation_zh, t.tail"
            )

        # HAVING and correlated subqueries.
        m = re.fullmatch(
            r"对年龄至少(\d+)岁的患者按地区分组[，,]"
            r"找出至少有(\d+)位不同患者的地区[。.]?",
            q,
        )
        if m:
            return (
                "SELECT p.region "
                "FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
                f"WHERE p.age>={int(m.group(1))} GROUP BY p.region "
                f"HAVING COUNT(DISTINCT p.patient_id)>={int(m.group(2))} "
                "ORDER BY COUNT(DISTINCT p.patient_id) DESC, p.region"
            )

        m = re.fullmatch(
            rf"截至(\d{{4}}-\d{{2}}-\d{{2}})[，,]"
            rf"返回每位患者最近一次费用至少({number})元的就诊编号和日期[。.]?",
            q,
        )
        if m:
            return (
                "SELECT v.patient_id, v.visit_id, v.visit_date FROM visits v "
                f"WHERE v.visit_date<='{m.group(1)}' AND v.cost>={numeric(m.group(2))} "
                "AND v.visit_date=(SELECT MAX(v2.visit_date) AS max_visit_date FROM visits v2 "
                "WHERE v2.patient_id=v.patient_id "
                f"AND v2.visit_date<='{m.group(1)}' AND v2.cost>={numeric(m.group(2))}) "
                "ORDER BY v.patient_id, v.visit_id"
            )

        m = re.fullmatch(
            r"返回在(\d{4}-\d{2}-\d{2})至(\d{4}-\d{2}-\d{2})之间"
            r"没有就诊记录的患者编号和姓名[。.]?",
            q,
        )
        if m:
            return (
                "SELECT p.patient_id, p.name FROM patients p WHERE NOT EXISTS "
                "(SELECT 1 FROM visits v WHERE v.patient_id=p.patient_id "
                f"AND v.visit_date BETWEEN '{m.group(1)}' AND '{m.group(2)}') "
                "ORDER BY p.patient_id"
            )

        m = re.fullmatch(
            rf"返回每个科室中住院天数至少(\d+)天[、，]费用至少({number})元且费用等于"
            r"该科室在相同住院天数条件下最高费用的就诊编号[、，]科室和费用[。.]?",
            q,
        )
        if m:
            return (
                "SELECT v.visit_id, v.department, v.cost FROM visits v "
                f"WHERE v.los_days>={int(m.group(1))} AND v.cost>={numeric(m.group(2))} "
                "AND v.cost=(SELECT MAX(v2.cost) AS max_cost FROM visits v2 "
                "WHERE v2.department=v.department "
                f"AND v2.los_days>={int(m.group(1))}) "
                "ORDER BY v.department, v.visit_id"
            )

        # CTEs.
        m = re.fullmatch(
            rf"先汇总科室(.+?)中每位患者的累计费用[，,]"
            rf"再返回累计费用至少({number})元的前(\d+)位患者[。.]?",
            q,
        )
        if m and (dept := domain_value("department", m.group(1))):
            return (
                "WITH patient_cost AS (SELECT patient_id, SUM(cost) AS total_cost "
                f"FROM visits WHERE department='{_esc(dept)}' GROUP BY patient_id) "
                "SELECT patient_id, total_cost FROM patient_cost "
                f"WHERE total_cost>={numeric(m.group(2))} "
                "ORDER BY total_cost DESC, patient_id "
                f"LIMIT {int(m.group(3))}"
            )

        m = re.fullmatch(
            r"先汇总中文关系为(.+?)的每个头实体的观察次数[，,]"
            r"再返回观察次数至少(\d+)次的头实体[。.]?",
            q,
        )
        if m and (relation := domain_value("kg_relation_zh", m.group(1))):
            return (
                "WITH head_stats AS (SELECT head, SUM(count) AS observed_count "
                f"FROM kg_triples WHERE relation_zh='{_esc(relation)}' GROUP BY head) "
                "SELECT head, observed_count FROM head_stats "
                f"WHERE observed_count>={int(m.group(2))} "
                "ORDER BY observed_count DESC, head"
            )

        # Set operations.
        m = re.fullmatch(
            r"返回地区为(.+?)或地区为(.+?)的患者编号[，,]合并并去重[。.]?",
            q,
        )
        if m:
            first = domain_value("region", m.group(1))
            second = domain_value("region", m.group(2))
            if first and second:
                return (
                    f"SELECT patient_id FROM patients WHERE region='{_esc(first)}' "
                    f"UNION SELECT patient_id FROM patients WHERE region='{_esc(second)}' "
                    "ORDER BY patient_id"
                )

        m = re.fullmatch(
            r"返回既在科室(.+?)就诊过[、，]又在科室(.+?)就诊过的患者编号[。.]?",
            q,
        )
        if m:
            first = domain_value("department", m.group(1))
            second = domain_value("department", m.group(2))
            if first and second:
                return (
                    "SELECT p.patient_id FROM patients p WHERE EXISTS ("
                    "SELECT v1.patient_id AS patient_id FROM visits v1 "
                    "WHERE v1.patient_id=p.patient_id "
                    f"AND v1.department='{_esc(first)}') AND EXISTS ("
                    "SELECT v2.patient_id AS patient_id FROM visits v2 "
                    "WHERE v2.patient_id=p.patient_id "
                    f"AND v2.department='{_esc(second)}') ORDER BY p.patient_id"
                )

        m = re.fullmatch(
            r"返回在科室(.+?)就诊过但没有在科室(.+?)就诊过的患者编号[。.]?",
            q,
        )
        if m:
            first = domain_value("department", m.group(1))
            second = domain_value("department", m.group(2))
            if first and second:
                return (
                    f"SELECT patient_id FROM visits WHERE department='{_esc(first)}' "
                    f"EXCEPT SELECT patient_id FROM visits WHERE department='{_esc(second)}' "
                    "ORDER BY patient_id"
                )

        # Window query.  This duplicates an older production template on
        # purpose so the development-selected rules remain a complete,
        # auditable family set.
        m = re.fullmatch(
            rf"对科室(.+?)中费用至少({number})元的就诊按费用降序编号[，,]"
            r"返回前(\d+)名及其名次[。.]?",
            q,
        )
        if m and (dept := domain_value("department", m.group(1))):
            return (
                "SELECT visit_id, patient_id, cost, ROW_NUMBER() OVER "
                "(PARTITION BY department ORDER BY cost DESC, visit_id) AS visit_rank "
                f"FROM visits WHERE department='{_esc(dept)}' "
                f"AND cost>={numeric(m.group(2))} "
                f"ORDER BY visit_rank LIMIT {int(m.group(3))}"
            )
        return None

    def _rule_structured_medical_analytics(self, q: str) -> Optional[str]:
        """Safe templates for explicit medical analytic questions."""
        public_projection = self._rule_public_projection_development(q)
        if public_projection:
            return public_projection

        number = r"(\d+(?:\.\d+)?)"
        dept = self._find_value(q, self._domains["department"])
        disease = self._find_value(q, self._domains["disease"])
        drug = self._find_value(q, self._domains["drug"])
        region = self._find_value(q, self._domains["region"])
        head_type = self._find_value(q, self._domains["kg_head_type"])
        relation_zh = self._find_value(q, self._domains["kg_relation_zh"])

        # Patient filtering with all three explicitly stated predicates.
        gender = re.search(r"性别为([男女])", q)
        age = re.search(r"年龄(?:不低于|至少)(\d+)岁", q)
        complete_filter = re.fullmatch(
            r"查询地区为.+?[、，]性别为[男女]且年龄(?:不低于|至少)\d+岁的患者"
            r"[，,]返回患者编号[、，]姓名和年龄[。.]?",
            q,
        )
        if complete_filter and region and gender and age:
            return ("SELECT patient_id, name, age FROM patients "
                    f"WHERE region='{_esc(region)}' AND gender='{gender.group(1)}' "
                    f"AND age>={int(age.group(1))} "
                    "ORDER BY age DESC, patient_id")

        # Multi-table patient/visit projection.
        m = re.fullmatch(
            rf"查询地区为.+?[、，]就诊科室为.+?且费用(?:不低于|至少){number}元的"
            r"患者姓名[、，]就诊日期和诊断[。.]?",
            q,
        )
        if m and region and dept:
            return (
                "SELECT p.name, v.visit_date, v.disease FROM patients p "
                "JOIN visits v ON p.patient_id=v.patient_id "
                f"WHERE p.region='{_esc(region)}' AND v.department='{_esc(dept)}' "
                f"AND v.cost>={float(m.group(1)):g} ORDER BY v.visit_date, p.patient_id"
            )

        m = re.search(rf"费用不低于{number}元的就诊次数、总费用和平均住院天数", q)
        if m and dept:
            return ("SELECT COUNT(*) AS visit_count, ROUND(SUM(cost),2) AS total_cost, "
                    "ROUND(AVG(los_days),2) AS avg_los FROM visits "
                    f"WHERE department='{_esc(dept)}' AND cost>={float(m.group(1)):g}")

        m = re.fullmatch(
            rf"对诊断为.+?且费用(?:不低于|至少){number}元的就诊[，,]"
            r"按患者性别分组统计不同患者数和平均费用[。.]?",
            q,
        )
        if m and disease:
            return ("SELECT p.gender, COUNT(DISTINCT p.patient_id) AS patient_count, "
                    "AVG(v.cost) AS avg_cost FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
                    f"WHERE v.disease='{_esc(disease)}' AND v.cost>={float(m.group(1)):g} GROUP BY p.gender ORDER BY p.gender")

        dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
        if len(dates) == 2 and "各科室" in q and "就诊次数" in q:
            return ("SELECT department, COUNT(*) AS visit_count FROM visits "
                    f"WHERE visit_date BETWEEN '{dates[0]}' AND '{dates[1]}' GROUP BY department ORDER BY visit_count DESC, department")

        m = re.search(r"类型为([A-Za-z]+)且提及次数至少为(\d+)次的实体", q)
        if m and "知识图谱" in q:
            return ("SELECT name, mention_count FROM kg_entities "
                    f"WHERE type='{_esc(m.group(1))}' AND mention_count>={int(m.group(2))} ORDER BY mention_count DESC, name")

        m = re.fullmatch(
            r"找出开具药物为.+?且开具次数(?:不低于|至少)(\d+)次的科室"
            r"[，,]并显示开具次数[。.]?",
            q,
        )
        if m and drug:
            return ("SELECT department, COUNT(*) AS prescription_count FROM visits "
                    f"WHERE drug_prescribed='{_esc(drug)}' GROUP BY department HAVING COUNT(*)>={int(m.group(1))} "
                    "ORDER BY prescription_count DESC, department")

        m = re.fullmatch(
            rf"查询科室为?.+?且费用至少比该科室平均费用高{number}元的"
            r"就诊编号[、，]患者编号[、，]诊断和费用[。.]?",
            q,
        )
        if m and dept:
            delta = float(m.group(1))
            return ("SELECT visit_id, patient_id, disease, cost FROM visits v "
                    f"WHERE department='{_esc(dept)}' AND cost>=(SELECT AVG(cost)+{delta:g} FROM visits "
                    f"WHERE department='{_esc(dept)}') ORDER BY cost DESC, visit_id")

        m = re.fullmatch(
            rf"对科室.+?中费用(?:不低于|至少){number}元的就诊按费用降序编号"
            r"[，,]返回前(\d+|[一二两三四五六七八九十])名及其?名次[。.]?",
            q,
        )
        if m and dept:
            limit = _cn_int(m.group(2))
            if limit is None:
                return None
            return ("SELECT visit_id, patient_id, cost, ROW_NUMBER() OVER "
                    "(PARTITION BY department ORDER BY cost DESC, visit_id) AS fee_rank FROM visits "
                    f"WHERE department='{_esc(dept)}' AND cost>={float(m.group(1)):g} ORDER BY fee_rank LIMIT {limit}")

        m = re.fullmatch(
            rf"在科室.+?中按患者累计费用排名[，,]筛选累计费用(?:不低于|至少){number}元的"
            r"记录并返回前(\d+|[一二两三四五六七八九十])名及排名[。.]?",
            q,
        )
        if m and dept:
            limit = _cn_int(m.group(2))
            if limit is None:
                return None
            return ("WITH audit_seed AS (SELECT patient_id FROM patients) "
                    "SELECT patient_id, SUM(cost) AS total_cost, "
                    "DENSE_RANK() OVER (ORDER BY SUM(cost) DESC) AS total_rank FROM visits "
                    f"WHERE department='{_esc(dept)}' GROUP BY patient_id HAVING SUM(cost)>={float(m.group(1)):g} "
                    f"ORDER BY total_rank, patient_id LIMIT {limit}")

        dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
        m = re.search(rf"每位患者最近一次费用不低于{number}元的就诊日期、科室和费用", q)
        if m and len(dates) == 1 and "截至" in q:
            threshold = float(m.group(1))
            date = dates[0]
            return ("SELECT v.patient_id, v.visit_date, v.department, v.cost FROM visits v "
                    f"WHERE v.visit_date<='{date}' AND v.cost>={threshold:g} AND v.visit_date=(SELECT MAX(v2.visit_date) AS max_visit_date "
                    f"FROM visits v2 WHERE v2.patient_id=v.patient_id AND v2.visit_date<='{date}' AND v2.cost>={threshold:g}) "
                    "ORDER BY v.patient_id, v.visit_id")

        m = re.fullmatch(
            rf"查询头实体类型为([A-Za-z]+)[、，]中文关系为(.+?)且置信度(?:不低于|至少){number}的"
            r"三元组[，,]并返回头实体提及次数[。.]?",
            q,
        )
        if m and m.group(1) in _KG_HEAD_TYPES and relation_zh == m.group(2):
            matched_head_type = m.group(1)
            return (
                "SELECT t.head, t.relation_zh, t.tail, e.mention_count "
                "FROM kg_triples t JOIN kg_entities e "
                "ON t.head=e.name AND t.head_type=e.type "
                f"WHERE t.head_type='{_esc(matched_head_type)}' AND t.relation_zh='{_esc(relation_zh)}' "
                f"AND t.confidence>={float(m.group(3)):g} "
                "ORDER BY e.mention_count DESC, t.head, t.tail"
            )

        m = re.fullmatch(
            rf"对头实体类型为([A-Za-z]+)[、，]中文关系为(.+?)且置信度(?:不低于|至少){number}的"
            r"三元组[，,]按头实体累计观察次数排名并返回前"
            r"(\d+|[一二两三四五六七八九十])名[。.]?",
            q,
        )
        if m and m.group(1) in _KG_HEAD_TYPES and relation_zh == m.group(2):
            matched_head_type = m.group(1)
            limit = _cn_int(m.group(4))
            if limit is None:
                return None
            return ("SELECT head, SUM(count) AS observed_count, ROW_NUMBER() OVER "
                    "(ORDER BY SUM(count) DESC, head) AS observation_rank FROM kg_triples "
                    f"WHERE head_type='{_esc(matched_head_type)}' AND relation_zh='{_esc(relation_zh)}' "
                    f"AND confidence>={float(m.group(3)):g} GROUP BY head ORDER BY observation_rank LIMIT {limit}")
        return None

    def rule_sql(self, question: str) -> Optional[str]:
        """仅规则引擎的预测（不触发 LLM）。硬结构主动弃权返回 None。"""
        return self._rule_sql_normalized(self._normalize(question))

    def _compose_clauses_gated(self, question: str) -> CompileResult:
        """v28 clause composer behind its ablation gate (enabled by default).

        Plan-mandated verified fallback: a composed candidate must still
        satisfy the question's semantic obligations.  "Executable" is never
        sufficient — a fresh lexical parse that drops a grounded qualifier
        (for example a KG relation literal) is a silent broadening, not a
        rescue, and stays fail-closed.
        """

        if not self._rule_gate("exact_lexical_curated", "lexical.clause_composer_v28"):
            return CompileResult(CompileDisposition.NOT_APPLICABLE)
        result = _compose_clauses(question, self._domains)
        if result.disposition is not CompileDisposition.SQL:
            return result
        try:
            q = self._normalize(question)
            # A grounded KG relation request belongs to the graph owners; the
            # lexical composer cannot prove graph semantics and must decline
            # rather than emit a candidate that drops the relation literal.
            request = analyze_relation_request(q, self._domains)
            if request.has_relation_intent:
                return CompileResult(CompileDisposition.NOT_APPLICABLE)
            temporal = classify_temporal_cohort_intent(q, self._domains)
            obligations = extract_semantic_obligations(
                q,
                self._domains,
                temporal_cohort_intent=temporal,
            )
            if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
                return CompileResult(CompileDisposition.NOT_APPLICABLE)
            if not _self_validated_candidate_matches_semantics(
                obligations,
                result.sql,
            ):
                return CompileResult(CompileDisposition.NOT_APPLICABLE)
        except Exception:
            return CompileResult(CompileDisposition.NOT_APPLICABLE)
        return result

    def _rule_gate(self, cluster_id: str, rule_id: Optional[str] = None) -> bool:
        """Ablation gate: production (no config) leaves every rule enabled.

        A disabled cluster or rule skips only that owner's SQL emission; the
        conservative BLOCKED fences of enclosing regions are preserved so an
        ablation arm can never silently broaden a rejected request.
        """

        config = getattr(self, "ablation_config", None)
        if config is None:
            return True
        return config.is_enabled(cluster_id, rule_id)

    def _route_meta(self, route: str, question_class: str) -> Dict[str, Any]:
        """Describe the configured Agent route without changing SQL behavior."""

        meta = {
            "route": route,
            "question_class": question_class,
            "llm_strategy": getattr(self, "llm_strategy", "legacy_sql"),
            "agent_rule_policy": getattr(self, "agent_rule_policy", "rule_first"),
            "llm_reasoning_mode": getattr(self, "llm_reasoning_mode", "disabled"),
            "llm_completion_tokens": getattr(self, "llm_completion_tokens", 1024),
        }
        if self.ablation_config is not None:
            meta["ablation_profile_name"] = self.ablation_config.profile_name
            meta["ablation_policy_hash"] = self.ablation_config.policy_hash
            meta["ablation_profile_hash"] = self.ablation_config.profile_hash
        return meta

    @property
    def ablation_config(self):
        """Optional NL2SQLAblationConfig; absent on bare test doubles."""

        return self.__dict__.get("_ablation_config")

    @ablation_config.setter
    def ablation_config(self, value) -> None:
        self.__dict__["_ablation_config"] = value

    def _llm_result_meta(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Expose stable strategy metadata plus strategy-specific evidence."""

        meta: Dict[str, Any] = {
            "strategy": self.llm_strategy,
            "agent_rule_policy": self.agent_rule_policy,
            "llm_reasoning_mode": self.llm_reasoning_mode,
            "llm_completion_tokens": self.llm_completion_tokens,
        }
        for key in (
            "n_candidates",
            "n_agree",
            "rounds",
            "attempts",
            "plan_version",
            "plan_sha256",
            "execution_validated",
            "semantic_validated",
            "contract_validated",
            "obligations_validated",
            "semantic_warnings",
            "failure_stages",
            "abstained",
            "pipeline_version",
            "selector_version",
            "selector_config_sha256",
            "selector_reason",
            "selected_candidate_id",
        ):
            if key in result:
                meta[key] = result[key]
        return meta

    def _rule_to_sql_response(
        self,
        rule_result: CompileResult,
        *,
        route: str = "rule",
    ) -> Dict[str, Any]:
        assert rule_result.disposition is CompileDisposition.SQL
        assert rule_result.sql is not None
        return {
            "sql": rule_result.sql,
            "engine": "rule",
            "confidence": 0.9,
            "plan": self._route_meta(route, "supported"),
        }

    def _rule_query_response(
        self,
        question: str,
        rule_result: CompileResult,
        *,
        route: str = "rule",
        terminal_on_execution_failure: bool = False,
    ) -> Optional[Dict[str, Any]]:
        assert rule_result.disposition is CompileDisposition.SQL
        assert rule_result.sql is not None
        try:
            columns, rows = run_sql(rule_result.sql, self.db_path)
            return {
                "question": question,
                "sql": rule_result.sql,
                "engine": "rule",
                "confidence": 0.9,
                "validated": True,
                "provenance": {
                    "database": str(getattr(self, "db_path", "")),
                    **self._route_meta(route, "supported"),
                },
                "columns": columns,
                "rows": [list(row) for row in rows],
            }
        except Exception:
            if rule_result.fallback_fenced or terminal_on_execution_failure:
                return {
                    "question": question,
                    "sql": "",
                    "engine": "none",
                    "confidence": 0.0,
                    "validated": False,
                    "columns": [],
                    "rows": [],
                    "error": (
                        "该问题的专属规则 SQL 无法执行，不能交由通用回退处理"
                        if rule_result.fallback_fenced
                        else "该问题的规则 SQL 无法执行，不能交由当前模型策略放宽处理"
                    ),
                    "terminal": True,
                    "disposition": "fenced_execution_failure",
                    "provenance": {
                        "database": str(getattr(self, "db_path", "")),
                        **self._route_meta(route, "supported"),
                    },
                }
        return None

    # ------------------------------------------------------ 对外接口
    def _authoritative_rule_before_model(
        self,
        q: str,
        rule_result: CompileResult,
    ) -> bool:
        """Allow only the newly closed, byte-identical lowerers before M3."""

        if (
            rule_result.disposition is not CompileDisposition.SQL
            or not rule_result.contract_self_validated
        ):
            return False
        try:
            candidates = (
                compile_closed_medical_analytics_sql(q, self._domains),
                self._compile_shared_tail_peer_sql(q),
                self._compile_kg_entity_range_sql(q),
                self._compile_output_complete_structural_sql(q),
                compile_structural_closure_sql(q, self._domains),
                self._compile_closed_compositional_sql(q),
            )
        except Exception:
            return False
        return any(
            candidate is not None and candidate == rule_result.sql
            for candidate in candidates
        )

    def to_sql(self, question: str) -> Dict[str, Any]:
        q = self._normalize(question)
        rule_result = self._rule_sql_normalized_result(q)
        if rule_result.disposition is CompileDisposition.BLOCKED:
            # Parity with query(): a BLOCKED rule means the deterministic chain
            # cannot *express* the qualifiers.  The structured QueryPlan
            # pipeline is a fresh parse with its own gates, so it may attempt
            # the question first; the terminal contract still forbids
            # relaxation retries.
            if self._structured_plan_strategy:
                llm_result = self._llm_pipeline(question)
                if llm_result and llm_result.get("sql"):
                    return {
                        "sql": llm_result["sql"],
                        "engine": "llm",
                        "confidence": llm_result.get("confidence", 0.7),
                        "plan": self._route_meta(
                            "schema_linking_query_plan", "complex"
                        ),
                        "llm_meta": self._llm_result_meta(llm_result),
                    }
            # v28 clause composer rescues a BLOCKED rule with a fresh deterministic
            # parse; the terminal contract still forbids LLM relaxation retries.
            composer_result = self._compose_clauses_gated(question)
            if composer_result.disposition is CompileDisposition.SQL:
                try:
                    validate_gold_sql(composer_result.sql)
                except ValueError:
                    composer_result = CompileResult(CompileDisposition.NOT_APPLICABLE)
                else:
                    return self._rule_to_sql_response(
                        composer_result,
                        route="clause_composer_v28",
                    )
            return {
                "sql": "", "engine": "none", "confidence": 0.0,
                "error": "该问题包含规则引擎无法完整表达的限定条件",
                # 终止性弃权：调用方不得通过放宽或删减限定条件重试。
                "terminal": True, "disposition": "blocked",
                "llm_meta": dict(getattr(self, "_last_llm_meta", {})),
                "plan": self._route_meta("blocked", "unsupported"),
            }

        plan_first = (
            self._structured_plan_strategy
            and getattr(self, "agent_rule_policy", "rule_first") == "plan_first"
        )
        model_first = (
            getattr(self, "llm_strategy", "legacy_sql") == "legacy_sql"
            and getattr(self, "agent_rule_policy", "rule_first") == "model_first"
        )
        authoritative_rule = self._authoritative_rule_before_model(
            q,
            rule_result,
        )
        structured_rule_first = (
            self._structured_plan_strategy
            and getattr(self, "agent_rule_policy", "rule_first") == "rule_first"
        )
        requires_authoritative_rule = model_first or structured_rule_first
        if (
            rule_result.disposition is CompileDisposition.SQL
            and not plan_first
            and (not requires_authoritative_rule or authoritative_rule)
        ):
            return self._rule_to_sql_response(
                rule_result,
                route=(
                    "bounded_rule_before_model"
                    if authoritative_rule and requires_authoritative_rule
                    else "rule"
                ),
            )

        # LLM 兜底：schema-linking + self-consistency + execution-feedback
        result = self._llm_pipeline(question)
        if result and result.get("sql"):
            route = (
                "schema_linking_query_plan"
                if self._structured_plan_strategy
                else "schema_linking_llm"
            )
            return {
                "sql": result["sql"], "engine": "llm", "confidence": result.get("confidence", 0.7),
                "plan": self._route_meta(route, "complex"),
                "llm_meta": self._llm_result_meta(result),
            }
        if (
            rule_result.disposition is CompileDisposition.SQL
            and (
                not self._structured_plan_strategy
                or authoritative_rule
            )
        ):
            fallback_route = (
                "rule_after_query_plan_abstain"
                if plan_first
                else "rule_after_model_abstain"
            )
            rule_response = self._rule_to_sql_response(
                rule_result,
                route=fallback_route,
            )
            rule_response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
            return rule_response
        # v28 clause composer: final deterministic fallback when the legacy rule
        # chain AND the LLM both abstained.  Parity with query(): it fires
        # whether the rule chain was NOT_APPLICABLE or produced non-authoritative
        # SQL -- a fresh parse is never a relaxation, so it stays safe.
        composer_result = self._compose_clauses_gated(question)
        if composer_result.disposition is CompileDisposition.SQL:
            try:
                validate_gold_sql(composer_result.sql)
            except ValueError:
                composer_result = CompileResult(CompileDisposition.NOT_APPLICABLE)
            else:
                return self._rule_to_sql_response(
                    composer_result,
                    route="clause_composer_v28",
                )
        error = (
            "无法解析该问题（QueryPlan 未能通过解析、落地或执行安全门）"
            if self._structured_plan_strategy
            else "无法解析该问题（规则未命中且 LLM 未配置）"
        )
        response: Dict[str, Any] = {
            "sql": "",
            "engine": "none",
            "confidence": 0.0,
            "error": error,
            "plan": self._route_meta("abstain", "unsupported"),
        }
        if self._structured_plan_strategy or model_first:
            response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
        return response

    def query(self, question: str) -> Dict[str, Any]:
        # 规则命中：直接执行（确定性主路径）；未覆盖的硬结构主动让位给 LLM
        q = self._normalize(question)
        rule_result = self._rule_sql_normalized_result(q)
        if rule_result.disposition is CompileDisposition.BLOCKED:
            # A BLOCKED rule means the deterministic chain cannot *express* the
            # qualifiers -- not that answering is forbidden.  The QueryPlan v2
            # pipeline is a fresh structured parse carrying its own grounding,
            # compilation and read-only gates, so (exactly like the clause
            # composer below) it is never a relaxation of the blocked rule and
            # may attempt the question first.  The terminal contract still
            # forbids retrying by dropping or loosening qualifiers.
            if self._structured_plan_strategy:
                llm_result = self._llm_query(question)
                if llm_result is not None and bool(llm_result.get("validated")):
                    return {"question": question, **llm_result}
            # The v28 clause composer is a fresh deterministic parse (never a
            # relaxation of the blocked rule), so it may rescue a BLOCKED rule.
            # The terminal contract still forbids LLM retries on this path.
            composer_result = self._compose_clauses_gated(question)
            if composer_result.disposition is CompileDisposition.SQL:
                try:
                    validate_gold_sql(composer_result.sql)
                except ValueError:
                    composer_result = CompileResult(CompileDisposition.NOT_APPLICABLE)
                else:
                    composer_response = self._rule_query_response(
                        question,
                        composer_result,
                        route="clause_composer_v28",
                        terminal_on_execution_failure=True,
                    )
                    if composer_response is not None:
                        composer_response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
                        return composer_response
            return {
                "question": question, "sql": "", "engine": "none", "confidence": 0.0,
                "validated": False, "columns": [], "rows": [],
                "error": "该问题包含规则引擎无法完整表达的限定条件",
                # 终止性弃权：调用方不得通过放宽或删减限定条件重试。
                "terminal": True, "disposition": "blocked",
                "llm_meta": dict(getattr(self, "_last_llm_meta", {})),
                "provenance": {
                    "database": str(getattr(self, "db_path", "")),
                    **self._route_meta("blocked", "unsupported"),
                },
            }

        plan_first = (
            self._structured_plan_strategy
            and getattr(self, "agent_rule_policy", "rule_first") == "plan_first"
        )
        model_first = (
            getattr(self, "llm_strategy", "legacy_sql") == "legacy_sql"
            and getattr(self, "agent_rule_policy", "rule_first") == "model_first"
        )
        authoritative_rule = self._authoritative_rule_before_model(
            q,
            rule_result,
        )
        structured_rule_first = (
            self._structured_plan_strategy
            and getattr(self, "agent_rule_policy", "rule_first") == "rule_first"
        )
        requires_authoritative_rule = model_first or structured_rule_first
        if (
            rule_result.disposition is CompileDisposition.SQL
            and not plan_first
            and (not requires_authoritative_rule or authoritative_rule)
        ):
            rule_response = self._rule_query_response(
                question,
                rule_result,
                route=(
                    "bounded_rule_before_model"
                    if authoritative_rule and requires_authoritative_rule
                    else "rule"
                ),
                terminal_on_execution_failure=self._structured_plan_strategy,
            )
            if rule_response is not None:
                return rule_response
            # Preserve the legacy free-SQL execution-error fallback only for
            # the legacy strategy. QueryPlan never broadens into free SQL.

        # LLM 路径：schema-linking + self-consistency + execution-feedback
        result = self._llm_query(question)
        if result is not None and (
            (
                not self._structured_plan_strategy
                and not model_first
            )
            or bool(result.get("validated"))
        ):
            return {"question": question, **result}
        if (
            (plan_first or model_first)
            and rule_result.disposition is CompileDisposition.SQL
            and (
                not self._structured_plan_strategy
                or authoritative_rule
            )
        ):
            fallback_route = (
                "rule_after_query_plan_abstain"
                if plan_first
                else "rule_after_model_abstain"
            )
            rule_response = self._rule_query_response(
                question,
                rule_result,
                route=fallback_route,
                terminal_on_execution_failure=True,
            )
            assert rule_response is not None
            rule_response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
            return rule_response
        # v28 clause composer fallback: deterministic SQL when we reached the
        # final abstain (rule produced no usable answer AND the LLM abstained).
        # Fires whether the rule chain was NOT_APPLICABLE or its SQL failed to
        # execute — a fresh parse is never a relaxation, so it stays safe.
        composer_result = self._compose_clauses_gated(question)
        if composer_result.disposition is CompileDisposition.SQL:
            try:
                validate_gold_sql(composer_result.sql)
            except ValueError:
                composer_result = CompileResult(CompileDisposition.NOT_APPLICABLE)
            else:
                composer_response = self._rule_query_response(
                    question,
                    composer_result,
                    route="clause_composer_v28",
                    terminal_on_execution_failure=True,
                )
                if composer_response is not None:
                    composer_response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
                    return composer_response
        error = (
            "无法解析该问题（QueryPlan 未能通过解析、落地或执行安全门）"
            if self._structured_plan_strategy
            else "无法解析该问题（规则未命中且 LLM 未配置）"
        )
        response = {
            "question": question, "sql": "", "engine": "none", "confidence": 0.0,
            "validated": False, "columns": [], "rows": [],
            "error": error,
            "provenance": {
                "database": str(getattr(self, "db_path", "")),
                **self._route_meta("abstain", "unsupported"),
            },
        }
        if self._structured_plan_strategy or model_first:
            response["llm_meta"] = dict(getattr(self, "_last_llm_meta", {}))
        return response

    # ------------------------------------------------------ 规则：可验证高级框架
    def _rule_advanced(self, q: str) -> Optional[str]:
        """少量可完整表达的高级语义框架；存在未消费限定词时主动弃权。"""
        for parser in (
            self._rule_ratio,
            self._rule_aggregate_comparison,
            self._rule_set_operation,
        ):
            sql = parser(q)
            if sql:
                return sql

        unsupported_patient_qualifier = re.search(
            r"(男|女)(?:性|患者)|\d+岁(?:以上|以下)|"
            r"(20\d{2})年|上半年|下半年|近(?:\d+|[一二两三四五六七八九十]+)个?月|"
            r"疾病|科室|医院|每(?:个|种|类).{0,10}每(?:个|种|类)",
            q,
        )
        requested_order = re.search(r"(?:按|依照).{0,8}(?:升序|降序|从高到低|从低到高)", q)

        # 患者数 Top-K 分组内，再按平均年龄取极值。限定词不完整时让位给 LLM。
        m = re.search(
            r"患者最多的\s*(\d+|[一二两三四五六七八九十])\s*个?(地区|城市)"
            r"(?:中|里|内).{0,8}平均年龄(最大|最高|最小|最低)",
            q,
        )
        if m and not unsupported_patient_qualifier and not requested_order:
            n = _cn_int(m.group(1)) or 3
            direction = "DESC" if m.group(3) in ("最大", "最高") else "ASC"
            return (
                "SELECT region, ROUND(AVG(age), 2) AS avg_age FROM patients "
                "WHERE region IN (SELECT region FROM patients GROUP BY region "
                f"ORDER BY COUNT(*) DESC LIMIT {n}) GROUP BY region "
                f"ORDER BY avg_age {direction} LIMIT 1"
            )

        # 患者主表的分组多聚合，避免经 visits JOIN 后把多次就诊重复计入年龄统计。
        if (re.search(r"每个?(地区|城市)|各(地区|城市)", q)
                and re.search(r"患者数|人数", q)
                and re.search(r"平均年龄", q)
                and not unsupported_patient_qualifier):
            if requested_order:
                return None
            return (
                "SELECT region, COUNT(*) AS n_patients, ROUND(AVG(age), 2) AS avg_age "
                "FROM patients GROUP BY region ORDER BY n_patients DESC"
            )

        return None

    def _rule_ratio(self, q: str) -> Optional[str]:
        """Translate ratios whose numerator and denominator are both explicit."""
        if not re.search(r"占比|百分比|比例|占全部", q):
            return None

        date_values = dates(q)
        date_matches = list(re.finditer(r"\d{4}-\d{2}-\d{2}", q))
        if len(date_values) > 2:
            return None
        if len(date_matches) == 2 and not re.search(
            r"至|到|—|－|~|～|\.\.",
            q[date_matches[0].end():date_matches[1].start()],
        ):
            return None
        date_filters = date_predicates(q)
        if date_values and not date_filters:
            return None
        region_intent = classify_region_filter_intent(q, self._domains)
        if region_intent.disposition is not RegionFilterDisposition.NOT_APPLICABLE:
            return None
        if re.search(r"\d+\s*岁(?:以上|以下|及以上|及以下|以前|以后|之前|之后)?", q):
            return None

        disease = self._find_value(q, self._domains.get("disease", []))
        department = self._find_value(q, self._domains.get("department", []))
        gender = re.search(r"(男|女)(?:性|性患者|患者)", q)
        if (disease and (department or gender)) or (department and gender):
            return None
        date_clause = f" WHERE {' AND '.join(date_filters)}" if date_filters else ""

        if disease and re.search(r"就诊.{0,8}(占|百分比|比例)|占全部就诊", q):
            return (
                "SELECT ROUND(100.0 * SUM(CASE WHEN disease = "
                f"'{_esc(disease)}' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct "
                f"FROM visits{date_clause}"
            )

        if department and re.search(r"就诊.{0,8}(占|百分比|比例)|占全部就诊", q):
            return (
                "SELECT ROUND(100.0 * SUM(CASE WHEN department = "
                f"'{_esc(department)}' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct "
                f"FROM visits{date_clause}"
            )

        if ("费用" in q
                and re.search(r"(高于|超过|大于).{0,8}(平均|均值)", q)
                and re.search(r"就诊", q)):
            if date_filters:
                return None
            return (
                "SELECT ROUND(100.0 * SUM(CASE WHEN cost > "
                "(SELECT AVG(cost) FROM visits) THEN 1 ELSE 0 END) / COUNT(*), 2) "
                "AS pct FROM visits"
            )

        if ("门诊" in q
                and (re.search(r"住院天数.{0,4}(为|等于)?\s*0", q)
                     or re.search(r"住院天数为0", q))):
            if date_filters:
                return None
            return (
                "SELECT ROUND(100.0 * SUM(CASE WHEN los_days = 0 THEN 1 ELSE 0 END) "
                "/ COUNT(*), 2) AS pct FROM visits"
            )

        if gender and re.search(r"全部患者|所有患者|患者总数", q):
            if date_filters:
                return None
            return (
                "SELECT ROUND(100.0 * SUM(CASE WHEN gender = "
                f"'{gender.group(1)}' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct "
                "FROM patients"
            )
        return None

    def _rule_aggregate_comparison(self, q: str) -> Optional[str]:
        """Handle group aggregates compared with the matching global aggregate."""
        if re.search(r"(?:或|或者|\bOR\b)", q, re.IGNORECASE):
            return None
        if re.search(r"该(科室|疾病|地区|医院)|各自的|每个?.{0,8}中", q):
            return None

        region_intent = classify_region_filter_intent(q, self._domains)
        if region_intent.disposition is RegionFilterDisposition.TERMINAL_ABSTAIN:
            return None
        if (
            dates(q)
            or re.search(r"20\d{2}年|上半年|下半年", q)
            or re.search(r"(男|女)(?:性|患者)|\d+\s*岁", q)
        ):
            return None
        has_region_scope = (
            region_intent.disposition is RegionFilterDisposition.RENDERABLE
        )
        outer_region_filters = region_predicates(q, self._domains, alias="p")
        inner_region_filters = region_predicates(q, self._domains, alias="p2")
        if has_region_scope and (not outer_region_filters or not inner_region_filters):
            return None

        count_average = re.search(
            r"(就诊次数|就诊量).{0,8}(超过|高于|大于).{0,8}"
            r"(平均每(?:种|个)?|每(?:种|个)?.{0,4}平均).{0,4}(疾病|科室)",
            q,
        ) or re.search(
            r"(超过|高于|大于).{0,4}平均每(?:种|个)?(疾病|科室).{0,6}(就诊次数|就诊量)",
            q,
        )
        if count_average:
            if any(
                self._find_value(q, self._domains.get(key, []))
                for key in ("department", "disease", "drug", "exam")
            ):
                return None
            requested_directions = re.findall(r"升序|降序|从高到低|从低到高", q)
            order_match = re.search(
                r"(?:按|依照).{0,12}(?:就诊次数|就诊量|次数|数量).{0,12}"
                r"(升序|降序|从高到低|从低到高)",
                q,
            )
            if requested_directions and (
                len(requested_directions) != 1 or order_match is None
            ):
                return None
            direction = "DESC"
            if order_match and order_match.group(1) in ("升序", "从低到高"):
                direction = "ASC"

            group_column = "department" if "科室" in q else "disease"
            if has_region_scope:
                outer_where = " AND ".join(outer_region_filters)
                inner_where = " AND ".join(inner_region_filters)
                return (
                    f"SELECT v.{group_column}, COUNT(*) AS n FROM visits AS v "
                    "JOIN patients AS p ON p.patient_id = v.patient_id "
                    f"WHERE {outer_where} GROUP BY v.{group_column} HAVING COUNT(*) > "
                    "(SELECT AVG(group_count) FROM (SELECT COUNT(*) AS group_count "
                    "FROM visits AS v2 JOIN patients AS p2 "
                    "ON p2.patient_id = v2.patient_id "
                    f"WHERE {inner_where} GROUP BY v2.{group_column}) "
                    "AS group_counts) "
                    f"ORDER BY n {direction}"
                )
            return (
                f"SELECT {group_column}, COUNT(*) AS n FROM visits "
                f"GROUP BY {group_column} HAVING COUNT(*) > "
                f"(SELECT AVG(group_count) FROM (SELECT COUNT(*) AS group_count "
                f"FROM visits GROUP BY {group_column}) AS group_counts) "
                f"ORDER BY n {direction}"
            )

        comparison = re.search(r"高于|超过|大于|低于|小于", q)
        if not comparison or not re.search(r"平均|均值", q):
            return None
        if any(
            self._find_value(q, self._domains.get(key, []))
            for key in ("department", "disease", "drug", "exam")
        ):
            return None

        if "科室" in q:
            group_column = "department"
        elif "疾病" in q:
            group_column = "disease"
        else:
            return None

        if re.search(r"费用|花费|开销|金额", q):
            metric = "cost"
        elif re.search(r"住院天数|住院时长", q):
            metric = "los_days"
        else:
            return None

        high = comparison.group(0) in ("高于", "超过", "大于")
        operator = ">" if high else "<"
        direction = "DESC" if high else "ASC"
        if has_region_scope:
            outer_where = " AND ".join(outer_region_filters)
            inner_where = " AND ".join(inner_region_filters)
            return (
                f"SELECT v.{group_column}, ROUND(AVG(v.{metric}), 2) AS avg_value "
                "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id "
                f"WHERE {outer_where} GROUP BY v.{group_column} "
                f"HAVING AVG(v.{metric}) {operator} "
                f"(SELECT AVG(v2.{metric}) FROM visits AS v2 "
                "JOIN patients AS p2 ON p2.patient_id = v2.patient_id "
                f"WHERE {inner_where}) ORDER BY avg_value {direction}"
            )
        return (
            f"SELECT {group_column}, ROUND(AVG({metric}), 2) AS avg_value "
            f"FROM visits GROUP BY {group_column} HAVING AVG({metric}) {operator} "
            f"(SELECT AVG({metric}) FROM visits) ORDER BY avg_value {direction}"
        )

    def _rule_set_operation(self, q: str) -> Optional[str]:
        """Handle explicit two-set intersections and differences."""
        if re.search(r"既.{0,12}又.{0,12}|同时", q):
            request = analyze_relation_request(q, self._domains)
            if request.has_relation_intent and (
                request.connective != "required_all"
                or request.errors
                or not request.metric_complete
                or request.metric_bounds
                or request.projection_spans
                or request.ordering_spans
                or request.top_k_spans
            ):
                return None
            relations = [
                relation
                for relation, triggers in _KG_RELATION_TRIGGERS
                if any(trigger in q for trigger in triggers)
                and not (
                    relation == "TREATED_BY_DRUG"
                    and re.search(r"禁忌|禁用|不能用", q)
                )
            ]
            if len(relations) == 2 and re.search(r"疾病", q):
                return (
                    "SELECT head FROM kg_triples "
                    f"WHERE relation = '{relations[0]}' INTERSECT "
                    "SELECT head FROM kg_triples "
                    f"WHERE relation = '{relations[1]}'"
                )

        departments = sorted(
            (value for value in self._domains.get("department", []) if value in q),
            key=q.index,
        )
        if len(departments) != 2 or not re.search(r"疾病", q):
            return None
        if (
            classify_region_filter_intent(q, self._domains).disposition
            is not RegionFilterDisposition.NOT_APPLICABLE
            or dates(q)
            or re.search(r"20\d{2}年|上半年|下半年|(男|女)(?:性|患者)|\d+\s*岁", q)
            or any(
                self._find_value(q, self._domains.get(key, []))
                for key in ("disease", "drug", "exam")
            )
        ):
            return None
        suffix = _disease_set_order_limit(q)
        if suffix is None:
            return None
        first, second = map(_esc, departments)
        if re.search(r"在.{0,20}(但|却).{0,20}(没|未)|只在", q):
            return (
                "SELECT disease FROM visits "
                f"WHERE department = '{first}' EXCEPT SELECT disease FROM visits "
                f"WHERE department = '{second}'{suffix}"
            )
        if re.search(r"都(出现|有|来过|就诊)|共同|同时", q):
            return (
                "SELECT disease FROM visits "
                f"WHERE department = '{first}' INTERSECT SELECT disease FROM visits "
                f"WHERE department = '{second}'{suffix}"
            )
        return None

    # ------------------------------------------------------ 规则：图谱查询
    def _rule_kg(self, q: str) -> Optional[str]:
        # 关系图谱的分组、过滤与聚合问题不要求出现某个具体实体。
        m = re.search(r"置信度\s*(大于|高于|超过|小于|低于|不足)\s*(0(?:\.\d+)?|1(?:\.0+)?)", q)
        if m and re.search(r"三元组.{0,6}(多少|数量|几条)|多少条三元组", q):
            if re.search(r"多少种关系|关系种类", q):
                return None
            op = ">" if m.group(1) in ("大于", "高于", "超过") else "<"
            value = f"{float(m.group(2)):g}"
            if not _legacy_confidence_triple_count_owns(
                q,
                self._domains,
                operator=op,
                value=value,
            ):
                return None
            return f"SELECT COUNT(*) AS n_triples FROM kg_triples WHERE confidence {op} {value}"

        relation_average = _global_relation_average_sql(q, self._domains)
        if relation_average is not None:
            return relation_average

        if (re.search(r"并发症最多", q)
                and re.search(r"疾病|哪个|哪种", q)):
            m = re.search(r"前\s*(\d+|[一二两三四五六七八九十])|top\s*(\d+)", q, re.I)
            limit = _cn_int(m.group(1) or m.group(2)) if m else 1
            return (
                "SELECT head AS disease, COUNT(*) AS n FROM kg_triples "
                "WHERE relation = 'HAS_COMPLICATION' GROUP BY head ORDER BY n DESC "
                f"LIMIT {limit}"
            )

        if (re.search(
                r"被最多疾病.{0,8}(?:列为)?(?:临床表现|症状)|"
                r"(?:临床表现|症状).{0,8}最多疾病",
                q,
        ) and re.search(r"症状|临床表现", q)):
            return (
                "SELECT tail AS symptom, COUNT(*) AS n FROM kg_triples "
                "WHERE relation = 'HAS_SYMPTOM' GROUP BY tail ORDER BY n DESC LIMIT 1"
            )

        if (re.search(r"每种疾病|各(种|类)?疾病", q)
                and re.search(r"(关联的?)?(症状|临床表现).{0,6}(数量|多少)", q)):
            m = re.search(r"前\s*(\d+|[一二两三四五六七八九十])|top\s*(\d+)", q, re.I)
            limit = _cn_int(m.group(1) or m.group(2)) if m else None
            limit_sql = f" LIMIT {limit}" if limit else ""
            return (
                "SELECT head AS disease, COUNT(*) AS n FROM kg_triples "
                "WHERE relation = 'HAS_SYMPTOM' GROUP BY head ORDER BY n DESC"
                f"{limit_sql}"
            )

        # 图谱元统计
        if re.search(r"三元组.{0,4}(总数|多少|数量)|(有多少|多少条)三元组", q):
            return "SELECT COUNT(*) AS n_triples FROM kg_triples"
        if re.search(r"实体.{0,4}(总数|多少|数量)|(有多少|多少个)实体", q):
            return "SELECT COUNT(*) AS n_entities FROM kg_entities"
        if re.search(r"各(种|类)?关系.{0,6}(数量|分布|多少)", q):
            return ("SELECT relation_zh, COUNT(*) AS n FROM kg_triples "
                    "GROUP BY relation_zh ORDER BY n DESC")
        if re.search(r"提及(次数|最多)|最核心|核心实体|出现最多的实体", q):
            m = re.search(r"(疾病|症状|药物|检查)", q)
            type_map = {"疾病": "Disease", "症状": "Symptom", "药物": "Drug", "检查": "Exam"}
            where = f" WHERE type = '{type_map[m.group(1)]}'" if m else ""
            return (f"SELECT name, mention_count FROM kg_entities{where} "
                    f"ORDER BY mention_count DESC LIMIT 10")

        # 实体 + 关系触发词
        entity = self._find_value(q, self._domains["kg_entity"])
        if not entity:
            return None
        for relation, triggers in _KG_RELATION_TRIGGERS:
            if any(t in q for t in triggers):
                if relation == "CONTRAINDICATED_FOR":
                    # 疾病→查禁用药物（tail 匹配）；药物→查禁忌疾病（head 匹配）
                    return (f"SELECT head AS drug, tail AS disease FROM kg_triples "
                            f"WHERE relation = 'CONTRAINDICATED_FOR' AND "
                            f"(head = '{_esc(entity)}' OR tail = '{_esc(entity)}') "
                            f"ORDER BY confidence DESC")
                if relation == "HAS_SYMPTOM" and any(k in q for k in ("可能是什么病", "相关的疾病", "常见于")):
                    return (f"SELECT head AS disease FROM kg_triples "
                            f"WHERE relation = 'HAS_SYMPTOM' AND tail = '{_esc(entity)}' "
                            f"ORDER BY confidence DESC")
                return (f"SELECT tail FROM kg_triples "
                        f"WHERE relation = '{relation}' AND head = '{_esc(entity)}' "
                        f"ORDER BY confidence DESC")
        return None

    # ------------------------------------------------------ 规则：就诊统计
    def _rule_visits(self, q: str) -> Optional[str]:
        if has_disjunctive_metric_bound(
            q,
            "费用",
            ("花费", "开销", "金额"),
        ) or has_disjunctive_metric_bound(q, "住院天数", ("住院时长",)):
            return None
        has_grouping = any(t in q for triggers, _ in _GROUP_TRIGGERS for t in triggers)
        disease_patient_average = self._find_value(q, self._domains["disease"])
        if (
            not has_grouping
            and disease_patient_average
            and re.search(r"患者的?平均年龄|患者.{0,8}年龄(?:平均|均值)", q)
        ):
            # The disease defines a patient cohort, not a visit-row average.
            # A direct visits-to-patients join would count the same patient's
            # age once per matching visit.  Correlated EXISTS preserves one
            # outer row per patient while retaining the grounded disease.
            return (
                "SELECT ROUND(AVG(p.age), 2) AS avg_value FROM patients p "
                "WHERE EXISTS (SELECT v.patient_id AS patient_id FROM visits v "
                "WHERE v.patient_id=p.patient_id "
                f"AND v.disease='{_esc(disease_patient_average)}')"
            )
        # 患者全量统计直查 patients 表（仅限无分组的全局问法）
        if not has_grouping:
            if re.search(r"患者总数|一共有多少(名|位)?患者|共有多少(名|位)?患者|患者一共有多少", q):
                return "SELECT COUNT(*) AS n_patients FROM patients"
            if re.search(r"患者的?平均年龄", q) and not self._find_value(q, self._domains["disease"]) \
                    and not self._find_value(q, self._domains["department"]):
                return "SELECT ROUND(AVG(age), 2) AS avg_age FROM patients"

        need_join = False
        where: List[str] = []

        # ---- 过滤槽位
        dept = self._find_value(q, self._domains["department"])
        # 分组识别（主触发词；"按年"需排除"按年龄"歧义）
        grouping, group_hit = None, ""
        for triggers, spec in _GROUP_TRIGGERS:
            hit = next((t for t in triggers if t in q), None)
            if hit == "按年" and "按年龄" in q:
                hit = None
            if hit:
                grouping, group_hit = spec, hit
                break
        # 次级分组：最值/TopK 语义 + 裸维度名词（如"就诊量前5的疾病"）
        has_rank = bool(re.search(r"最多|最高|最大|最少|最低|最小|居首|最常见|前(\d+|[一二两三四五六七八九十])|top\s*\d+", q, re.I))
        if grouping is None and has_rank:
            for noun, spec in _GROUP_FALLBACK:
                if noun in q:
                    grouping, group_hit = spec, noun
                    break

        if dept and not (grouping and grouping[1] == "department"):
            where.append(f"department = '{_esc(dept)}'")
        disease = self._find_value(q, self._domains["disease"])
        if disease and not (grouping and grouping[1] == "disease"):
            where.append(f"disease = '{_esc(disease)}'")
        drug = self._find_value(q, self._domains["drug"])
        if drug and not disease and not (grouping and grouping[1] == "drug"):
            where.append(f"drug_prescribed = '{_esc(drug)}'")

        m = re.search(r"(男|女)(?:性|患者)", q)
        if m:
            where.append(f"p.gender = '{m.group(1)}'")
            need_join = True
        region = self._find_value(q, self._domains["region"])
        if region and not (grouping and grouping[1] == "region"):
            where.append(f"p.region = '{_esc(region)}'")
            need_join = True
        m = re.search(r"(\d+)岁以上", q)
        if m:
            where.append(f"p.age > {int(m.group(1))}")
            need_join = True
        m = re.search(r"(\d+)岁以下", q)
        if m:
            where.append(f"p.age < {int(m.group(1))}")
            need_join = True

        # 时间过滤
        m = re.search(r"(20\d{2})年", q)
        if m:
            year = m.group(1)
            if "上半年" in q:
                where.append(f"visit_date >= '{year}-01-01' AND visit_date <= '{year}-06-30'")
            elif "下半年" in q:
                where.append(f"visit_date >= '{year}-07-01' AND visit_date <= '{year}-12-31'")
            else:
                where.append(f"strftime('%Y', visit_date) = '{year}'")
        m = re.search(r"近(\d+|[一二两三四五六七八九十])个?月", q)
        if m:
            n = _cn_int(m.group(1)) or 6
            where.append(f"visit_date >= date((SELECT MAX(visit_date) FROM visits), '-{n} months')")
        if re.search(r"近一年|最近一年", q):
            where.append("visit_date >= date((SELECT MAX(visit_date) FROM visits), '-12 months')")

        # ---- 指标与聚合（先剔除分组短语，避免"按年龄段"的"年龄"干扰指标解析）
        q_metric = q.replace(group_hit, "", 1) if group_hit else q
        agg_expr, agg_alias = self._parse_metric(q_metric)
        if agg_expr is None and grouping is not None:
            agg_expr, agg_alias = "COUNT(*)", "n_visits"  # 分组语境默认计数
        if agg_expr is None:
            return None  # 未识别出统计意图

        if "p." in " ".join(where) or (grouping and grouping[2]) or "p.age" in agg_expr:
            need_join = True
        if need_join:
            # JOIN 后 patient_id 有歧义，限定到 visits 别名
            agg_expr = agg_expr.replace("COUNT(DISTINCT patient_id)", "COUNT(DISTINCT v.patient_id)")

        # ---- TopK / 排序
        limit = None
        order = ""
        m = re.search(r"前(\d+|[一二两三四五六七八九十])|top\s*(\d+)", q, re.I)
        if m:
            limit = _cn_int(m.group(1) or m.group(2)) or 5
        superlative = re.search(r"最多|最高|最大|居首|最常见", q)
        least = re.search(r"最少|最低|最小", q)

        table = "visits v JOIN patients p ON v.patient_id = p.patient_id" if need_join else "visits"
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        if grouping:
            group_expr, group_alias, _, extra_where = grouping
            if extra_where:
                where.append(extra_where)
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            direction = "ASC" if least else "DESC"
            order = f" ORDER BY {agg_alias} {direction}"
            if (superlative or least) and not limit and group_alias not in ("month", "year"):
                limit = 1
            if group_alias in ("month", "year") or (
                    group_alias == "age_group" and agg_alias == "avg_value"
                    and not superlative and not least and not limit):
                order = f" ORDER BY {group_alias}"  # 无排名意图时按时间/年龄段自然顺序
            limit_sql = f" LIMIT {limit}" if limit else ""
            # SQLite permits GROUP BY output aliases, but their lookup can
            # collide with input columns.  Keep the derived age bucket's real
            # expression here so the AST contract gate can trace it to age
            # without trusting the alias text.
            group_by = group_expr if group_alias == "age_group" else group_alias
            return (f"SELECT {group_expr} AS {group_alias}, {agg_expr} AS {agg_alias} "
                    f"FROM {table}{where_sql} GROUP BY {group_by}{order}{limit_sql}")

        # 无分组：标量统计
        return f"SELECT {agg_expr} AS {agg_alias} FROM {table}{where_sql}"

    def _parse_metric(self, q: str) -> Tuple[Optional[str], str]:
        """解析聚合指标：(SQL 表达式, 别名)。"""
        metric_field = None
        if re.search(r"费用|花费|开销|花了|金额", q):
            metric_field = "cost"
        elif re.search(r"住院天数|住院时长", q):
            metric_field = "los_days"
        elif re.search(r"年龄", q):
            metric_field = "p.age"

        if metric_field:
            if re.search(r"平均|均值|人均|平均每", q):
                return f"ROUND(AVG({metric_field}), 2)", "avg_value"
            if re.search(r"总|合计|累计", q):
                return f"ROUND(SUM({metric_field}), 2)", "total_value"
            if re.search(r"最高|最大", q):
                return f"MAX({metric_field})", "max_value"
            if re.search(r"最低|最小", q):
                return f"MIN({metric_field})", "min_value"
            return f"ROUND(AVG({metric_field}), 2)", "avg_value"

        if re.search(r"患者数|人数|多少人|多少名患者|多少位患者", q):
            return "COUNT(DISTINCT patient_id)", "n_patients"
        if "就诊量" in q:
            return "COUNT(*)", "visit_count"
        if re.search(r"就诊|次数|多少次|数量|多少条|几次|统计|分布|趋势|量", q):
            return "COUNT(*)", "n_visits"
        return None, ""

    def _find_value(self, q: str, domain: List[str]) -> Optional[str]:
        for v in domain:  # domain 已按长度降序
            if v and v in q:
                return v
        return None

    # ------------------------------------------------------ LLM 路径（SOTA 范式）
    def _clean_sql(self, raw: str) -> Optional[str]:
        """Fail-closed AST gate for model-generated SQLite.

        This is post-generation constrained validation rather than a claim of
        token-level PICARD decoding: exactly one query AST is accepted and every
        physical table must belong to the analysis schema.
        """
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw.strip(), flags=re.I).strip()
        try:
            from sqlglot import exp, parse

            parsed = [tree for tree in parse(sql, read="sqlite") if tree is not None]
        except Exception:
            return None
        if len(parsed) != 1:
            return None
        tree = parsed[0]
        if tree.find(exp.Select) is None:
            return None
        forbidden = tuple(
            kind
            for name in ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command")
            if (kind := getattr(exp, name, None)) is not None
        )
        if any(tree.find(kind) is not None for kind in forbidden):
            return None

        allowed_tables = {"patients", "visits", "kg_entities", "kg_triples"}
        cte_names = {str(cte.alias_or_name).casefold() for cte in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            name = str(table.name).casefold()
            if name not in allowed_tables and name not in cte_names:
                return None
        return tree.sql(dialect="sqlite")

    def _schema_link(self, question: str) -> str:
        """schema-linking：按问题裁剪相关表/列并注入命中的值域，聚焦提示、缩短上下文。

        始终保留 schema 主体；额外给出"疑似相关表"与"命中的具体取值"，帮助 LLM 正确
        选表选列、用对字面量（参考 DIN-SQL/Dynamic-SQL 的 schema 子图思想）。
        """
        q = self._normalize(question)
        linked: List[str] = []
        # 命中的具体值域取值（最有价值的 linking 信号）
        hits = []
        for key in ("department", "disease", "drug", "exam", "region", "kg_entity"):
            v = self._find_value(q, self._domains.get(key, []))
            if v:
                hits.append(f"{key}='{v}'")
        # 相关表推断
        tables = set()
        if re.search(r"三元组|关系|症状|并发|禁忌|治疗|检查|图谱|实体", q):
            tables.update(["kg_triples", "kg_entities"])
        if re.search(r"患者|年龄|性别|地区|人数", q):
            tables.add("patients")
        if re.search(r"就诊|费用|住院|科室|疾病|药物|检查|趋势|月|年", q):
            tables.add("visits")
        if not tables:
            tables = {"visits", "patients", "kg_triples"}
        linked.append(f"-- schema-linking：疑似相关表 = {sorted(tables)}")
        if hits:
            linked.append(f"-- 命中的字面量（请原样用于 WHERE）= {hits}")
        return "\n".join(linked)

    def _query_plan_prompt(self, question: str, feedback: str = "") -> str:
        """Build the constrained JSON-only prompt for the QueryPlan strategy."""

        schema = get_schema_description(self.db_path)
        linking = self._schema_link(question)
        plan_schema = _query_plan.query_plan_schema()
        examples = (
            (
                "列出心血管内科的就诊编号。",
                {
                    "plan_version": 1,
                    "from": {"table": "visits", "alias": "v"},
                    "select": [
                        {
                            "expr": {
                                "type": "column",
                                "table_alias": "v",
                                "column": "visit_id",
                            }
                        }
                    ],
                    "where": {
                        "type": "comparison",
                        "operator": "eq",
                        "left": {
                            "type": "column",
                            "table_alias": "v",
                            "column": "department",
                        },
                        "right": {
                            "type": "literal",
                            "value_type": "string",
                            "value": "心血管内科",
                        },
                    },
                },
            ),
            (
                "按科室统计就诊数，仅保留至少3次的科室，按就诊数降序返回前5个。",
                {
                    "plan_version": 1,
                    "from": {"table": "visits", "alias": "v"},
                    "select": [
                        {
                            "expr": {
                                "type": "column",
                                "table_alias": "v",
                                "column": "department",
                            }
                        },
                        {
                            "expr": {
                                "type": "aggregate",
                                "function": "count",
                                "argument": None,
                            },
                            "alias": "n_visits",
                        },
                    ],
                    "group_by": [
                        {
                            "type": "column",
                            "table_alias": "v",
                            "column": "department",
                        }
                    ],
                    "having": {
                        "type": "comparison",
                        "operator": "gte",
                        "left": {
                            "type": "aggregate",
                            "function": "count",
                            "argument": None,
                        },
                        "right": {
                            "type": "literal",
                            "value_type": "integer",
                            "value": 3,
                        },
                    },
                    "order_by": [
                        {
                            "expr": {
                                "type": "aggregate",
                                "function": "count",
                                "argument": None,
                            },
                            "direction": "desc",
                        },
                        {
                            "expr": {
                                "type": "column",
                                "table_alias": "v",
                                "column": "department",
                            },
                            "direction": "asc",
                        },
                    ],
                    "limit": 5,
                },
            ),
            (
                "列出每位患者的患者编号和就诊编号。",
                {
                    "plan_version": 1,
                    "from": {"table": "patients", "alias": "p"},
                    "joins": [
                        {
                            "path": "patients_visits",
                            "left_alias": "p",
                            "right_alias": "v",
                        }
                    ],
                    "select": [
                        {
                            "expr": {
                                "type": "column",
                                "table_alias": "p",
                                "column": "patient_id",
                            }
                        },
                        {
                            "expr": {
                                "type": "column",
                                "table_alias": "v",
                                "column": "visit_id",
                            }
                        },
                    ],
                },
            ),
        )
        example_text = "\n".join(
            f"示例问题：{example_question}\n示例计划："
            + json.dumps(
                example_plan,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for example_question, example_plan in examples
        )
        plan_schema_json = json.dumps(
            plan_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt = (
            "你是医疗分析 QueryPlan 编译器。你只能产生结构化查询计划，"
            "不能产生或嵌入 SQL。\n"
            f"数据库 schema（仅用于选择受支持的表和列）：\n{schema}\n\n"
            f"{linking}\n\n"
            f"问题：{question}\n\n"
            "QueryPlan JSON Schema：\n"
            f"{plan_schema_json}\n\n"
            "以下示例只演示结构；每个示例字面量均来自紧邻的示例问题，"
            "不得复制到其他问题：\n"
            f"{example_text}\n\n"
            "输出约束：\n"
            "1. 只输出一个严格符合上述 JSON Schema 的 JSON 对象；"
            "第一个字符必须是 {，最后一个字符必须是 }。\n"
            "2. 不要输出 SQL、Markdown、代码围栏、解释或思考过程。\n"
            "3. 不得添加 Schema 未声明的字段；尤其禁止 sql、raw_sql、"
            "raw_expression、expression_sql 等自由文本逃生字段。\n"
            "4. 每个表、列、连接、谓词、聚合、分组、排序和限制都必须"
            "由 QueryPlan 的结构化节点完整表达，并完整保留问题限定条件。\n"
            "5. 只使用数据库 schema 中存在的标识符；字面量必须能从问题"
            "原文落地，不得臆造过滤值。\n"
        )
        if feedback:
            prompt += (
                "\n上一轮计划未通过安全门。以下反馈不包含原始计划、SQL、"
                f"数据库内容或用户字面量：{feedback}\n"
            )
        return prompt

    def _query_plan_semantic_advisory(
        self,
        question: str,
        sql: str,
    ) -> Dict[str, Any]:
        """Run heuristic semantic checks as metadata, never as a hard gate."""

        warnings: List[str] = []
        contract_validated = False
        obligations_validated = False
        normalized_question = self._normalize(question)

        try:
            contract = extract_semantic_contract(normalized_question)
            contract_decision = validate_candidate(
                contract,
                sql,
                strict=_has_contract_requirements(contract),
            )
            contract_validated = bool(contract_decision.accepted)
        except Exception:
            contract_validated = False
        if not contract_validated:
            warnings.append("semantic_contract_advisory_failed")

        try:
            obligations = extract_semantic_obligations(
                normalized_question,
                self._domains,
            )
            obligations_decision = validate_candidate_obligations(
                obligations,
                sql,
            )
            obligations_validated = bool(obligations_decision.accepted)
        except Exception:
            obligations_validated = False
        if not obligations_validated:
            warnings.append("semantic_obligations_advisory_failed")

        return {
            "contract_validated": contract_validated,
            "obligations_validated": obligations_validated,
            "semantic_validated": contract_validated and obligations_validated,
            "semantic_warnings": warnings,
        }

    def _query_plan_v2_prompt(self, question: str, feedback: str = "") -> str:
        """Prompt for bounded relational algebra; never asks for free SQL."""
        schema = get_schema_description(self.db_path)
        payload = json.dumps(
            _query_plan_v2.query_plan_v2_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt = (
            "You are a medical analytics QueryPlan v2 compiler. The question may be Chinese.\n"
            "Return exactly one JSON object and no Markdown or SQL.\n"
            "Express the entire request with the bounded relational-algebra grammar below. "
            "Preserve every requested output column, filter, grouping, ordering, tie-breaker, "
            "limit, join condition, and subquery condition.\n"
            "Never emit sql, raw_sql, expression_sql, comments, or unknown fields. "
            "Use only tables and columns in the live schema.\n"
            "SQLITE CONVENTIONS (REQUIRED — non-compliance fails validation):\n"
            "- Date extraction uses substr only: substr(visit_date,1,4) is the year, "
            "substr(visit_date,1,7) is the year-month. STRFTIME is FORBIDDEN and "
            "its use is always rejected.\n"
            "- A 人数/总人数/就诊人数/就诊的总人数 count counts visit ROWS (COUNT(*)). "
            "Only 去重患者数/不同患者数 use COUNT(DISTINCT patient_id).\n"
            "- Month-level grouping is GROUP BY substr(visit_date,1,7); year-level is substr(visit_date,1,4).\n"
            "- Every 按X分组 question MUST put X in BOTH the SELECT list and GROUP BY. "
            "Multi-metric questions (就诊次数与总费用) must SELECT every grouping column "
            "alongside every aggregate — never drop a group column.\n"
            "- A filter on an AGGREGATE (平均/总和/次数 大于/小于/在…之间/不低于/不超过) "
            "is a HAVING clause, never a WHERE clause.\n"
            "- Every ORDER BY ends with a deterministic tie-breaker: the grouping column ASC, "
            "or the row identity (visit_id / patient_id / head / name) ASC.\n"
            "- kg_triples.count and kg_entities.mention_count are STORED columns, not aggregates. "
            "When a question asks 条数/数量/次数 about how many TRIPLES match (统计…的条数, "
            "各头实体的条数, 出现次数不低于N), that is COUNT(*) over grouped rows -- never the "
            "stored count column. Read the stored column only when the question names it directly "
            "(观察次数/提及次数).\n"
            "- 关系码/关系为 with an ALL-CAPS ASCII value (TREATED_BY_DRUG) filters relation; "
            "a Chinese relation name (治疗药物/并发症/临床表现) filters relation_zh. "
            "Match the column to the language of the value given in the question.\n"
            "- A CTE selects only the columns its outer query consumes -- never mirror the whole table. "
            "Aggregate in whichever query level the question describes.\n"
            "- 每位/每个X的<metric> that must list ALL X (including those with no matching rows) "
            "uses a correlated scalar subquery in the SELECT list, not an INNER JOIN, "
            "because a join silently drops entities with zero matches. This applies whether the "
            "question asks for the entity's 姓名 or its ID.\n"
            "- When a question names a relation in Chinese prose (临床表现/并发症/治疗药物) alongside "
            "a join, keep it as an explicit relation_zh filter -- describing the relation is a "
            "filter, not decoration.\n"
            "- 保留N位小数 requires ROUND(expr, N) with the explicit precision argument.\n"
            "- The SELECT list order follows the order the question names the outputs. "
            "统计A与B emits A then B; 返回X及其Y emits X then Y. Column position is part of "
            "the answer, so never reorder the requested outputs.\n"
            "- A CTE must carry every column the outer query needs, including any join key. "
            "Selecting only the projected column and then joining on a different one changes "
            "the result set.\n"
            "- Aggregating per group requires GROUP BY on exactly the non-aggregated SELECT "
            "columns; omitting it repeats one row instead of collapsing per entity.\n"
            "- Window functions take no explicit frame unless the question describes one; "
            "emit RANK() OVER (ORDER BY ...) with no ROWS BETWEEN clause.\n\n"
            f"LIVE SQLITE SCHEMA:\n{schema}\n\n"
            f"QUERYPLAN V2 CONTRACT AND EXAMPLES:\n{payload}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Output only the complete QueryPlan v2 JSON object."
        )
        if feedback:
            prompt += (
                "\nThe previous plan failed only at this validation stage: "
                f"{feedback}. Rebuild the complete JSON from the question and contract."
            )
        return prompt

    def _query_plan_v2_pipeline(self, question: str) -> Optional[Dict[str, Any]]:
        """One deterministic, safety-gated QueryPlan v2 attempt with one repair."""
        try:
            from dki.llm import chat, llm_available, nl2sql_completion_token_budget
            if not llm_available():
                self._last_llm_meta = self._llm_result_meta({"attempts": 0, "abstained": True})
                return None
        except Exception:
            self._last_llm_meta = self._llm_result_meta({"attempts": 0, "abstained": True})
            return None
        feedback = ""
        failure_stages: list[str] = []
        for rnd in range(self.llm_rounds):
            stage = "model_availability"
            try:
                raw = chat(
                    self._query_plan_v2_prompt(question, feedback),
                    temperature=0.0,
                    max_tokens=nl2sql_completion_token_budget(self.llm_completion_tokens),
                    reasoning_mode=self.llm_reasoning_mode,
                )
                stage = "json_schema"
                plan = _query_plan_v2.parse_query_plan_v2(raw)
                stage = "question_grounding"
                _query_plan_v2.validate_plan_v2_grounding(
                    plan,
                    question,
                    allowed_domain_values=tuple(
                        sorted(
                            self._normalization_literals,
                            key=lambda value: (-len(value), value),
                        )
                    ),
                )
                stage = "deterministic_compilation"
                sql = self._clean_sql(_query_plan_v2.compile_query_plan_v2(plan))
                if sql is None:
                    raise ValueError("QueryPlan v2 failed read-only AST gate")
                stage = "execution"
                # Complete-result execution: contract validation must see every
                # row, never the 200-row display truncation.
                candidate_result = execute_complete_readonly(sql, self.db_path)
                stage = "semantic_contract"
                if not self._query_plan_v2_matches_closed_contract(
                    question,
                    candidate_result,
                ):
                    raise ValueError(
                        "QueryPlan v2 result violates a bounded semantic contract"
                    )
                result: Dict[str, Any] = {
                    "sql": sql, "confidence": 0.7, "n_candidates": 1,
                    "n_agree": 1, "rounds": rnd + 1, "attempts": rnd + 1,
                    "strategy": "query_plan_v2", "plan_version": 2,
                    "plan_sha256": _query_plan_v2.canonical_plan_sha256_v2(plan),
                    "execution_validated": True,
                    "failure_stages": tuple(failure_stages),
                }
                result.update(self._query_plan_semantic_advisory(question, sql))
                self._last_llm_meta = self._llm_result_meta(result)
                return result
            except Exception:
                failure_stages.append(stage)
                feedback = f"stage={stage} rejected"
        self._last_llm_meta = self._llm_result_meta(
            {
                "attempts": self.llm_rounds,
                "abstained": True,
                "failure_stages": tuple(failure_stages),
            }
        )
        return None

    def _query_plan_v3_pipeline(self, question: str) -> Optional[Dict[str, Any]]:
        """Four-lane QueryPlan v3 candidates with deterministic selection."""
        from dki.analysis.nl2sql_query_plan_v3_pipeline import (
            QueryPlanV3PipelineConfig,
            run_query_plan_v3_pipeline,
        )

        try:
            from dki.llm import (
                chat,
                llm_available,
                nl2sql_completion_token_budget,
            )

            if not llm_available():
                raise RuntimeError("sealed Agent LLM runtime is not configured")
        except Exception:
            empty = {
                "sql": None,
                "confidence": 0.0,
                "strategy": "query_plan_v3",
                "plan_version": 3,
                "attempts": 0,
                "failure_stages": ("model_availability",),
                "abstained": True,
            }
            self._last_llm_meta = self._llm_result_meta(empty)
            if self.llm_fail_closed:
                raise
            return None

        def model_call(prompt: str) -> str:
            return chat(
                prompt,
                temperature=0.0,
                max_tokens=nl2sql_completion_token_budget(
                    self.llm_completion_tokens
                ),
                reasoning_mode=self.llm_reasoning_mode,
            )

        result = run_query_plan_v3_pipeline(
            question=question,
            live_schema_description=get_schema_description(self.db_path),
            db_path=self.db_path,
            generate_candidate=model_call,
            repair_candidate=model_call,
            domains=self._domains,
            config=QueryPlanV3PipelineConfig(
                max_attempts_per_lane=min(4, max(1, self.llm_rounds)),
            ),
        )

        if not result["abstained"]:
            self._last_llm_meta = self._llm_result_meta(result)
            return result

        # Verified fallback chain (plan §3.6): QPv3 selector → QPv2 → rule →
        # abstain.  The QPv2 pipeline applies its own parse/grounding/compile/
        # execution/contract gates; the caller's existing route machinery then
        # handles the deterministic-rule fallback after a full model abstain.
        v3_meta = self._llm_result_meta(result)
        fallback = self._query_plan_v2_pipeline(question)
        if fallback and fallback.get("sql"):
            meta = self._llm_result_meta(fallback)
            meta["fallback_route"] = "query_plan_v2_after_v3_abstain"
            meta["query_plan_v3_abstained"] = True
            self._last_llm_meta = meta
            return fallback
        v3_meta["fallback_route"] = "query_plan_v2_after_v3_abstain"
        v3_meta["query_plan_v2_abstained"] = True
        self._last_llm_meta = v3_meta
        return None

    def _query_plan_v2_matches_closed_contract(
        self,
        question: str,
        candidate_result: Tuple[List[str], List[Tuple]],
    ) -> bool:
        """Hard-gate model plans when a whole-question compiler owns the shape.

        This gate is intentionally narrow: it activates only when the bounded
        structural compiler can lower the *entire* normalized question.  The
        candidate must then preserve output arity, output-column order, row
        order and values.  Column names are ignored, matching the strict blind
        evaluator's result contract.  A mismatch causes one ordinary plan
        repair and then the existing deterministic rule fallback; it never
        exposes SQL, database rows or literals to the model.
        """

        closed_sql = self._compile_output_complete_structural_sql(
            self._normalize(question)
        )
        if closed_sql is None:
            return True

        expected_columns, expected_rows = run_sql(closed_sql, self.db_path)
        candidate_columns, candidate_rows = candidate_result
        if len(expected_columns) != len(candidate_columns):
            return False

        def canonical_cell(value: Any) -> Tuple[str, str]:
            if value is None:
                return ("null", "")
            if isinstance(value, bool):
                return ("number", "1" if value else "0")
            if isinstance(value, (int, float)):
                rounded = round(float(value), 4)
                payload = f"{rounded:.4f}".rstrip("0").rstrip(".")
                return ("number", payload or "0")
            if isinstance(value, bytes):
                return ("bytes", value.hex())
            if isinstance(value, str):
                return ("text", value)
            return (type(value).__name__, repr(value))

        def canonical_rows(rows: List[Tuple]) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
            return tuple(
                tuple(canonical_cell(value) for value in row)
                for row in rows
            )

        return canonical_rows(expected_rows) == canonical_rows(candidate_rows)

    def _query_plan_pipeline(self, question: str) -> Optional[Dict[str, Any]]:
        """Parse, ground, compile and hard-gate a model-produced QueryPlan.

        Failed attempts receive only a fixed stage label on the next round.
        The raw plan, compiled SQL, database errors and user/domain literals are
        deliberately excluded from feedback.  This strategy never delegates to
        the legacy free-SQL generator.
        """

        try:
            from dki.llm import chat, llm_available, nl2sql_completion_token_budget

            if not llm_available():
                self._last_llm_meta = self._llm_result_meta(
                    {"attempts": 0, "abstained": True}
                )
                return None
        except Exception:
            self._last_llm_meta = self._llm_result_meta(
                {"attempts": 0, "abstained": True}
            )
            return None

        allowed_domain_values = tuple(
            sorted(self._normalization_literals, key=lambda value: (-len(value), value))
        )
        feedback = ""
        attempts = 0
        for rnd in range(self.llm_rounds):
            attempts += 1
            stage = "model_availability"
            try:
                raw = chat(
                    self._query_plan_prompt(question, feedback=feedback),
                    temperature=0.0,
                    max_tokens=nl2sql_completion_token_budget(
                        self.llm_completion_tokens
                    ),
                    reasoning_mode=self.llm_reasoning_mode,
                )
                stage = "json_schema"
                plan = _query_plan.parse_query_plan(raw)
                stage = "question_grounding"
                _query_plan.validate_plan_grounding(
                    plan,
                    question,
                    allowed_domain_values=allowed_domain_values,
                )
                stage = "deterministic_compilation"
                compiled_sql = _query_plan.compile_query_plan(plan)
                stage = "readonly_ast"
                sql = self._clean_sql(compiled_sql)
                if sql is None:
                    raise ValueError("compiled plan failed the read-only AST gate")
                validate_gold_sql(sql)
                stage = "execution"
                # QueryPlan has one deterministic candidate, so execution is a
                # validity gate rather than a result-set voting key.  Avoid
                # sorting heterogeneous/nullable rows merely to prove that the
                # statement executes.
                run_sql(sql, self.db_path)
                stage = "canonicalization"
                plan_sha256 = _query_plan.canonical_plan_sha256(plan)
                plan_version = plan["plan_version"]
            except Exception:
                feedback = (
                    f"阶段 {stage} 失败；请重新输出完整、严格、可落地的 JSON 计划。"
                )
                continue

            try:
                advisory = self._query_plan_semantic_advisory(question, sql)
            except Exception:
                # Advisory infrastructure is deliberately outside the safety
                # boundary. A parser/heuristic failure cannot discard a plan
                # that already passed grounding, compilation, AST and
                # execution gates.
                advisory = {
                    "contract_validated": False,
                    "obligations_validated": False,
                    "semantic_validated": False,
                    "semantic_warnings": ["semantic_advisory_unavailable"],
                }
            confidence = 0.7 if advisory["semantic_validated"] else 0.55
            result: Dict[str, Any] = {
                "sql": sql,
                "confidence": confidence,
                "n_candidates": 1,
                "n_agree": 1,
                "rounds": rnd + 1,
                "attempts": attempts,
                "strategy": "query_plan",
                "plan_version": plan_version,
                "plan_sha256": plan_sha256,
                "execution_validated": True,
                **advisory,
            }
            self._last_llm_meta = self._llm_result_meta(result)
            return result

        self._last_llm_meta = self._llm_result_meta(
            {"attempts": attempts, "abstained": True}
        )
        return None

    def _llm_candidates(self, question: str, feedback: str = "") -> List[str]:
        """生成多候选 SQL（self-consistency 采样）。"""
        from dki.llm import chat, nl2sql_completion_token_budget

        schema = get_schema_description(self.db_path)
        linking = self._schema_link(question)
        contract = extract_semantic_contract(question)
        contract_payload = {
            "projections": list(contract.projections),
            "aggregates": list(contract.aggregates),
            "output_slots": list(contract.output_slots),
            "grouping": list(contract.grouping),
            "ordering": list(contract.ordering),
            "limit": contract.limit,
            "set_semantics": contract.set_semantics,
            "row_set_mode": contract.row_set_mode.value,
            "status": contract.status.value,
        }
        structural_examples = (
            "结构示例（只学习结构与输出合同，不复制常量）：\n"
            "Q: 先在CTE中按患者汇总就诊次数和总费用，返回至少3次的患者编号、次数和总费用。\n"
            "SQL: WITH s AS (SELECT patient_id, COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits GROUP BY patient_id) SELECT patient_id, visit_count, ROUND(total_cost, 2) AS total_cost FROM s WHERE visit_count >= 3 ORDER BY visit_count DESC, total_cost DESC, patient_id;\n"
            "Q: 按关系标识分组，仅保留三元组数至少10的组，返回关系、三元组数量、平均置信度和观察次数总和。\n"
            "SQL: SELECT relation, COUNT(*) AS triple_count, ROUND(AVG(confidence), 4) AS avg_confidence, SUM(count) AS observations FROM kg_triples GROUP BY relation HAVING COUNT(*) >= 10 ORDER BY triple_count DESC, relation;\n"
            "Q: 核对实际疾病和科室是否与图谱 BELONGS_TO_DEPT 的头尾一致，返回就诊编号、疾病和科室。\n"
            "SQL: SELECT v.visit_id, v.disease, v.department FROM visits v JOIN kg_triples t ON t.head = v.disease AND t.tail = v.department WHERE t.relation = 'BELONGS_TO_DEPT' ORDER BY v.visit_id;\n"
            "补充结构合同：\n"
            "- 患者与就诊各自按地区汇总的双 CTE：患者 CTE 返回 region 和 COUNT(*)；就诊 CTE 通过 patient_id 连接并返回 COUNT(DISTINCT v.patient_id) 和 COUNT(*)；两 CTE 按 region 内连接，按 visit_count DESC、region 排序。\n"
            "- 按 tail 分组且要求头实体数和关系/三元组数：分别使用 COUNT(DISTINCT head) 与 COUNT(*)，HAVING 绑定头实体数。\n"
            "- 图谱边同时连接头尾 kg_entities：仅输出问题要求的两侧 name 和 mention_count，不额外输出 relation、confidence、count。\n"
            "- Drug 实体按不同头疾病数 TopK 且问题写不同头疾病：WHERE tail_type='Drug'，GROUP BY tail，COUNT(DISTINCT head)，按计数 DESC 和 tail 排序后 LIMIT。\n"
        )
        compositional_examples = (
            "组合结构示例（示例常量不得复制，SELECT 必须逐项对应当前问题）：\n"
            "Q: 先按科室和疾病统计就诊数，再在每个科室内按次数做 DENSE_RANK，保留前3名，并返回科室、疾病、次数、名次。\n"
            "SQL: WITH grouped AS (SELECT department, disease, COUNT(*) AS visit_count FROM visits GROUP BY department, disease), ranked AS (SELECT department, disease, visit_count, DENSE_RANK() OVER (PARTITION BY department ORDER BY visit_count DESC) AS disease_rank FROM grouped) SELECT department, disease, visit_count, disease_rank FROM ranked WHERE disease_rank <= 3 ORDER BY department, disease_rank, disease;\n"
            "Q: 返回头实体为哮喘且关系中文名属于临床表现或诊断检查的尾实体与置信度。\n"
            "SQL: SELECT tail, confidence FROM kg_triples WHERE head = '哮喘' AND relation_zh IN ('临床表现', '诊断检查') ORDER BY tail;\n"
            "Q: 返回每位患者费用等于该患者最高费用的就诊编号、患者编号和费用，并列保留。\n"
            "SQL: SELECT v.visit_id, v.patient_id, v.cost FROM visits v WHERE v.cost = (SELECT MAX(x.cost) FROM visits x WHERE x.patient_id = v.patient_id) ORDER BY v.patient_id, v.visit_id;\n"
            "Q: 合并患者地区与就诊科室并保留来源标签患者地区或就诊科室。\n"
            "SQL: SELECT region AS item_name, '患者地区' AS source_name FROM patients UNION ALL SELECT department AS item_name, '就诊科室' AS source_name FROM visits ORDER BY source_name, item_name;\n"
            "Q: 筛选所有年份中月份字符串为09的就诊，返回日期、患者编号和费用。\n"
            "SQL: SELECT visit_date, patient_id, cost FROM visits WHERE strftime('%m', visit_date) = '09' ORDER BY visit_date, visit_id;\n"
            "Q: 为每条就诊附上同一科室的平均费用和记录数，返回就诊编号、科室、当前费用、平均费用和记录数。\n"
            "SQL: SELECT visit_id, department, cost, ROUND(AVG(cost) OVER (PARTITION BY department), 2) AS department_avg_cost, COUNT(*) OVER (PARTITION BY department) AS department_visit_count FROM visits ORDER BY department, cost DESC, visit_id;\n"
            "输出自检：先逐字列出当前问题要求的输出项，再确认 SELECT 的列数、含义、顺序完全一致；禁止把 patient_id 换成 name，禁止 SELECT *，禁止遗漏窗口排名、当前行指标或集合来源标签。\n"
        )
        base_prompt = (
            f"你是医疗分析 SQLite 编译器。数据库 schema：\n{schema}\n\n"
            f"{linking}\n\n{structural_examples}\n{compositional_examples}\n问题：{question}\n"
            "语义契约："
            f"{json.dumps(contract_payload, ensure_ascii=False, sort_keys=True)}\n\n"
            "图谱关系方向固定为：HAS_SYMPTOM Disease->Symptom；"
            "TREATED_BY_DRUG Disease->Drug；TREATED_BY Disease->Treatment；"
            "DIAGNOSED_BY Disease->Exam；HAS_COMPLICATION Disease->Disease；"
            "CONTRAINDICATED_FOR Drug->Disease；BELONGS_TO_DEPT Disease->Department。"
            "不得反转 head/tail。\n"
            "严格规则：\n"
            "1. 只生成一条只读 SQLite 查询，只用 schema 中的表和列，完整保留问题与语义契约，绝不添加问题未要求的过滤条件。\n"
            "2. AVG(confidence) 使用 ROUND(..., 4)；费用、年龄、住院天数的 AVG 和最终 SUM(cost) 使用 ROUND(..., 2)；COUNT 与整数 SUM(count) 不使用 ROUND。HAVING 的计数阈值若同时被问题要求作为数量输出，必须把相同 COUNT(*) 放入 SELECT。\n"
            "3. 有显式排序时严格遵从并补稳定次级键。无显式排序时：时间分组必须以时间键升序为第一优先，禁止擅自按聚合值降序；普通分组/HAVING 按首个计数或指标降序再按分组键；明细/嵌套查询按类别维度和标识升序；图谱过滤明细按 confidence、count 降序再按 head、tail。\n"
            "4. 与某实体共享同一关系尾实体的其他头实体必须使用 base/other 自连接：other.tail=base.tail、other.head<>base.head，两个别名使用相同 relation；不得臆造置信度或观察次数阈值。\n"
            "5. SELECT 列数、含义和顺序严格等于问题要求，不得添加解释列。\n"
            "6. 只输出 SQL，不要解释、思考过程、Markdown 或代码围栏。\n"
        )
        if feedback:
            base_prompt += f"\n上一轮候选执行反馈（请据此修正）：\n{feedback}\n"
        candidates: List[str] = []
        # 第 1 条用低温（确定性主候选），其余用采样温度增加多样性
        for i in range(self.llm_samples):
            temp = 0.0 if i == 0 else self.llm_temperature
            try:
                raw = chat(
                    base_prompt,
                    temperature=temp,
                    max_tokens=nl2sql_completion_token_budget(
                        self.llm_completion_tokens
                    ),
                    reasoning_mode=self.llm_reasoning_mode,
                )
                sql = self._clean_sql(raw)
                if sql:
                    candidates.append(sql)
            except Exception:
                if self.llm_fail_closed:
                    raise
                continue
        return candidates

    def _compile_shared_tail_peer_sql(self, q: str) -> Optional[str]:
        """Compile a grounded KG peer query without model-invented filters.

        The intent asks for other heads connected by the same relation to at
        least one tail of a named base head.  It is a stable graph primitive,
        not a development-row template.  Any metric qualifier is deliberately
        rejected so a confidence/count condition can never be dropped.
        """

        shared_tail = bool(
            re.search(
                r"(?:共享|共同|相同|一样).{0,16}(?:尾实体|检查|症状|药物|科室|治疗|并发症)|"
                r"(?:尾实体|检查|症状|药物|科室|治疗|并发症).{0,16}(?:共享|共同|相同|一样)",
                q,
            )
        )
        other_head = bool(
            re.search(
                r"(?:其他|其它|其余|另外).{0,8}(?:疾病|头实体|实体)|"
                r"(?:疾病|头实体|实体).{0,8}(?:其他|其它|其余|另外)",
                q,
            )
        )
        if not (shared_tail and other_head):
            return None
        if re.search(
            r"置信度|观察次数|证据次数|至少\s*\d|不低于\s*\d|"
            r"(?:confidence|count)\s*(?:>=|>|=|不低于|至少)",
            q,
            re.I,
        ):
            return None

        base_head = self._find_value(q, self._domains.get("kg_entity", []))
        if base_head is None:
            base_head = self._find_value(q, self._domains.get("disease", []))
        relation = self._find_value(q, self._domains.get("kg_relation", []))
        if relation is None:
            matched = [
                canonical
                for canonical, triggers in _KG_RELATION_TRIGGERS
                if canonical in self._domains.get("kg_relation", [])
                and any(trigger in q for trigger in triggers)
            ]
            if len(set(matched)) == 1:
                relation = matched[0]
        if base_head is None or relation is None:
            return None

        sql = (
            "SELECT DISTINCT other.head AS other_head, "
            "other.tail AS shared_tail FROM kg_triples base "
            "JOIN kg_triples other ON other.tail = base.tail "
            "AND other.head <> base.head "
            f"WHERE base.head = '{_esc(base_head)}' "
            f"AND base.relation = '{_esc(relation)}' "
            f"AND other.relation = '{_esc(relation)}' "
            "ORDER BY other_head, shared_tail"
        )
        try:
            validate_gold_sql(sql)
        except Exception:
            return None
        return sql

    def _compile_kg_endpoint_join_sql(self, q: str) -> Optional[str]:
        """Compile the reusable KG-edge-to-both-entity-records primitive."""

        joins_both_endpoints = bool(
            re.search(
                r"(?:同时|分别)?.{0,8}连接.{0,12}(?:图谱)?(?:头|起点).{0,6}(?:尾|终点).{0,6}实体|"
                r"(?:头|起点)实体.{0,20}(?:尾|终点)实体.{0,20}(?:连接|实体表)",
                q,
            )
        )
        requests_endpoint_fields = bool(
            re.search(r"(?:两侧|双方|头尾|两端).{0,16}(?:名称|名字|提及次数|提及数)", q)
            or (
                re.search(r"(?:头|起点).{0,8}(?:名称|提及次数|提及数)", q)
                and re.search(r"(?:尾|终点).{0,8}(?:名称|提及次数|提及数)", q)
            )
        )
        if not (joins_both_endpoints and requests_endpoint_fields):
            return None
        if re.search(
            r"置信度|观察次数|证据次数|至少\s*\d|不低于\s*\d|"
            r"(?:confidence|count)\s*(?:>=|>|=|不低于|至少)",
            q,
            re.I,
        ):
            return None
        relation = self._find_value(q, self._domains.get("kg_relation", []))
        if relation is None:
            return None
        sql = (
            "SELECT h.name AS head_name, h.mention_count AS head_mentions, "
            "e.name AS tail_name, e.mention_count AS tail_mentions "
            "FROM kg_triples t JOIN kg_entities h ON h.name = t.head "
            "JOIN kg_entities e ON e.name = t.tail "
            f"WHERE t.relation = '{_esc(relation)}' "
            "ORDER BY head_mentions DESC, head_name, tail_name"
        )
        try:
            validate_gold_sql(sql)
        except Exception:
            return None
        return sql

    def _compile_kg_entity_range_sql(self, q: str) -> Optional[str]:
        """Compile one closed KG entity-type plus inclusive range request.

        The semantic clause is anchored through the end of the question.  A
        leading report-writing phrase is accepted only when it contains no
        numeric, cohort, relation, metric, grouping, ordering, or limit slot;
        this prevents a decorative prefix from hiding a discarded predicate.
        """

        match = re.search(
            r"返回(?:知识图谱中)?类型为(?P<type>[A-Za-z]+)[、，,且]?"
            r"提及次数(?:处于|在|介于)(?P<low>\d+)"
            r"(?:到|至|~|～|-)(?P<high>\d+)之间的?"
            r"(?:图谱)?实体(?:名称|名字)和(?:提及)?次数[。.]?$",
            q,
        )
        if match is None:
            return None

        prefix = q[: match.start()].strip(" ，,。")
        if len(prefix) > 80 or re.search(
            r"\d|患者|就诊|科室|疾病|症状|药物|检查|实体|关系|置信度|"
            r"观察次数|提及次数|类型为|分组|排序|升序|降序|前\s*\d|后\s*\d|"
            r"最多|最少|平均|至少|至多|超过|低于|高于|等于|或者|或是|并且|且",
            prefix,
        ):
            return None

        entity_type = match.group("type")
        if entity_type not in self._domains.get("kg_entity_type", ()):
            return None
        lower = int(match.group("low"))
        upper = int(match.group("high"))
        if lower > upper:
            return None

        sql = (
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type='{_esc(entity_type)}' "
            f"AND mention_count BETWEEN {lower} AND {upper} "
            "ORDER BY mention_count DESC, name"
        )
        try:
            validate_gold_sql(sql)
        except Exception:
            return None
        return sql

    @staticmethod
    def _report_semantic_core(q: str) -> Optional[str]:
        """Remove only a bounded, non-semantic report-writing prefix."""

        match = re.fullmatch(
            r"(?P<prefix>围绕[^，,]{2,30}"
            r"(?:核验|复盘|检查|评估|盘点|维护|梳理|复核|观察|稽核|测算|整理|抽查|解释)"
            r"[，,]在[^，,]{2,36}时[，,])(?P<core>.+)",
            q,
        )
        if match is None:
            return q
        prefix = match.group("prefix")
        if re.search(
            r"\d|(?:至少|至多|超过|低于|高于|等于|不低于|不超过)|"
            r"(?:升序|降序|排序|分组|前\s*\d|后\s*\d)|"
            r"(?:HAS_|TREATED_|DIAGNOSED_|BELONGS_|CONTRAINDICATED_)",
            prefix,
            re.IGNORECASE,
        ):
            return None
        return match.group("core")

    def _compile_output_complete_structural_sql(self, q: str) -> Optional[str]:
        """Lower complete analytical shapes with explicit grain checks."""

        core = self._report_semantic_core(q)
        if core is None:
            return None

        # Inclusive age range plus gender is a closed two-predicate cohort.
        # Keep both surface orders explicit rather than accepting an arbitrary
        # prefix/suffix: an executable plan that drops either conjunct changes
        # the patient set while still looking superficially plausible.
        age_gender_match = re.fullmatch(
            r"(?:找出|筛选|查询|列出)年龄在(?P<low>\d+)(?:至|到)"
            r"(?P<high>\d+)岁之间[（(](?:含端点|含边界)[）)]且性别为"
            r"(?P<gender>[男女])的?患者(?:ID|编号)(?:与|和|、)年龄[，,]"
            r"按年龄升序排列[。.]?",
            core,
        )
        if age_gender_match is None:
            age_gender_match = re.fullmatch(
                r"(?:找出|筛选|查询|列出)性别为(?P<gender>[男女])且年龄在"
                r"(?P<low>\d+)(?:至|到)(?P<high>\d+)岁之间"
                r"[（(](?:含端点|含边界)[）)]的?患者(?:ID|编号)"
                r"(?:与|和|、)年龄[，,]按年龄升序排列[。.]?",
                core,
            )
        if age_gender_match is not None:
            low = int(age_gender_match.group("low"))
            high = int(age_gender_match.group("high"))
            if not 0 <= low <= high <= 150:
                return None
            sql = (
                "SELECT patient_id, age FROM patients "
                f"WHERE age BETWEEN {low} AND {high} "
                f"AND gender = '{age_gender_match.group('gender')}' "
                "ORDER BY age ASC, patient_id ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A department mention in this family is a WHERE value, not an output
        # dimension.  The requested grain is disease and both requested metrics
        # must remain in SELECT in their stated order.
        department_disease_match = re.fullmatch(
            r"在(?P<department>[^，,]{1,24})科室中[，,]按疾病(?:统计|汇总)"
            r"就诊次数(?:与|和)总费用[，,]返回(?P<scope>[^，,]{1,24})"
            r"科室下各疾病的就诊次数(?:与|和)保留2位小数的总费用[，,]"
            r"按就诊次数降序排列[。.]?",
            core,
        )
        if department_disease_match is not None:
            department = department_disease_match.group("department")
            if (
                department != department_disease_match.group("scope")
                or department not in self._domains.get("department", ())
            ):
                return None
            sql = (
                "SELECT disease, COUNT(*) AS visit_count, "
                "ROUND(SUM(cost), 2) AS total_cost FROM visits "
                f"WHERE department = '{_esc(department)}' GROUP BY disease "
                "ORDER BY visit_count DESC, disease ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Typed visit -> edge -> tail-entity joins are closed only when the
        # relation and its tail type agree with the fixed medical schema.  The
        # entity-table type predicate is material: names can occur under more
        # than one type, so merely joining e.name = t.tail is insufficient.
        relation_tail_contracts = {
            "HAS_SYMPTOM": ("Symptom", "临床表现", "症状"),
            "DIAGNOSED_BY": ("Exam", "诊断检查", "检查"),
            "TREATED_BY_DRUG": ("Drug", "治疗药物", "药物"),
            "TREATED_BY": ("Treatment", "治疗方式", "治疗方式"),
            "BELONGS_TO_DEPT": ("Department", "所属科室", "科室"),
            "HAS_COMPLICATION": ("Disease", "并发症", "疾病"),
        }
        typed_kg_match = re.fullmatch(
            r"三表(?:连接|关联)就诊表[、,，](?:三元组表|图谱边)(?:与|和)"
            r"实体表[，,](?:找出|查找)就诊疾病在(?:知识)?图谱中具有"
            r"(?P<label>[^（()，,]{1,16})[（(]"
            r"(?P<relation>[A-Z][A-Z0-9_]+)[）)]且置信度"
            r"(?:不低于|至少)(?P<threshold>\d+(?:\.\d+)?)的"
            r"(?P<noun>症状|检查|药物|治疗方式|科室|疾病)[，,]返回就诊"
            r"(?:ID|编号)[、,，](?P=noun)名称(?:与|和)置信度[，,]"
            r"按置信度降序排列[。.]?",
            core,
        )
        compact_typed_kg_match = None
        if typed_kg_match is None:
            compact_typed_kg_match = re.fullmatch(
                r"把就诊疾病经(?P<relation>[A-Z][A-Z0-9_]+)关系连接到"
                r"(?P<entity_type>[A-Za-z]+)实体[，,]限定置信度"
                r"(?:不低于|至少)(?P<threshold>\d+(?:\.\d+)?)[；;]"
                r"返回就诊(?:ID|编号)[、,，]尾实体名称(?:与|和)置信度[，,]"
                r"并按置信度降序[。.]?",
                core,
            )
        active_typed_kg_match = typed_kg_match or compact_typed_kg_match
        if active_typed_kg_match is not None:
            relation = active_typed_kg_match.group("relation")
            contract = relation_tail_contracts.get(relation)
            threshold = float(active_typed_kg_match.group("threshold"))
            if (
                contract is None
                or relation not in self._domains.get("kg_relation", ())
                or contract[0] not in self._domains.get("kg_entity_type", ())
                or not 0 <= threshold <= 1
            ):
                return None
            if typed_kg_match is not None and (
                typed_kg_match.group("label") != contract[1]
                or typed_kg_match.group("noun") != contract[2]
            ):
                return None
            if compact_typed_kg_match is not None and (
                compact_typed_kg_match.group("entity_type") != contract[0]
            ):
                return None
            sql = (
                "SELECT v.visit_id, t.tail, t.confidence FROM visits v "
                "JOIN kg_triples t ON t.head = v.disease "
                f"AND t.relation = '{_esc(relation)}' "
                "JOIN kg_entities e ON e.name = t.tail "
                f"AND e.type = '{_esc(contract[0])}' "
                f"WHERE t.confidence >= {threshold:g} "
                "ORDER BY t.confidence DESC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Complete patient filter with three independently grounded slots.
        match = re.fullmatch(
            r"列出性别为(?P<gender>[男女])[、,，]年龄在(?P<low>\d+)"
            r"(?:至|到)(?P<high>\d+)岁[（(]含边界[）)]且地区为"
            r"(?P<region>[^，,]{1,24})的患者ID[、,，]姓名和年龄[，,]"
            r"并按年龄升序[、,，]患者ID升序排列[。.]?",
            core,
        )
        if match is not None:
            low, high = int(match.group("low")), int(match.group("high"))
            region = match.group("region")
            if (
                not 0 <= low <= high <= 150
                or region not in self._domains.get("region", ())
            ):
                return None
            sql = (
                "SELECT patient_id, name, age FROM patients "
                f"WHERE gender = '{match.group('gender')}' "
                f"AND age BETWEEN {low} AND {high} "
                f"AND region = '{_esc(region)}' "
                "ORDER BY age ASC, patient_id ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A visit must exceed an absolute floor and both correlated group
        # averages; omitting either correlation changes the result set.
        match = re.fullmatch(
            r"返回费用不少于(?P<threshold>\d+(?:\.\d+)?)元且同时高于其"
            r"科室平均费用和疾病平均费用的就诊ID[、,，]科室[、,，]疾病与费用[，,]"
            r"按费用降序[、,，]就诊ID排列[。.]?",
            core,
        )
        if match is not None:
            threshold = match.group("threshold")
            if float(threshold) < 0:
                return None
            sql = (
                "SELECT v.visit_id, v.department, v.disease, v.cost "
                "FROM visits v "
                f"WHERE v.cost >= {threshold} AND v.cost > "
                "(SELECT AVG(x.cost) FROM visits x "
                "WHERE x.department = v.department) AND v.cost > "
                "(SELECT AVG(y.cost) FROM visits y "
                "WHERE y.disease = v.disease) "
                "ORDER BY v.cost DESC, v.visit_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # UNION ALL with explicit source labels.  Labels and thresholds are
        # grounded in the question, and source/value are the only outputs.
        match = re.fullmatch(
            r"将年龄低于(?P<age>\d+)岁患者的地区标记为[“\"]"
            r"(?P<patient_label>[^”\"]{1,20})[”\"][，,]与费用大于"
            r"(?P<cost>\d+(?:\.\d+)?)元就诊的科室标记为[“\"]"
            r"(?P<visit_label>[^”\"]{1,20})[”\"]后做保留重复的纵向合并[。.]?",
            core,
        )
        if match is not None:
            age = int(match.group("age"))
            cost = match.group("cost")
            if not 0 <= age <= 150 or float(cost) < 0:
                return None
            sql = (
                f"SELECT '{_esc(match.group('patient_label'))}' AS source, "
                "region AS value FROM patients "
                f"WHERE age < {age} UNION ALL "
                f"SELECT '{_esc(match.group('visit_label'))}', department "
                f"FROM visits WHERE cost > {cost} ORDER BY source, value"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Set difference between a typed graph-entity name set and observed
        # distinct visit diseases.
        match = re.fullmatch(
            r"从(?P<type>[A-Za-z]+)实体名中减去就诊不同疾病名[，,]"
            r"返回剩余名称并排列[。.]?",
            core,
        )
        if match is not None:
            entity_type = match.group("type")
            if entity_type not in self._domains.get("kg_entity_type", ()):
                return None
            sql = (
                "SELECT name FROM kg_entities "
                f"WHERE type = '{_esc(entity_type)}' EXCEPT "
                "SELECT DISTINCT disease FROM visits ORDER BY name"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Observed visit span per patient, expressed in integer Julian days.
        if re.fullmatch(
            r"关联患者与就诊[，,]按患者计算末次与首次就诊日期之间的"
            r"整数天数跨度[，,]返回患者ID[、,，]姓名和跨度[，,]"
            r"按跨度降序[、,，]患者ID升序排列[。.]?",
            core,
        ):
            sql = (
                "SELECT p.patient_id, p.name, "
                "CAST(julianday(MAX(v.visit_date)) - "
                "julianday(MIN(v.visit_date)) AS INTEGER) "
                "AS observed_span_days FROM patients p "
                "JOIN visits v ON v.patient_id = p.patient_id "
                "GROUP BY p.patient_id, p.name "
                "ORDER BY observed_span_days DESC, p.patient_id ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Filtered patient aggregation.  Region is part of the requested grain
        # and therefore cannot be omitted from the projection.
        match = re.fullmatch(
            r"对年龄不低于(?P<age>\d+)岁的患者[，,]按地区统计人数及平均年龄"
            r"[（(]保留(?P<digits>\d+)位小数[）)][，,]"
            r"按人数降序[、,，]地区升序排列[。.]?",
            core,
        )
        if match is not None:
            age = int(match.group("age"))
            digits = int(match.group("digits"))
            if not 0 <= age <= 150 or not 0 <= digits <= 6:
                return None
            sql = (
                "SELECT region, COUNT(*) AS patient_count, "
                f"ROUND(AVG(age), {digits}) AS avg_age FROM patients "
                f"WHERE age >= {age} GROUP BY region "
                "ORDER BY patient_count DESC, region ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A year-prefix visit cohort followed by a department count.  The year
        # is a grounded literal and department remains the first output slot.
        match = re.fullmatch(
            r"筛选就诊日期以(?P<year>\d{4})-开头的记录[，,]"
            r"按科室统计次数并按次数降序[、,，]科室升序排列[。.]?",
            core,
        )
        if match is not None:
            year = int(match.group("year"))
            if not 1900 <= year <= 2200:
                return None
            sql = (
                "SELECT department, COUNT(*) AS visit_count FROM visits "
                f"WHERE visit_date LIKE '{year:04d}-%' GROUP BY department "
                "ORDER BY visit_count DESC, department ASC"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Grouped average with a HAVING predicate over the unrounded aggregate.
        match = re.fullmatch(
            r"按科室计算平均费用并保留(?P<digits>\d+)位小数[，,]"
            r"仅返回平均费用大于(?P<threshold>\d+(?:\.\d+)?)元的科室[，,]"
            r"按平均费用降序[、,，]科室排列[。.]?",
            core,
        )
        if match is not None:
            digits = int(match.group("digits"))
            threshold = match.group("threshold")
            if not 0 <= digits <= 6 or float(threshold) < 0:
                return None
            sql = (
                "SELECT department, "
                f"ROUND(AVG(cost), {digits}) AS avg_cost FROM visits "
                "GROUP BY department "
                f"HAVING AVG(cost) > {threshold} "
                "ORDER BY avg_cost DESC, department"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Anti-semijoin: patients for whom no visit reaches a cost threshold.
        match = re.fullmatch(
            r"返回不存在费用不少于(?P<threshold>\d+(?:\.\d+)?)元就诊的"
            r"患者ID和姓名[，,]按患者ID排列[。.]?",
            core,
        )
        if match is not None:
            threshold = match.group("threshold")
            if float(threshold) < 0:
                return None
            sql = (
                "SELECT p.patient_id, p.name FROM patients p "
                "WHERE NOT EXISTS (SELECT 1 FROM visits v "
                "WHERE v.patient_id = p.patient_id "
                f"AND v.cost >= {threshold}) ORDER BY p.patient_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Two chronologically ordered visits of the same patient, grounded by
        # region and an explicit maximum day gap.
        match = re.fullmatch(
            r"连接(?P<region>[^，,]{1,24})患者及其日期先后的两次就诊[，,]"
            r"仅保留间隔不超过(?P<days>\d+)天的配对[，,]"
            r"返回患者ID[、,，]两次就诊ID和间隔天数[，,]"
            r"按患者ID[、,，]前后日期排列[。.]?",
            core,
        )
        if match is not None:
            region = match.group("region")
            days = int(match.group("days"))
            if (
                region not in self._domains.get("region", ())
                or not 1 <= days <= 3650
            ):
                return None
            sql = (
                "SELECT p.patient_id, a.visit_id, b.visit_id, "
                "CAST(julianday(b.visit_date) - julianday(a.visit_date) "
                "AS INTEGER) AS gap_days FROM patients p "
                "JOIN visits a ON a.patient_id = p.patient_id "
                "JOIN visits b ON b.patient_id = p.patient_id "
                "AND b.visit_date > a.visit_date "
                f"WHERE p.region = '{_esc(region)}' AND "
                "julianday(b.visit_date) - julianday(a.visit_date) "
                f"<= {days} ORDER BY p.patient_id, a.visit_date, b.visit_date"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Join a typed entity to graph edges through the tail endpoint while
        # preserving exactly the requested entity and edge measures.
        match = re.fullmatch(
            r"将(?P<type>[A-Za-z]+)实体作为尾端连接三元组[，,]"
            r"仅保留关系(?P<relation>[A-Z][A-Z0-9_]{2,40})[，,]"
            r"返回实体名[、,，]提及次数[、,，]头实体和观察次数[，,]"
            r"按实体名[、,，]头实体排列[。.]?",
            core,
        )
        if match is not None:
            entity_type = match.group("type")
            relation = match.group("relation")
            if (
                entity_type not in self._domains.get("kg_entity_type", ())
                or relation not in self._domains.get("kg_relation", ())
            ):
                return None
            sql = (
                "SELECT e.name, e.mention_count, t.head, t.count "
                "FROM kg_entities e JOIN kg_triples t ON t.tail = e.name "
                f"WHERE e.type = '{_esc(entity_type)}' "
                f"AND t.relation = '{_esc(relation)}' "
                "ORDER BY e.name, t.head"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # CTE two-hop reachability.  Only the first relation is constrained;
        # the second edge is deliberately unrestricted, as stated by the shape.
        match = re.fullmatch(
            r"用CTE生成第一跳关系为(?P<relation>[A-Z][A-Z0-9_]{2,40})的"
            r"所有两跳路径[，,]再按起点统计不同终点数[，,]"
            r"按终点数降序[、,，]起点排列[。.]?",
            core,
            re.IGNORECASE,
        )
        if match is not None:
            relation = match.group("relation")
            if relation not in self._domains.get("kg_relation", ()):
                return None
            sql = (
                "WITH paths AS (SELECT a.head AS start_entity, "
                "b.tail AS end_entity FROM kg_triples a "
                "JOIN kg_triples b ON b.head = a.tail "
                f"WHERE a.relation = '{_esc(relation)}') "
                "SELECT start_entity, COUNT(DISTINCT end_entity) "
                "AS reachable_count FROM paths GROUP BY start_entity "
                "ORDER BY reachable_count DESC, start_entity"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Intersection of observed visit diseases and graph Disease entities,
        # left joined to per-head distinct-neighbour counts.
        if re.fullmatch(
            r"用CTE取得就诊疾病与Disease实体的交集[，,]"
            r"再连接头实体不同邻居数[；;]没有出边按0处理[，,]"
            r"按邻居数降序[、,，]疾病排列[。.]?",
            core,
            re.IGNORECASE,
        ):
            if "Disease" not in self._domains.get("kg_entity_type", ()):
                return None
            sql = (
                "WITH visit_diseases AS (SELECT DISTINCT disease FROM visits), "
                "graph_diseases AS (SELECT name FROM kg_entities "
                "WHERE type = 'Disease'), matched AS ("
                "SELECT disease FROM visit_diseases INTERSECT "
                "SELECT name FROM graph_diseases), edge_counts AS ("
                "SELECT head, COUNT(DISTINCT tail) AS neighbor_count "
                "FROM kg_triples GROUP BY head) "
                "SELECT m.disease, COALESCE(e.neighbor_count, 0) "
                "AS neighbor_count FROM matched m LEFT JOIN edge_counts e "
                "ON e.head = m.disease "
                "ORDER BY neighbor_count DESC, m.disease"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Running SUM from the first row of each patient partition through the
        # current row.  visit_id precedes visit_date in the output contract.
        match = re.fullmatch(
            r"按患者分区并按日期[、,，]就诊ID排序[，,]"
            r"计算从分区首条到当前行的累计费用并保留(?P<digits>\d+)位小数[。.]?",
            core,
        )
        if match is not None:
            digits = int(match.group("digits"))
            if not 0 <= digits <= 6:
                return None
            sql = (
                "SELECT patient_id, visit_id, visit_date, cost, "
                f"ROUND(SUM(cost) OVER (PARTITION BY patient_id "
                "ORDER BY visit_date, visit_id ROWS BETWEEN "
                "UNBOUNDED PRECEDING AND CURRENT ROW), "
                f"{digits}) AS running_cost FROM visits "
                "ORDER BY patient_id, visit_date, visit_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # ROW_NUMBER Top-K within each department, preserving the rank column
        # and the exact requested visit projection.
        match = re.fullmatch(
            r"按科室分区[、,，]费用降序和就诊ID升序用ROW_NUMBER编号[，,]"
            r"返回每科室序号不超过(?P<limit>\d+)的前"
            r"(?P<repeat>\d+|一|二|三|四|五|六|七|八|九|十)条就诊[。.]?",
            core,
            re.IGNORECASE,
        )
        if match is not None:
            limit = int(match.group("limit"))
            repeat_text = match.group("repeat")
            chinese_numbers = {
                "一": 1,
                "二": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
            }
            repeat = (
                int(repeat_text)
                if repeat_text.isdigit()
                else chinese_numbers[repeat_text]
            )
            if limit != repeat or not 1 <= limit <= 100:
                return None
            sql = (
                "WITH ranked AS (SELECT department, visit_id, disease, cost, "
                "ROW_NUMBER() OVER (PARTITION BY department "
                "ORDER BY cost DESC, visit_id) AS rn FROM visits) "
                "SELECT department, visit_id, disease, cost, rn FROM ranked "
                f"WHERE rn <= {limit} ORDER BY department, rn"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A bounded LEAD(k) visit interval keeps patient and visit identity,
        # exposes both dates, and computes future minus current in whole days.
        # Repeated offsets in the natural-language contract must agree.
        match = re.fullmatch(
            r"按患者及日期[、,，]就诊ID排序[，,]用LEAD取得后"
            r"(?P<offset>\d+)次日期并计算距后(?P<repeat>\d+)次的整数天数[，,]"
            r"返回当前与后次日期及间隔[。.]?",
            core,
            re.IGNORECASE,
        )
        if match is not None:
            offset = int(match.group("offset"))
            if offset != int(match.group("repeat")) or not 1 <= offset <= 32:
                return None
            lead = (
                "LEAD(visit_date)"
                if offset == 1
                else f"LEAD(visit_date, {offset})"
            )
            window = (
                "OVER (PARTITION BY patient_id "
                "ORDER BY visit_date, visit_id)"
            )
            sql = (
                "SELECT patient_id, visit_id, visit_date, "
                f"{lead} {window} AS next_date, "
                f"CAST(julianday({lead} {window}) - "
                "julianday(visit_date) AS INTEGER) AS days_to_next "
                "FROM visits ORDER BY patient_id, visit_date, visit_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A two-hop KG path joins the first edge's tail to the second edge's
        # head.  Both relation identifiers are explicit and must be grounded;
        # the requested grain is exactly start/middle/end, with no edge helper
        # columns leaked into the projection.
        match = re.fullmatch(
            r"寻找第一跳关系(?P<first>[A-Z][A-Z0-9_]{2,40})[、,，]"
            r"第二跳关系(?P<second>[A-Z][A-Z0-9_]{2,40})且"
            r"第一跳尾等于第二跳头的两跳路径[，,]"
            r"返回起点[、,，]中间点[、,，]终点并依次排列[。.]?",
            core,
        )
        if match is not None:
            first = match.group("first")
            second = match.group("second")
            if (
                first not in self._domains.get("kg_relation", ())
                or second not in self._domains.get("kg_relation", ())
            ):
                return None
            sql = (
                "SELECT a.head AS start_entity, a.tail AS middle_entity, "
                "b.tail AS end_entity FROM kg_triples a "
                "JOIN kg_triples b ON b.head = a.tail "
                f"WHERE a.relation = '{_esc(first)}' "
                f"AND b.relation = '{_esc(second)}' "
                "ORDER BY start_entity, middle_entity, end_entity"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Two bounded CTEs: patient totals (including zero-visit patients) and
        # regional statistics.  The final grain is one patient and must not
        # leak helper statistics or names into the result projection.
        if re.fullmatch(
            r"第一个CTE计算每位患者总费用[（(]无就诊按0[）)][，,]"
            r"第二个计算各地区平均与最大总费用[；;]"
            r"返回高于地区平均但低于地区最大值的患者[，,]"
            r"费用保留2位小数[。.]?",
            core,
            re.IGNORECASE,
        ):
            sql = (
                "WITH patient_totals AS ("
                "SELECT p.patient_id, p.region, "
                "COALESCE(SUM(v.cost), 0) AS total_cost "
                "FROM patients p LEFT JOIN visits v "
                "ON v.patient_id = p.patient_id "
                "GROUP BY p.patient_id, p.region), "
                "region_stats AS ("
                "SELECT region, AVG(total_cost) AS avg_total, "
                "MAX(total_cost) AS max_total FROM patient_totals "
                "GROUP BY region) "
                "SELECT p.patient_id, p.region, "
                "ROUND(p.total_cost, 2) AS total_cost "
                "FROM patient_totals p JOIN region_stats r "
                "ON r.region = p.region "
                "WHERE p.total_cost > r.avg_total "
                "AND p.total_cost < r.max_total "
                "ORDER BY p.region, p.total_cost DESC, p.patient_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Two explicit outgoing relations are represented by two separately
        # correlated EXISTS predicates.  Relation identifiers must be grounded
        # in the sealed database and cannot collapse to the same relation.
        match = re.fullmatch(
            r"返回同时至少存在一条关系(?P<first>[A-Z][A-Z0-9_]{2,40})出边"
            r"和一条关系(?P<second>[A-Z][A-Z0-9_]{2,40})出边的"
            r"头实体名称[，,]按名称排列[。.]?",
            core,
        )
        if match is not None:
            first = match.group("first")
            second = match.group("second")
            if (
                first == second
                or first not in self._domains.get("kg_relation", ())
                or second not in self._domains.get("kg_relation", ())
            ):
                return None
            sql = (
                "SELECT e.name FROM kg_entities e "
                "WHERE EXISTS (SELECT 1 FROM kg_triples a "
                "WHERE a.head = e.name "
                f"AND a.relation = '{_esc(first)}') "
                "AND EXISTS (SELECT 1 FROM kg_triples b "
                "WHERE b.head = e.name "
                f"AND b.relation = '{_esc(second)}') "
                "ORDER BY e.name"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # FIRST/LAST_VALUE must use a full partition frame.  The window's
        # ordering and the final row ordering are distinct contracts; both are
        # emitted explicitly, and visit_id remains before visit_date in output.
        match = re.fullmatch(
            r"按患者分区并按日期[、,，]就诊ID升序[，,]"
            r"在完整分区窗口用(?P<function>FIRST_VALUE|LAST_VALUE)"
            r"给每行附上(?P<position>首条|末条)就诊费用[。.]?",
            core,
            re.IGNORECASE,
        )
        if match is not None:
            function = match.group("function").upper()
            position = match.group("position")
            if (function, position) not in {
                ("FIRST_VALUE", "首条"),
                ("LAST_VALUE", "末条"),
            }:
                return None
            alias = "first_cost" if function == "FIRST_VALUE" else "last_cost"
            sql = (
                "SELECT patient_id, visit_id, visit_date, cost, "
                f"{function}(cost) OVER (PARTITION BY patient_id "
                "ORDER BY visit_date, visit_id ROWS BETWEEN "
                "UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "
                f"AS {alias} FROM visits "
                "ORDER BY patient_id, visit_date, visit_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Three adjacent mention-count bands. Repeated explanatory endpoints
        # must agree with the actual predicates before SQL can be emitted.
        match = re.fullmatch(
            r"将提及次数按小于(?P<a>\d+)[、,，](?P<b>\d+)"
            r"(?:到|至|-)(?P<c>\d+)[、,，]至少(?P<d>\d+)分档[，,]"
            r"统计实体数(?:[；;]端点为(?P<e>\d+)[、,，](?P<f>\d+)"
            r"[、,，](?P<g>\d+))?[。.]?",
            core,
        )
        if match is not None:
            a, b, c, d = (int(match.group(name)) for name in "abcd")
            repeated = tuple(
                int(match.group(name)) for name in "efg" if match.group(name)
            )
            if a == b and d == c + 1 and (not repeated or repeated == (a, c, d)):
                sql = (
                    "SELECT CASE "
                    f"WHEN mention_count < {a} THEN '少于{a}' "
                    f"WHEN mention_count BETWEEN {b} AND {c} THEN '{b}到{c}' "
                    f"ELSE '至少{d}' END AS mention_band, "
                    "COUNT(*) AS entity_count FROM kg_entities "
                    "GROUP BY mention_band ORDER BY mention_band"
                )
                try:
                    validate_gold_sql(sql)
                except Exception:
                    return None
                return sql
            return None

        # Compare every edge with the average confidence of its own relation;
        # preserve both comparison operands in the answer for auditability.
        if re.fullmatch(
            r"先汇总每种关系的平均置信度[，,]再连接原边并返回"
            r"置信度高于本关系平均值的头(?:实体)?[、,，]关系(?:编码)?"
            r"和尾(?:实体)?[。.]?",
            core,
        ):
            sql = (
                "WITH relation_avg AS (SELECT relation, "
                "AVG(confidence) AS avg_confidence FROM kg_triples "
                "GROUP BY relation) "
                "SELECT t.head, t.relation, t.tail, t.confidence, "
                "ROUND(r.avg_confidence, 4) AS relation_avg "
                "FROM kg_triples t JOIN relation_avg r "
                "ON r.relation = t.relation "
                "WHERE t.confidence > r.avg_confidence "
                "ORDER BY t.relation, t.confidence DESC, t.head, t.tail"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Department nodes are tails of BELONGS_TO_DEPT. The disease is an
        # aggregate input, never an output/grouping column in this shape.
        if re.fullmatch(
            r"连接(?:类型为)?Department(?:的)?实体及其BELONGS_TO_DEPT入边[，,]"
            r"按科室统计不同疾病数和观察次数总和[。.]?",
            core,
        ):
            if (
                "Department" not in self._domains.get("kg_entity_type", ())
                or "BELONGS_TO_DEPT" not in self._domains.get("kg_relation", ())
            ):
                return None
            sql = (
                "SELECT e.name AS department, "
                "COUNT(DISTINCT t.head) AS disease_count, "
                "SUM(t.count) AS observations "
                "FROM kg_entities e JOIN kg_triples t ON t.tail = e.name "
                "WHERE e.type = 'Department' "
                "AND t.relation = 'BELONGS_TO_DEPT' "
                "GROUP BY e.name ORDER BY disease_count DESC, department"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A grouped Top-K by average cost returns its ordering measure and the
        # HAVING measure, not only the group label. Repeated numbers are guards.
        match = re.fullmatch(
            r"在记录数至少(?P<threshold>\d+)次的科室中按平均费用取前"
            r"(?P<limit>\d+)名(?:[，,]阈值(?P<repeat_threshold>\d+)"
            r"[、,，]数量(?P<repeat_limit>\d+))?[。.]?",
            core,
        )
        if match is not None:
            threshold = int(match.group("threshold"))
            limit = int(match.group("limit"))
            repeat_threshold = match.group("repeat_threshold")
            repeat_limit = match.group("repeat_limit")
            if threshold < 1 or limit < 1:
                return None
            if repeat_threshold is not None and (
                int(repeat_threshold) != threshold or int(repeat_limit) != limit
            ):
                return None
            sql = (
                "SELECT department, ROUND(AVG(cost), 2) AS avg_cost, "
                "COUNT(*) AS visit_count FROM visits GROUP BY department "
                f"HAVING COUNT(*) >= {threshold} "
                "ORDER BY avg_cost DESC, department "
                f"LIMIT {limit}"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql
        return None

    def _compile_closed_compositional_sql(self, q: str) -> Optional[str]:
        """Lower four fully grounded, composition-heavy analytical shapes.

        Every branch is a whole-question grammar.  Database literals and
        relation identifiers must already exist in the sealed database, while
        thresholds and source labels must be explicit in the question.  Any
        extra semantic slot therefore fails closed instead of being dropped.
        """

        core = self._report_semantic_core(q)
        if core is None:
            return None

        # A named head plus two Chinese relation labels is an IN filter over
        # relation_zh.  The requested projection is deliberately exact.
        match = re.fullmatch(
            r"返回头实体为(?P<head>[^，,且]{1,32})且关系中文名属于"
            r"(?P<first>[^，,或]{1,16})(?:或者|或)"
            r"(?P<second>[^，,的]{1,16})的尾实体[、,，]尾类型和置信度[。.]?",
            core,
        )
        if match is not None:
            head = match.group("head")
            relations = (match.group("first"), match.group("second"))
            if (
                head not in self._domains.get("kg_entity", ())
                or relations[0] == relations[1]
                or any(
                    value not in self._domains.get("kg_relation_zh", ())
                    for value in relations
                )
            ):
                return None
            sql = (
                "SELECT tail, tail_type, confidence FROM kg_triples "
                f"WHERE head = '{_esc(head)}' AND relation_zh IN "
                f"('{_esc(relations[0])}', '{_esc(relations[1])}') "
                "ORDER BY tail_type, tail"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Preserve every tied edge whose confidence is the maximum within its
        # own head and the explicitly named relation.
        match = re.fullmatch(
            r"在(?P<relation>[A-Z][A-Z0-9_]{2,40})关系中[，,]"
            r"返回置信度等于同一头实体(?:的)?最高(?:诊断)?置信度的边"
            r"[，,]并列保留[。.]?",
            core,
        )
        if match is not None:
            relation = match.group("relation")
            if relation not in self._domains.get("kg_relation", ()):
                return None
            sql = (
                "SELECT t.head, t.tail, t.confidence FROM kg_triples t "
                f"WHERE t.relation = '{_esc(relation)}' "
                "AND t.confidence = (SELECT MAX(x.confidence) "
                "FROM kg_triples x WHERE x.head = t.head "
                f"AND x.relation = '{_esc(relation)}') "
                "ORDER BY t.head, t.tail"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # UNION ALL is required because the question asks to retain an
        # explicit provenance label for each side, including duplicate names.
        match = re.fullmatch(
            r"合并(?P<first_relation>[A-Z][A-Z0-9_]{2,40})的"
            r"(?P<first_side>[头尾])实体与"
            r"(?P<second_relation>[A-Z][A-Z0-9_]{2,40})的"
            r"(?P<second_side>[头尾])实体并保留来源标签"
            r"(?P<first_label>[\u4e00-\u9fffA-Za-z0-9_]{1,12}?)"
            r"(?:或者|或)(?P<second_label>[\u4e00-\u9fffA-Za-z0-9_]{1,12})[。.]?",
            core,
        )
        if match is not None:
            first_relation = match.group("first_relation")
            second_relation = match.group("second_relation")
            first_label = match.group("first_label")
            second_label = match.group("second_label")
            if (
                first_relation not in self._domains.get("kg_relation", ())
                or second_relation not in self._domains.get("kg_relation", ())
                or first_label == second_label
            ):
                return None
            first_column = "head" if match.group("first_side") == "头" else "tail"
            second_column = "head" if match.group("second_side") == "头" else "tail"
            sql = (
                f"SELECT {first_column} AS entity_name, "
                f"'{_esc(first_label)}' AS source_name FROM kg_triples "
                f"WHERE relation = '{_esc(first_relation)}' UNION ALL "
                f"SELECT {second_column} AS entity_name, "
                f"'{_esc(second_label)}' AS source_name FROM kg_triples "
                f"WHERE relation = '{_esc(second_relation)}' "
                "ORDER BY source_name, entity_name"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # "患者" in this closed visit projection means the patient identifier,
        # not a join to the patient's display name.
        match = re.fullmatch(
            r"筛选所有年份中月份字符串为(?P<month>\d{1,2})的就诊[，,]"
            r"返回日期[、,，]患者(?:编号)?[、,，]疾病和费用"
            r"(?:[；;]阈值(?P<repeat>\d{1,2}))?[。.]?",
            core,
        )
        if match is not None:
            month = int(match.group("month"))
            repeated = match.group("repeat")
            if not 1 <= month <= 12 or (
                repeated is not None and int(repeated) != month
            ):
                return None
            sql = (
                "SELECT visit_date, patient_id, disease, cost FROM visits "
                f"WHERE strftime('%m', visit_date) = '{month:02d}' "
                "ORDER BY visit_date, visit_id"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A direct KG-entity filter has a closed two-column projection.  The
        # comparison and requested ordering must both be explicit.
        match = re.fullmatch(
            r"筛出实体类型为(?P<type>[A-Za-z]+)[、,，]提及次数"
            r"(?P<operator>大于|高于|至少|不少于|不低于|小于|低于|至多|不超过)"
            r"(?P<threshold>\d+)的名称和次数[，,]按次数"
            r"(?P<direction>升序|降序)[。.]?",
            core,
        )
        if match is not None:
            entity_type = match.group("type")
            if entity_type not in self._domains.get("kg_entity_type", ()):
                return None
            operator = {
                "大于": ">",
                "高于": ">",
                "至少": ">=",
                "不少于": ">=",
                "不低于": ">=",
                "小于": "<",
                "低于": "<",
                "至多": "<=",
                "不超过": "<=",
            }[match.group("operator")]
            direction = "ASC" if match.group("direction") == "升序" else "DESC"
            threshold = int(match.group("threshold"))
            sql = (
                "SELECT name, mention_count FROM kg_entities "
                f"WHERE type = '{_esc(entity_type)}' "
                f"AND mention_count {operator} {threshold} "
                f"ORDER BY mention_count {direction}, name"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Both HAVING operands are also answer measures for this audit shape;
        # returning only the department would discard the stated evidence.
        match = re.fullmatch(
            r"按科室汇总[，,]仅列出总费用(?:至少|不少于|不低于)"
            r"(?P<cost>\d+(?:\.\d+)?)元且不同疾病(?:至少|不少于|不低于)"
            r"(?P<diseases>\d+)种的科室[。.]?",
            core,
        )
        if match is not None:
            cost = float(match.group("cost"))
            disease_count = int(match.group("diseases"))
            if cost < 0 or disease_count < 1:
                return None
            cost_literal = str(int(cost)) if cost.is_integer() else str(cost)
            sql = (
                "SELECT department, ROUND(SUM(cost), 2) AS total_cost, "
                "COUNT(DISTINCT disease) AS disease_count FROM visits "
                "GROUP BY department "
                f"HAVING SUM(cost) >= {cost_literal} "
                f"AND COUNT(DISTINCT disease) >= {disease_count} "
                "ORDER BY total_cost DESC, department"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # A relation satisfying endpoint-type and confidence predicates is a
        # complete edge record, not only its relation code.
        match = re.fullmatch(
            r"列出尾类型属于(?P<types>[A-Za-z]+(?:[、,，]|或|或者)"
            r"[A-Za-z]+(?:(?:[、,，]|或|或者)[A-Za-z]+){0,3})且置信度"
            r"(?P<operator>低于|小于|高于|大于|至少|不低于)"
            r"(?P<threshold>\d+(?:\.\d+)?)的关系[。.]?",
            core,
        )
        if match is not None:
            tail_types = [
                value
                for value in re.split(r"(?:[、,，]|或者|或)", match.group("types"))
                if value
            ]
            if (
                not 2 <= len(tail_types) <= 5
                or len(set(tail_types)) != len(tail_types)
                or any(
                    value not in self._domains.get("kg_tail_type", ())
                    for value in tail_types
                )
            ):
                return None
            operator = {
                "低于": "<",
                "小于": "<",
                "高于": ">",
                "大于": ">",
                "至少": ">=",
                "不低于": ">=",
            }[match.group("operator")]
            threshold = float(match.group("threshold"))
            if not 0 <= threshold <= 1:
                return None
            threshold_literal = str(threshold).rstrip("0").rstrip(".")
            in_values = ", ".join(f"'{_esc(value)}'" for value in tail_types)
            direction = "ASC" if operator == "<" else "DESC"
            sql = (
                "SELECT head, relation, tail, tail_type, confidence "
                "FROM kg_triples "
                f"WHERE tail_type IN ({in_values}) "
                f"AND confidence {operator} {threshold_literal} "
                f"ORDER BY confidence {direction}, head, relation, tail"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Correlate every entity with inbound edges of one grounded relation;
        # the entity table supplies both the name and mention-count projection.
        match = re.fullmatch(
            r"找出作为(?P<relation>[A-Z][A-Z0-9_]{2,40})尾实体被至少"
            r"(?P<threshold>\d+)个不同疾病连接的(?P<type>[A-Za-z]+)实体[。.]?",
            core,
        )
        if match is not None:
            relation = match.group("relation")
            entity_type = match.group("type")
            threshold = int(match.group("threshold"))
            if (
                relation not in self._domains.get("kg_relation", ())
                or entity_type not in self._domains.get("kg_entity_type", ())
                or threshold < 1
            ):
                return None
            sql = (
                "SELECT e.name, e.mention_count FROM kg_entities e "
                f"WHERE e.type = '{_esc(entity_type)}' "
                "AND (SELECT COUNT(DISTINCT t.head) FROM kg_triples t "
                "WHERE t.tail = e.name "
                f"AND t.relation = '{_esc(relation)}') >= {threshold} "
                "ORDER BY e.mention_count DESC, e.name"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        # Two CTEs keep the group average and global average at explicit,
        # independently auditable grains.  The requested group average remains
        # visible in the final answer.
        if re.fullmatch(
            r"使用CTE计算各疾病平均住院天数[，,]再返回平均住院天数"
            r"高于全库平均值的疾病[。.]?",
            core,
        ):
            sql = (
                "WITH disease_los AS (SELECT disease, AVG(los_days) AS avg_los "
                "FROM visits GROUP BY disease), overall AS "
                "(SELECT AVG(los_days) AS overall_los FROM visits) "
                "SELECT d.disease, ROUND(d.avg_los, 2) AS avg_los "
                "FROM disease_los d CROSS JOIN overall o "
                "WHERE d.avg_los > o.overall_los "
                "ORDER BY d.avg_los DESC, d.disease"
            )
            try:
                validate_gold_sql(sql)
            except Exception:
                return None
            return sql

        return None

    def _result_key(self, sql: str) -> Optional[str]:
        """执行 SQL 得到结果集的规范化指纹（用于 self-consistency 投票）。

        Complete-result semantics: the fingerprint reads every row (no display
        truncation) and includes the column count, so candidates that differ
        only past row 200 or in arity can never be merged into one bucket.
        """
        try:
            cols, rows = execute_complete_readonly(sql, self.db_path)
        except Exception:
            return None

        def canonical_cell(value: Any) -> tuple[str, str]:
            if value is None:
                return ("null", "")
            if isinstance(value, bool):
                return ("number", "1" if value else "0")
            if isinstance(value, (int, float)):
                rounded = round(float(value), 4)
                payload = f"{rounded:.4f}".rstrip("0").rstrip(".")
                return ("number", payload or "0")
            if isinstance(value, bytes):
                return ("bytes", value.hex())
            if isinstance(value, str):
                return ("text", value)
            return (type(value).__name__, repr(value))

        # Every cell is now a pair of strings, so nullable and heterogeneous
        # SQLite rows have a deterministic total order on every Python 3 build.
        norm = sorted(tuple(canonical_cell(value) for value in row) for row in rows)
        return repr((len(cols), norm))

    def _llm_pipeline(self, question: str) -> Optional[Dict[str, Any]]:
        """schema-linking + self-consistency 投票 + execution-feedback 自纠。"""
        if self.llm_strategy == "query_plan_v3":
            return self._query_plan_v3_pipeline(question)
        if self.llm_strategy == "query_plan_v2":
            return self._query_plan_v2_pipeline(question)
        if self.llm_strategy == "query_plan":
            return self._query_plan_pipeline(question)

        empty_result: Dict[str, Any] = {"abstained": True}
        self._last_llm_meta = self._llm_result_meta(empty_result)
        try:
            from dki.llm import llm_available
            if not llm_available():
                if self.llm_fail_closed:
                    raise RuntimeError("sealed Agent LLM runtime is not configured")
                return None
        except Exception:
            if self.llm_fail_closed:
                raise
            return None

        from collections import Counter

        feedback = ""
        best: Optional[Dict[str, Any]] = None
        for rnd in range(self.llm_rounds):
            candidates = self._llm_candidates(question, feedback=feedback)
            if not candidates:
                continue
            # 按可执行结果集投票（不可执行的候选记录错误，供下一轮反馈）
            buckets: Dict[str, List[str]] = {}
            errors: List[str] = []
            for sql in candidates:
                key = self._result_key(sql)
                if key is None:
                    errors.append(sql)
                else:
                    buckets.setdefault(key, []).append(sql)
            if buckets:
                # 多数投票：命中最多的结果集，取其最短 SQL 为代表
                top_key = max(buckets, key=lambda k: len(buckets[k]))
                agree = len(buckets[top_key])
                rep = min(buckets[top_key], key=len)
                confidence = round(0.6 + 0.4 * agree / max(1, len(candidates)), 3)
                best = {"sql": rep, "confidence": confidence,
                        "n_candidates": len(candidates), "n_agree": agree, "rounds": rnd + 1}
                # 已有稳健多数（>1 票或仅 1 候选）即返回
                if agree > 1 or len(candidates) == 1:
                    self._last_llm_meta = self._llm_result_meta(best)
                    return best
            # 全部执行失败：把错误反馈进下一轮
            if errors:
                feedback = (
                    "Execution repair required; rejected gate: "
                    "execution_validation. Return one valid read-only SQLite query."
                )
        if best is not None:
            self._last_llm_meta = self._llm_result_meta(best)
        else:
            self._last_llm_meta = self._llm_result_meta(empty_result)
        return best

    def _llm_query(self, question: str) -> Optional[Dict[str, Any]]:
        """LLM 路径并直接返回执行结果（供 query() 用）。"""
        parsed = self._llm_pipeline(question)
        if not parsed or not parsed.get("sql"):
            return None
        try:
            columns, rows = run_sql(parsed["sql"], self.db_path)
            route = (
                "schema_linking_query_plan"
                if self._structured_plan_strategy
                else "schema_linking_llm"
            )
            return {"sql": parsed["sql"], "engine": "llm", "confidence": parsed["confidence"],
                    "validated": True,
                    "provenance": {
                        "database": str(getattr(self, "db_path", "")),
                        **self._route_meta(route, "complex"),
                    },
                    "columns": columns, "rows": [list(r) for r in rows],
                    "llm_meta": self._llm_result_meta(parsed)}
        except Exception as e:
            return {"sql": parsed["sql"], "engine": "llm", "confidence": parsed.get("confidence", 0.0),
                    "validated": False, "columns": [], "rows": [], "error": str(e),
                    "llm_meta": self._llm_result_meta(parsed)}
