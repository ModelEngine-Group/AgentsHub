"""可选质量模式的医疗 NER 规则校准。

规则只修正神经模型容易混淆、但医学语义稳定的两类边界：
- 升高/控制不良的血压、血糖表述属于 Disease finding；
- BP/FBG 数值、CT、抽血/血检、拍片等显式检查表述属于 Exam。

该模块不接入默认 dict、BERT 或 RBT6 性能路径；仅由显式 quality 模式调用。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


_DISEASE_FINDINGS = (
    re.compile(r"血压(?:一直|老是|又|也|总是|始终)?(?:有点|偏|较)?高(?:了|着|老不好)?"),
    re.compile(r"高压(?:都到|达到|约|为)?\s*\d+(?:多)?"),
    re.compile(r"BP(?:也)?\s*(?:↑|升高|偏高)"),
    re.compile(r"血糖(?:也|一直|总是)?(?:有点|偏|较)?高"),
    re.compile(r"血糖(?:也)?(?:不好控制|控制不住|控不好|控制不良|控制不佳)"),
    re.compile(r"血糖问题"),
)

_EXAM_FINDINGS = (
    re.compile(r"BP\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"FBG\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"(?:血压|血糖)(?:空腹)?\s*\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?"),
    re.compile(r"(?:肺部|胸部|头颅|颅脑)?(?:CT|MRI)(?:平扫|检查|扫描)?", re.IGNORECASE),
    re.compile(r"(?:抽血化验|血检|指尖血糖|眼底照相|尿常规|血常规)"),
    re.compile(r"(?:拍了?|照了?)(?:个|下)?[^，。；,;]{0,4}(?:片子|片)"),
)

_EXAM_VERBS = {"查", "验", "测", "量", "做", "拍"}

_EXAM_ABBREVIATIONS = re.compile(
    r"(?<![A-Za-z0-9])(?:PSG|ANA|ECG|EKG|EEG|MRI|CT|Hb|MCV|TSH|T3|T4|FBG|BP|"
    r"CRP|ESR|ALT|AST|RF|B超|彩超)(?![A-Za-z])",
    re.IGNORECASE,
)
_TREATMENT_TERMS = re.compile(
    r"(?:CBT-I|认知行为治疗|局部热敷|补液盐水|多饮温水|温盐水漱口|硝酸甘油舌下含服|"
    r"他汀类药物治疗|饮食控制|适度运动(?:锻炼)?|减轻体重|休息制动|康复训练|"
    r"物理治疗|手术治疗|放疗|化疗)",
    re.IGNORECASE,
)
_DRUG_ABBREVIATIONS = re.compile(r"(?<![A-Za-z])NSAIDs?(?![A-Za-z])", re.IGNORECASE)


def _contains_cjk(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def _plausible_quality_entity(ent: Dict) -> bool:
    """只过滤明显 BIO 碎片，不按词表压制开放实体。"""
    mention = str(ent.get("mention") or ent.get("name") or "").strip()
    if not mention:
        return False
    if _contains_cjk(mention) and len(mention) == 1:
        return False
    if ent.get("type") == "Symptom" and mention in {"老是", "有点", "一下", "这个", "问题"}:
        return False
    if not _contains_cjk(mention) and len(mention) < 2:
        return False
    return True


def _dedupe_entities(entities: List[Dict]) -> List[Dict]:
    """按同类型跨度合并：优先 quality 模型/规则，BERT 只补 quality 漏项。"""
    result: List[Dict] = []
    for ent in sorted(entities, key=lambda item: (
        int(item.get("span", [10**9, 10**9])[0]),
        -int(item.get("span", [0, 0])[1]) + int(item.get("span", [0, 0])[0]),
        item.get("type", ""),
    )):
        span = ent.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            result.append(ent)
            continue
        interval = (int(span[0]), int(span[1]))
        if any(old.get("type") == ent.get("type") and
               isinstance(old.get("span"), (list, tuple)) and len(old["span"]) == 2 and
               _overlap(interval, (int(old["span"][0]), int(old["span"][1])))
               for old in result):
            continue
        result.append(ent)
    return result


def merge_quality_recall(text: str, quality: List[Dict], baseline: List[Dict]) -> List[Dict]:
    """高召回模式：quality 为主，过滤明显碎片后用原 BERT 补充互补实体。"""
    primary = [dict(ent) for ent in quality if _plausible_quality_entity(ent)]
    additions = []
    quality_types = {ent.get("type") for ent in primary}
    # 原 BERT 在开放语料的 Disease/Drug 补充误报较多；开发集实测互补命中主要来自
    # Symptom/Exam/Treatment，故只允许这些类型补召回。若 quality 已识别 Treatment，
    # 不再用 baseline 扩张该低频类型。
    rescue_types = {"Symptom", "Exam"}
    if "Treatment" not in quality_types:
        rescue_types.add("Treatment")
    for ent in baseline:
        if ent.get("type") not in rescue_types:
            continue
        if not _plausible_quality_entity(ent):
            continue
        span = ent.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        interval = (int(span[0]), int(span[1]))
        if any(old.get("type") == ent.get("type") and
               isinstance(old.get("span"), (list, tuple)) and len(old["span"]) == 2 and
               _overlap(interval, (int(old["span"][0]), int(old["span"][1])))
               for old in primary):
            continue
        item = dict(ent)
        item["source"] = "quality-baseline-rescue"
        additions.append(item)

    # 稳定的英文检查/治疗/药物缩写可由规则补齐，避免中文弱监督模型对拉丁 token 漏识别。
    rules = []
    for pattern, entity_type in (
        (_EXAM_ABBREVIATIONS, "Exam"),
        (_TREATMENT_TERMS, "Treatment"),
        (_DRUG_ABBREVIATIONS, "Drug"),
    ):
        rules.extend(_rule_spans(text, (pattern,), entity_type))
    return _dedupe_entities(primary + additions + rules)


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _rule_spans(text: str, patterns, entity_type: str) -> List[dict]:
    spans = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            s, e = match.span()
            spans.append({
                "name": match.group(0),
                "mention": match.group(0),
                "type": entity_type,
                "span": [s, e],
                "source": "quality-rule",
                "confidence": 0.95,
            })
    spans.sort(key=lambda item: (item["span"][0], -(item["span"][1] - item["span"][0])))
    out = []
    for ent in spans:
        if any(ent["type"] == old["type"] and _overlap(tuple(ent["span"]), tuple(old["span"])) for old in out):
            continue
        out.append(ent)
    return out


def calibrate_quality_entities(text: str, entities: List[Dict]) -> List[Dict]:
    """校准 quality 模型输出；输入列表不原地修改。"""
    disease_rules = _rule_spans(text, _DISEASE_FINDINGS, "Disease")
    exam_rules = _rule_spans(text, _EXAM_FINDINGS, "Exam")
    rules = disease_rules + exam_rules

    calibrated = []
    for original in entities:
        ent = dict(original)
        span = ent.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            calibrated.append(ent)
            continue
        interval = (int(span[0]), int(span[1]))
        mention = ent.get("mention") or text[interval[0]:interval[1]]
        # 模型偶尔把检查动作单字单独抽成实体，真正检查对象会由模型或规则保留。
        if ent.get("type") == "Exam" and mention in _EXAM_VERBS:
            continue
        # 升高指标常被口语 OOD 模型误分为 Symptom；用更具体的规则实体替换，
        # 同一模型跨度里同时出现血压/血糖时也能拆成两个实体。
        covering = [r for r in disease_rules if _overlap(interval, tuple(r["span"]))]
        if ent.get("type") == "Symptom" and covering:
            continue
        calibrated.append(ent)

    # 规则实体与已有同类型实体重叠时，保留已有模型实体；不同类型则让规则覆盖
    # 上面的已知混淆类型，避免同一提及重复输出。
    for rule in rules:
        interval = tuple(rule["span"])
        same = [e for e in calibrated if e.get("type") == rule["type"] and
                isinstance(e.get("span"), (list, tuple)) and len(e["span"]) == 2 and
                _overlap(interval, (int(e["span"][0]), int(e["span"][1])))]
        if same:
            continue
        calibrated.append(rule)

    calibrated.sort(key=lambda item: (
        item.get("span", [10**9, 10**9])[0],
        item.get("span", [10**9, 10**9])[1],
        item.get("type", ""),
    ))
    return calibrated
