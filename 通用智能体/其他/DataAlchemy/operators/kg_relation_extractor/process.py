# -*- coding: utf-8 -*-
"""
KGRelationExtractor - 医疗关系抽取算子

基于模式匹配 + 实体类型对推理，从医疗文本中抽取4类关系：
- 导致 (Disease → Symptom)
- 治疗 (Drug → Disease)
- 用于 (Drug → Symptom)
- 禁忌 (Drug → Disease)

架构：
- Layer 4（公开接口）：validate / process / get_summary / get_schema
- Layer 3（抽取流水线）：PIPELINE_ORDER → STEP_HANDLERS → _run_step_safe
- Layer 2（关系抽取器）：_extract_by_pattern / _extract_by_cooccurrence / _merge_relations
- Layer 1（基础设施）：异常类 / 触发词规则 / 输入标准化 / 输出构建

版本: 1.0.0
"""

import sys
import os
import re
import logging
from typing import Any, Dict, List, Tuple, Optional, Set

# 日志配置
logger = logging.getLogger("KGRelationExtractor")

# 路径配置（容器中自动跳过，不影响部署）
try:
    _OP_DIR = os.path.dirname(os.path.abspath(__file__))
    if _OP_DIR not in sys.path:
        sys.path.insert(0, _OP_DIR)
except Exception:
    pass


# ============================================================================
# Layer 1: 异常类定义
# ============================================================================

class RelationExtractorError(Exception):
    """关系抽取算子基础异常"""
    pass


class ValidationError(RelationExtractorError):
    """参数校验失败"""
    pass


class ProcessingError(RelationExtractorError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# Layer 1: 关系抽取规则定义
# ============================================================================

# 实体类型映射（中文内部表示 → 英文代码）
ENTITY_TYPES = {
    "疾病": "disease",
    "症状": "symptom",
    "药物": "drug",
    "检查": "examination",
}

# 反向映射
ENTITY_TYPES_CN = {v: k for k, v in ENTITY_TYPES.items()}

# 关系类型定义
RELATION_TYPES = ["导致", "治疗", "用于", "禁忌"]

# 有效实体类型对 → 关系映射（用于默认推理）
VALID_ENTITY_PAIRS: Dict[Tuple[str, str], List[str]] = {
    ("疾病", "症状"): ["导致"],
    ("药物", "疾病"): ["治疗", "禁忌"],
    ("药物", "症状"): ["用于"],
    ("疾病", "疾病"): ["导致"],  # 并发症
}

# 模式触发词规则
# 格式: (触发词模式, 关系类型, head_index, tail_index)
# head_index=0 表示触发词前的实体是head，tail_index=1 表示触发词后的实体是tail
PATTERN_RULES: List[Tuple[str, str, int, int]] = [
    # 导致关系 (Disease → Symptom)
    (r"导致", "导致", 0, 1),
    (r"引起", "导致", 0, 1),
    (r"造成", "导致", 0, 1),
    (r"诱发", "导致", 0, 1),
    (r"引发", "导致", 0, 1),
    (r"致使", "导致", 0, 1),
    (r"出现", "导致", 0, 1),      # "糖尿病出现多饮多尿" → (糖尿病, 导致, 多饮多尿)
    (r"伴有", "导致", 0, 1),      # "高血压伴有头痛"
    (r"伴随", "导致", 0, 1),
    (r"并发", "导致", 0, 1),
    
    # 治疗关系 (Drug → Disease)
    (r"治疗", "治疗", 1, 0),      # "服用二甲双胍治疗糖尿病" → (糖尿病, 治疗, 二甲双胍) → 反转
    (r"控制", "治疗", 1, 0),      # "服用硝苯地平控制血压"
    (r"用于治疗", "治疗", 0, 1),   # "二甲双胍用于治疗糖尿病"
    (r"对.*有效", "治疗", 1, 0),   # "XX药对XX病有效"
    (r"缓解", "治疗", 0, 1),      # "药物缓解症状"
    (r"改善", "治疗", 0, 1),
    (r"抑制", "治疗", 0, 1),
    
    # 用于关系 (Drug → Symptom)
    (r"止痛", "用于", 0, 1),      # "布洛芬止痛" → (布洛芬, 用于, 止痛)
    (r"退热", "用于", 0, 1),
    (r"降温", "用于", 0, 1),
    (r"止咳", "用于", 0, 1),
    (r"化痰", "用于", 0, 1),
    
    # 禁忌关系 (Drug → Disease)
    (r"禁忌", "禁忌", 0, 1),
    (r"禁用", "禁忌", 0, 1),
    (r"慎用", "禁忌", 0, 1),
    (r"不适用于", "禁忌", 0, 1),
    (r"忌用", "禁忌", 0, 1),
    (r"不宜用于", "禁忌", 0, 1),
]

# 句末标点（用于分割句子）
SENTENCE_BOUNDARY = re.compile(r"[。！？\n\r]+")


def _split_sentences_with_pos(text: str) -> List[Tuple[str, int]]:
    """分割句子并返回每个句子在原文中的起始位置"""
    if not text:
        return []
    sentences: List[Tuple[str, int]] = []
    pos = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        seg = text[pos:match.start()].strip()
        if seg:
            sentences.append((seg, pos + text[pos:match.start()].find(seg)))
        pos = match.end()
    remaining = text[pos:].strip()
    if remaining:
        sentences.append((remaining, pos))
    # 如果一个句子都没有（无标点），整段作为一个句子
    if not sentences and text.strip():
        sentences.append((text.strip(), 0))
    return sentences


def _find_entity_in_text(text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将实体按类型分组
    
    Returns:
        {entity_type_cn: [entity_info, ...]}
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ent in entities:
        t = ent.get("type", "")
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(ent)
    return grouped


def _split_sentences(text: str) -> List[str]:
    """按句末标点分割文本为句子列表"""
    if not text:
        return []
    sentences = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _find_entities_in_sentence(sentence: str,
                                all_entities: List[Dict[str, Any]],
                                sent_start: int = 0) -> List[Dict[str, Any]]:
    """找出出现在某句子范围内的实体"""
    sent_end = sent_start + len(sentence)
    
    # 用位置过滤（如果有位置信息）
    result = []
    for ent in all_entities:
        ent_start = ent.get("start", -1)
        ent_end = ent.get("end", -1)
        if ent_start >= sent_start and ent_end <= sent_end:
            result.append(ent)
    
    # 如果都没有位置信息，用名称匹配
    if not result:
        for ent in all_entities:
            name = ent.get("name", "")
            if name and name in sentence:
                result.append(ent)
    return result


# ============================================================================
# Layer 2: 关系抽取器
# ============================================================================

def _extract_by_pattern(text: str,
                         entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    基于模式触发词的关系抽取
    
    对于每条规则，在文本中找到触发词的位置，然后找触发词前后最近的实体
    """
    relations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    
    for pattern, rel_type, head_idx, tail_idx in PATTERN_RULES:
        for match in re.finditer(pattern, text):
            match_pos = match.start()
            match_end = match.end()
            
            # 找触发词之前最近的实体
            prev_ent = None
            prev_dist = float('inf')
            for ent in entities:
                ent_start = ent.get("start", -1)
                ent_end = ent.get("end", -1)
                if ent_start >= 0 and ent_end >= 0:
                    if ent_end <= match_pos:
                        dist = match_pos - ent_end
                        if dist < prev_dist:
                            prev_ent = ent
                            prev_dist = dist
            
            # 找触发词之后最近的实体
            next_ent = None
            next_dist = float('inf')
            for ent in entities:
                ent_start = ent.get("start", -1)
                if ent_start >= match_end:
                    ent_end = ent.get("end", -1)
                    dist = ent_start - match_end
                    if dist < next_dist:
                        next_ent = ent
                        next_dist = dist
            
            if prev_ent is None or next_ent is None:
                continue
            
            # 距离阈值：触发词和实体之间不超过15个字符
            if prev_dist > 15 or next_dist > 15:
                continue
            
            # 根据 head_idx/tail_idx 确定 head 和 tail
            candidates = [prev_ent, next_ent]
            head = candidates[head_idx]
            tail = candidates[tail_idx]
            
            # 实体类型校验
            head_type = head.get("type", "")
            tail_type = tail.get("type", "")
            
            # 如果结果需要反转（比如"治疗"模式中，head=疾病, tail=药物时）
            if rel_type == "治疗":
                # 如果head是药物、tail是疾病 → 交换
                if head_type == "药物" and tail_type == "疾病":
                    key = (head["name"], rel_type, tail["name"])
                    if key not in seen:
                        relations.append({
                            "head": head["name"],
                            "head_type": head_type,
                            "relation": rel_type,
                            "tail": tail["name"],
                            "tail_type": tail_type,
                            "confidence": "high",
                            "evidence": match.group(),
                        })
                        seen.add(key)
                    continue
                # 如果head是疾病、tail是药物 → 正常
                elif head_type == "疾病" and tail_type == "药物":
                    key = (tail["name"], rel_type, head["name"])
                    if key not in seen:
                        relations.append({
                            "head": tail["name"],  # 药物在head位置
                            "head_type": tail_type,
                            "relation": rel_type,
                            "tail": head["name"],  # 疾病在tail位置
                            "tail_type": head_type,
                            "confidence": "high",
                            "evidence": match.group(),
                        })
                        seen.add(key)
                    continue
            
            # 通用情况：用 head_idx/tail_idx 决定
            key = (head["name"], rel_type, tail["name"])
            if key not in seen:
                relations.append({
                    "head": head["name"],
                    "head_type": head_type,
                    "relation": rel_type,
                    "tail": tail["name"],
                    "tail_type": tail_type,
                    "confidence": "high",
                    "evidence": match.group(),
                })
                seen.add(key)
    
    return relations


def _extract_by_cooccurrence(text: str,
                              entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    基于共现的关系推理
    
    对于同一句子中的实体对，根据实体类型从 VALID_ENTITY_PAIRS 推断可能的关系
    和 pattern 抽取的结果合并，但如果已被 pattern 覆盖则跳过
    """
    relations: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()
    
    # 先收集pattern已覆盖的关系
    # (这个函数会在merge时处理去重，这里只做共现推理)
    
    sentences = _split_sentences_with_pos(text)
    
    for sentence, sent_start in sentences:
        sent_entities = _find_entities_in_sentence(sentence, entities, sent_start)
        
        # 取同一句子中的实体对
        for i, ent_a in enumerate(sent_entities):
            for j, ent_b in enumerate(sent_entities):
                if i >= j:
                    continue
                
                type_a = ent_a.get("type", "")
                type_b = ent_b.get("type", "")
                name_a = ent_a.get("name", "")
                name_b = ent_b.get("name", "")
                
                # 检查 (type_a, type_b) 方向
                possible_rels = VALID_ENTITY_PAIRS.get((type_a, type_b), [])
                head_name, head_type = name_a, type_a
                tail_name, tail_type = name_b, type_b
                
                # 如果正向无匹配，尝试反向 (type_b, type_a)
                if not possible_rels:
                    possible_rels = VALID_ENTITY_PAIRS.get((type_b, type_a), [])
                    head_name, head_type = name_b, type_b
                    tail_name, tail_type = name_a, type_a
                
                if not possible_rels:
                    continue
                
                for rel in possible_rels:
                    key = (head_name, rel, tail_name)
                    if key not in seen_pairs:
                        relations.append({
                            "head": head_name,
                            "head_type": head_type,
                            "relation": rel,
                            "tail": tail_name,
                            "tail_type": tail_type,
                            "confidence": "medium",
                            "evidence": "(共现推理)",
                        })
                        seen_pairs.add(key)
                
    return relations


def _merge_relations(pattern_rels: List[Dict[str, Any]],
                     cooccur_rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并两种来源的关系，去重
    
    优先级：pattern 结果 > co-occurrence 结果
    相同 (head, relation, tail) 时保留 pattern 结果
    """
    seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    
    # 先加入 co-occurrence（低优先级）
    for rel in cooccur_rels:
        key = (rel["head"], rel["relation"], rel["tail"])
        seen[key] = rel
    
    # 再覆盖 pattern（高优先级）
    for rel in pattern_rels:
        key = (rel["head"], rel["relation"], rel["tail"])
        seen[key] = rel
    
    return list(seen.values())


# ============================================================================
# Layer 3: 抽取流水线引擎
# ============================================================================

PIPELINE_ORDER = [
    "validate_input",       # 1. 校验输入
    "prepare_entities",     # 2. 准备实体数据
    "extract_relations",    # 3. 关系抽取（pattern + co-occurrence）
    "build_output",         # 4. 构建输出
]

STEP_HANDLERS = {
    "validate_input":      "_step_validate_input",
    "prepare_entities":    "_step_prepare_entities",
    "extract_relations":   "_step_extract_relations",
    "build_output":        "_step_build_output",
}


# ============================================================================
# Layer 4: 主处理类
# ============================================================================

class KGRelationExtractor:
    """
    医疗关系抽取算子
    
    功能：
    1. 基于模式触发词抽取明确的关系
    2. 基于实体类型对推理可能的关系（共现）
    3. 合并去重，返回结构化关系列表
    
    输入格式：
        {
            "text": "患者糖尿病多年，服用二甲双胍治疗",
            "entities": [
                {"type": "疾病", "name": "糖尿病", "start": 2, "end": 5},
                {"type": "药物", "name": "二甲双胍", "start": 10, "end": 14}
            ]
        }
    
    输出格式：
        {
            "relations": [
                {
                    "head": "糖尿病", "head_type": "疾病",
                    "relation": "治疗",
                    "tail": "二甲双胍", "tail_type": "药物",
                    "confidence": "high",
                    "evidence": "服用二甲双胍治疗"
                }
            ]
        }
    
    处理顺序：validate_input → prepare_entities → extract_relations → build_output
    """

    # =========================================================================
    # 常量定义
    # =========================================================================

    SUPPORTED_INPUT_KEYS = ["text", "entities", "data"]

    # =========================================================================
    # 公开接口
    # =========================================================================

    def validate(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验输入数据
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        if input_data is None:
            errors.append("输入数据为空")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if isinstance(input_data, dict):
            if "text" not in input_data and "data" not in input_data:
                warnings.append(
                    f"输入字典中未找到 text 或 data key，"
                    f"当前keys: {list(input_data.keys())}"
                )
        elif isinstance(input_data, list):
            if not input_data:
                warnings.append("输入列表为空")
        else:
            errors.append(f"不支持的输入类型: {type(input_data).__name__}")
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def process(self, input_data: Any, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        主处理入口
        """
        if params is None:
            params = {}
        
        pipeline_params = dict(params)
        pipeline_params.setdefault("data_key", "data")
        
        return self._run_pipeline(input_data, pipeline_params)

    def get_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取结果摘要
        """
        relations = result.get("relations", [])
        if not relations:
            return {"relation_count": 0, "type_distribution": {}}
        
        type_dist = {}
        for rel in relations:
            r = rel.get("relation", "未知")
            type_dist[r] = type_dist.get(r, 0) + 1
        
        return {
            "relation_count": len(relations),
            "type_distribution": type_dist,
            "high_confidence_count": sum(
                1 for r in relations if r.get("confidence") == "high"
            ),
        }

    def get_schema(self) -> Dict[str, Any]:
        """
        获取算子的输入输出Schema定义
        """
        return {
            "input": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "医疗文本"
                    },
                    "entities": {
                        "type": "array",
                        "description": "识别出的实体列表（来自EntityRecognizer输出）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "name": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"}
                            }
                        }
                    }
                },
                "required": ["text", "entities"]
            },
            "output": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "description": "抽取的关系列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "head": {"type": "string"},
                                "head_type": {"type": "string"},
                                "relation": {"type": "string"},
                                "tail": {"type": "string"},
                                "tail_type": {"type": "string"},
                                "confidence": {"type": "string"},
                                "evidence": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

    # =========================================================================
    # 流水线引擎（内部）
    # =========================================================================

    def _run_pipeline(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        state = {
            "input": input_data,
            "params": params,
            "errors": [],
            "warnings": [],
            "relations": [],
        }
        
        for step_name in PIPELINE_ORDER:
            handler_name = STEP_HANDLERS.get(step_name)
            if handler_name is None:
                state["errors"].append(f"未找到步骤处理器: {step_name}")
                continue
            
            handler = getattr(self, handler_name, None)
            if handler is None:
                state["errors"].append(f"未实现步骤方法: {handler_name}")
                continue
            
            state = self._run_step_safe(state, step_name, handler)
            
            if state.get("_fatal"):
                break
        
        return self._build_final_output(state)

    def _run_step_safe(self, state: Dict[str, Any],
                       step_name: str,
                       handler) -> Dict[str, Any]:
        try:
            logger.debug(f"执行步骤: {step_name}")
            return handler(state)
        except ValidationError as e:
            state["errors"].append(f"[{step_name}] 校验失败: {e}")
            state["_fatal"] = True
            return state
        except ProcessingError as e:
            state["errors"].append(f"[{step_name}] 处理失败: {e}")
            state["_fatal"] = True
            return state
        except Exception as e:
            state["errors"].append(f"[{step_name}] 未知错误: {e}")
            logger.exception(f"步骤 {step_name} 发生未预期异常")
            state["_fatal"] = True
            return state

    # =========================================================================
    # 流水线步骤实现
    # =========================================================================

    def _step_validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        input_data = state["input"]
        
        validation = self.validate(input_data, state["params"])
        state["warnings"].extend(validation.get("warnings", []))
        
        if not validation.get("valid", False):
            state["errors"].extend(validation.get("errors", []))
            state["_fatal"] = True
            return state
        
        return state

    def _step_prepare_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """准备实体数据：从输入中提取文本和实体列表"""
        input_data = state["input"]
        
        text = ""
        entities: List[Dict[str, Any]] = []
        
        if isinstance(input_data, dict):
            text = input_data.get("text", "")
            entities = input_data.get("entities", [])
            
            # 如果 entities 是 dict（比如从EntityRecognizer包装过的输出）
            if isinstance(entities, dict):
                entities = entities.get("entities", entities)
        
        if not text:
            state["warnings"].append("输入文本为空")
        
        if not entities:
            state["warnings"].append("输入实体列表为空，无法进行关系抽取")
        
        state["text"] = text
        state["entities"] = entities
        return state

    def _step_extract_relations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """关系抽取：使用pattern + co-occurrence"""
        text = state.get("text", "")
        entities = state.get("entities", [])
        
        if not text or not entities:
            state["relations"] = []
            return state
        
        # Step 1: 基于模式触发词
        pattern_rels = _extract_by_pattern(text, entities)
        
        # Step 2: 基于共现推理
        cooccur_rels = _extract_by_cooccurrence(text, entities)
        
        # Step 3: 合并
        relations = _merge_relations(pattern_rels, cooccur_rels)
        
        state["relations"] = relations
        return state

    def _step_build_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def _build_final_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        output = {
            "relations": state.get("relations", []),
            "relation_count": len(state.get("relations", [])),
        }
        
        if state.get("errors"):
            output["errors"] = state["errors"]
        if state.get("warnings"):
            output["warnings"] = state["warnings"]
        
        return output
