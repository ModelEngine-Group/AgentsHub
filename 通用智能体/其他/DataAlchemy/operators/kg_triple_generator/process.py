# -*- coding: utf-8 -*-
"""
KGTripleGenerator - 知识图谱三元组生成算子

将实体识别 + 关系抽取的结果组合为标准化三元组：
- 去重（相同 head-relation-tail 只保留一条）
- 冲突检测（相同 head-relation 但不同 tail）
- 上下文约束字段预留
- 实体类型校验

架构：
- Layer 4（公开接口）：validate / process / get_summary / get_schema
- Layer 3（生成流水线）：PIPELINE_ORDER → STEP_HANDLERS → _run_step_safe
- Layer 2（组合引擎）：_build_triples / _detect_conflicts / _deduplicate
- Layer 1（基础设施）：异常类 / Schema校验 / 输入标准化 / 输出构建

版本: 1.0.0
"""

import sys
import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict

# 日志配置
logger = logging.getLogger("KGTripleGenerator")

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

class TripleGeneratorError(Exception):
    """三元组生成算子基础异常"""
    pass


class ValidationError(TripleGeneratorError):
    """参数校验失败"""
    pass


class ProcessingError(TripleGeneratorError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# Layer 1: Schema 定义与校验
# ============================================================================

VALID_ENTITY_TYPES = {"疾病", "症状", "药物", "检查"}
VALID_RELATION_TYPES = {"导致", "治疗", "用于", "禁忌"}

# 关系 → 合法实体类型对
RELATION_ENTITY_CONSTRAINTS: Dict[str, List[Tuple[str, str]]] = {
    "导致": [("疾病", "症状"), ("疾病", "疾病")],
    "治疗": [("药物", "疾病")],
    "用于": [("药物", "症状")],
    "禁忌": [("药物", "疾病")],
}


def _validate_entity_type(entity_type: str) -> bool:
    """校验实体类型是否合法"""
    return entity_type in VALID_ENTITY_TYPES


def _validate_relation_type(relation: str) -> bool:
    """校验关系类型是否合法"""
    return relation in VALID_RELATION_TYPES


def _validate_entity_pair(head_type: str, relation: str, tail_type: str) -> bool:
    """校验实体类型对是否符合关系约束"""
    valid_pairs = RELATION_ENTITY_CONSTRAINTS.get(relation, [])
    return (head_type, tail_type) in valid_pairs


# ============================================================================
# Layer 2: 三元组组合引擎
# ============================================================================

def _build_triples(entities: List[Dict[str, Any]],
                    relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将实体和关系组合为标准化三元组
    
    1. 从relations直接生成标准三元组
    2. 头实体和尾实体必须都在entities中（验证实体存在性）
    3. 实体类型校验
    """
    # 构建实体名称→类型映射
    entity_map: Dict[str, str] = {}
    for ent in entities:
        name = ent.get("name", "")
        etype = ent.get("type", "")
        if name and etype:
            entity_map[name] = etype
    
    triples: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    warnings: List[str] = []
    
    for rel in relations:
        head = rel.get("head", "")
        relation = rel.get("relation", "")
        tail = rel.get("tail", "")
        confidence = rel.get("confidence", "medium")
        evidence = rel.get("evidence", "")
        head_type = rel.get("head_type", entity_map.get(head, ""))
        tail_type = rel.get("tail_type", entity_map.get(tail, ""))
        
        # 跳过无效值
        if not head or not relation or not tail:
            warnings.append(f"跳过无效关系: head={head}, rel={relation}, tail={tail}")
            continue
        
        # 校验关系类型
        if not _validate_relation_type(relation):
            warnings.append(f"跳过非法关系类型: {relation}")
            continue
        
        # 校验实体存在性
        if head not in entity_map:
            warnings.append(f"头实体 '{head}' 不在实体列表中，跳过")
            continue
        if tail not in entity_map:
            warnings.append(f"尾实体 '{tail}' 不在实体列表中，跳过")
            continue
        
        # 类型校验
        actual_head_type = entity_map[head]
        actual_tail_type = entity_map[tail]
        
        type_warning = False
        if head_type and head_type != actual_head_type:
            warnings.append(
                f"头实体类型不匹配: '{head}' 类型声明为 '{head_type}'，"
                f"实际为 '{actual_head_type}'"
            )
            type_warning = True
        if tail_type and tail_type != actual_tail_type:
            warnings.append(
                f"尾实体类型不匹配: '{tail}' 类型声明为 '{tail_type}'，"
                f"实际为 '{actual_tail_type}'"
            )
            type_warning = True
        
        # 实体对关系约束校验
        pair_valid = _validate_entity_pair(actual_head_type, relation, actual_tail_type)
        if not pair_valid:
            warnings.append(
                f"实体类型对不合法: ({actual_head_type})--[{relation}]-->({actual_tail_type}) "
                f"'{head}--{relation}-->{tail}'"
            )
        
        # 去重 key
        triple_key = (head, relation, tail)
        if triple_key in seen:
            continue
        seen.add(triple_key)
        
        # 构建标准三元组
        triple = {
            "head": head,
            "head_type": actual_head_type,
            "relation": relation,
            "tail": tail,
            "tail_type": actual_tail_type,
            "confidence": confidence,
        }
        
        if evidence:
            triple["evidence"] = evidence
        
        # 添加上下文约束字段（预留）
        triple["context_constraints"] = []
        
        # 标记是否违反了实体类型约束
        if not pair_valid:
            triple["_type_warning"] = True
        
        triples.append(triple)
    
    return triples, warnings


def _detect_conflicts(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    检测规则冲突
    
    冲突类型：
    1. 直接冲突：同一 head-relation 对应不同 tail（如 阿司匹林-治疗-糖尿病 vs 阿司匹林-治疗-高血压）
    2. 禁忌冲突：同一药物既治疗又禁忌同一疾病（如 阿司匹林-治疗-胃溃疡 vs 阿司匹林-禁忌-胃溃疡）
    """
    conflicts: List[Dict[str, Any]] = []
    
    # 1. 直接冲突检测
    group_key = lambda t: (t["head"], t["relation"])
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    
    for t in triples:
        if t.get("_type_warning"):
            continue
        groups[group_key(t)].append(t)
    
    for key, group in groups.items():
        if len(group) > 1:
            # 同一个 head-relation 有多个不同的 tail → 冲突
            tails = [t["tail"] for t in group]
            if len(set(tails)) > 1:
                conflicts.append({
                    "type": "direct_conflict",
                    "head": key[0],
                    "relation": key[1],
                    "tails": tails,
                    "description": f"'{key[0]}--{key[1]}' 存在多条不同尾实体: {tails}",
                    "severity": "warning",
                })
    
    # 2. 禁忌冲突检测
    drug_disease_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for t in triples:
        if t.get("_type_warning"):
            continue
        if t["relation"] in ("治疗", "禁忌") and t["head_type"] == "药物":
            drug_disease_map[(t["head"], t["tail"])].append(t["relation"])
    
    for (drug, disease), rels in drug_disease_map.items():
        if "治疗" in rels and "禁忌" in rels:
            conflicts.append({
                "type": "contraindication_conflict",
                "head": drug,
                "relation": "治疗|禁忌",
                "tail": disease,
                "description": f"'{drug}' 同时标注为 '{disease}' 的治疗药物和禁忌药物",
                "severity": "critical",
            })
    
    return conflicts


def _deduplicate(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去重（额外安全保障，_build_triples 已经做了一次去重）
    
    但这里还要按置信度排序保留最优：
    - 同一 head-relation-tail 保留置信度最高的
    - 同一 head-relation-tail 有 evidence 的优先
    """
    seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    
    for t in triples:
        key = (t["head"], t["relation"], t["tail"])
        
        if key not in seen:
            seen[key] = t
        else:
            existing = seen[key]
            # 冲突时保留置信度更高的
            conf_rank = {"high": 3, "medium": 2, "low": 1}
            current_rank = conf_rank.get(t.get("confidence", "low"), 0)
            existing_rank = conf_rank.get(existing.get("confidence", "low"), 0)
            
            if current_rank > existing_rank:
                seen[key] = t
            elif current_rank == existing_rank:
                # 同等置信度，优先保留有evidence的
                if t.get("evidence") and not existing.get("evidence"):
                    seen[key] = t
    
    return list(seen.values())


# ============================================================================
# Layer 3: 生成流水线引擎
# ============================================================================

PIPELINE_ORDER = [
    "validate_input",      # 1. 校验输入
    "validate_schema",     # 2. Schema校验
    "generate_triples",    # 3. 生成三元组
    "detect_conflicts",    # 4. 冲突检测
    "deduplicate",         # 5. 去重
    "build_output",        # 6. 构建输出
]

STEP_HANDLERS = {
    "validate_input":     "_step_validate_input",
    "validate_schema":    "_step_validate_schema",
    "generate_triples":   "_step_generate_triples",
    "detect_conflicts":   "_step_detect_conflicts",
    "deduplicate":        "_step_deduplicate",
    "build_output":       "_step_build_output",
}


# ============================================================================
# Layer 4: 主处理类
# ============================================================================

class KGTripleGenerator:
    """
    知识图谱三元组生成算子
    
    功能：
    1. 将实体列表和关系列表组合为标准三元组
    2. 实体存在性和类型校验
    3. 关系-实体类型对合法性校验
    4. 去重与冲突检测
    5. 上下文约束字段预留
    
    输入格式：
        {
            "entities": [
                {"type": "疾病", "name": "糖尿病"},
                {"type": "药物", "name": "二甲双胍"}
            ],
            "relations": [
                {"head": "二甲双胍", "relation": "治疗", "tail": "糖尿病", "confidence": "high"}
            ]
        }
    
    输出格式：
        {
            "triples": [
                {
                    "head": "二甲双胍", "head_type": "药物",
                    "relation": "治疗",
                    "tail": "糖尿病", "tail_type": "疾病",
                    "confidence": "high",
                    "context_constraints": []
                }
            ],
            "conflicts": [],
            "warnings": []
        }
    
    处理顺序：validate_input → validate_schema → generate_triples → detect_conflicts → deduplicate → build_output
    """

    # =========================================================================
    # 常量定义
    # =========================================================================

    SUPPORTED_INPUT_KEYS = ["entities", "relations", "data"]

    # =========================================================================
    # 公开接口
    # =========================================================================

    def validate(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """校验输入数据"""
        errors: List[str] = []
        warnings: List[str] = []
        
        if input_data is None:
            errors.append("输入数据为空")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if isinstance(input_data, dict):
            if "entities" not in input_data:
                errors.append("缺少必要字段 'entities'")
            elif not isinstance(input_data["entities"], list):
                errors.append("'entities' 必须是列表")
            
            if "relations" not in input_data:
                errors.append("缺少必要字段 'relations'")
            elif not isinstance(input_data["relations"], list):
                errors.append("'relations' 必须是列表")
        else:
            errors.append(f"不支持的输入类型: {type(input_data).__name__}")
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def process(self, input_data: Any, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """主处理入口"""
        if params is None:
            params = {}
        return self._run_pipeline(input_data, dict(params))

    def get_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取结果摘要"""
        triples = result.get("triples", [])
        conflicts = result.get("conflicts", [])
        warnings = result.get("warnings", [])
        
        rel_dist = {}
        for t in triples:
            r = t.get("relation", "未知")
            rel_dist[r] = rel_dist.get(r, 0) + 1
        
        return {
            "triple_count": len(triples),
            "relation_distribution": rel_dist,
            "conflict_count": len(conflicts),
            "warning_count": len(warnings),
            "critical_conflicts": sum(
                1 for c in conflicts if c.get("severity") == "critical"
            ),
        }

    def get_schema(self) -> Dict[str, Any]:
        """获取输入输出Schema定义"""
        return {
            "input": {
                "type": "object",
                "required": ["entities", "relations"],
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "实体列表（来自EntityRecognizer输出）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "name": {"type": "string"}
                            },
                            "required": ["type", "name"]
                        }
                    },
                    "relations": {
                        "type": "array",
                        "description": "关系列表（来自RelationExtractor输出）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "head": {"type": "string"},
                                "relation": {"type": "string"},
                                "tail": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["head", "relation", "tail"]
                        }
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "triples": {
                        "type": "array",
                        "description": "标准化三元组列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "head": {"type": "string"},
                                "head_type": {"type": "string"},
                                "relation": {"type": "string"},
                                "tail": {"type": "string"},
                                "tail_type": {"type": "string"},
                                "confidence": {"type": "string"},
                                "context_constraints": {
                                    "type": "array",
                                    "description": "上下文约束（预留）"
                                }
                            }
                        }
                    },
                    "conflicts": {
                        "type": "array",
                        "description": "检测到的冲突列表"
                    },
                    "warnings": {
                        "type": "array",
                        "description": "处理过程中的警告"
                    }
                }
            }
        }

    # =========================================================================
    # 流水线引擎
    # =========================================================================

    def _run_pipeline(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        state = {
            "input": input_data,
            "params": params,
            "errors": [],
            "warnings": [],
            "triples": [],
            "conflicts": [],
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
                       step_name: str, handler) -> Dict[str, Any]:
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
        validation = self.validate(state["input"], state["params"])
        state["warnings"].extend(validation.get("warnings", []))
        if not validation.get("valid", False):
            state["errors"].extend(validation.get("errors", []))
            state["_fatal"] = True
        return state

    def _step_validate_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Schema校验：检查实体类型和关系类型的合法性"""
        input_data = state["input"]
        entities = input_data.get("entities", [])
        relations = input_data.get("relations", [])
        schema_warnings: List[str] = []
        
        # 校验实体类型
        for ent in entities:
            etype = ent.get("type", "")
            if not _validate_entity_type(etype):
                schema_warnings.append(
                    f"非法实体类型 '{etype}'（实体: {ent.get('name', '')}），"
                    f"合法类型: {VALID_ENTITY_TYPES}"
                )
        
        # 校验关系类型
        for rel in relations:
            rtype = rel.get("relation", "")
            if not _validate_relation_type(rtype):
                schema_warnings.append(
                    f"非法关系类型 '{rtype}'，合法类型: {VALID_RELATION_TYPES}"
                )
        
        state["warnings"].extend(schema_warnings)
        return state

    def _step_generate_triples(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成三元组"""
        input_data = state["input"]
        entities = input_data.get("entities", [])
        relations = input_data.get("relations", [])
        
        triples, build_warnings = _build_triples(entities, relations)
        
        state["triples"] = triples
        state["warnings"].extend(build_warnings)
        return state

    def _step_detect_conflicts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """冲突检测"""
        conflicts = _detect_conflicts(state.get("triples", []))
        state["conflicts"] = conflicts
        return state

    def _step_deduplicate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """去重"""
        state["triples"] = _deduplicate(state.get("triples", []))
        return state

    def _step_build_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def _build_final_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 移除内部标记字段
        clean_triples = []
        for t in state.get("triples", []):
            clean = {k: v for k, v in t.items() if not k.startswith("_")}
            clean_triples.append(clean)
        
        output = {
            "triples": clean_triples,
            "triple_count": len(clean_triples),
            "conflicts": state.get("conflicts", []),
            "conflict_count": len(state.get("conflicts", [])),
        }
        
        if state.get("errors"):
            output["errors"] = state["errors"]
        if state.get("warnings"):
            output["warnings"] = state["warnings"]
        
        return output
