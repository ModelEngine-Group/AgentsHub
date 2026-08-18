# -*- coding: utf-8 -*-
"""
KGEntityRecognizer - 医疗实体识别算子

基于医疗词典 + 正则匹配，从文本中识别预定义的4类医疗实体：
- 疾病 (Disease)
- 症状 (Symptom)
- 药物 (Drug)
- 检查 (Examination)

架构：
- Layer 4（公开接口）：validate / process / get_summary / get_schema
- Layer 3（识别流水线）：PIPELINE_ORDER → STEP_HANDLERS → _run_step_safe
- Layer 2（实体抽取器）：_extract_by_dict / _extract_by_regex / _merge_results
- Layer 1（基础设施）：异常类 / 医疗词典 / 输入标准化 / 输出构建

与 data_cleaner / data_exporter 保持一致的架构模式
版本: 1.0.0
"""

import sys
import os
import re
import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Set

# 日志配置
logger = logging.getLogger("KGEntityRecognizer")

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

class EntityRecognizerError(Exception):
    """实体识别算子基础异常"""
    pass


class ValidationError(EntityRecognizerError):
    """参数校验失败"""
    pass


class ProcessingError(EntityRecognizerError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# Layer 1: 医疗实体词典（预定义4类实体）
# ============================================================================

MEDICAL_DICT: Dict[str, List[str]] = {
    "疾病": [
        "糖尿病", "2型糖尿病", "高血压", "冠心病", "肺炎", "胃溃疡",
        "支气管炎", "贫血", "高血脂", "高脂血症", "脑梗塞", "脑梗死",
        "心肌梗死", "心梗", "肝炎", "肾炎", "甲状腺功能亢进", "甲亢",
        "骨质疏松", "类风湿关节炎", "哮喘", "慢性阻塞性肺疾病", "慢阻肺",
        "脂肪肝", "胆结石", "胆囊炎", "胰腺炎", "阑尾炎", "白内障",
        "青光眼", "前列腺增生", "肾功能不全", "心力衰竭", "心衰",
        "心律失常", "房颤", "感冒", "流感", "上呼吸道感染", "扁桃体炎",
        "胃炎", "肠炎", "结肠炎", "痔疮", "痛风", "甲状腺结节",
        "乳腺增生", "子宫肌瘤", "前列腺炎", "尿路感染", "肾结石",
        "中耳炎", "鼻窦炎", "咽喉炎", "口腔溃疡", "角膜炎",
        "抑郁症", "焦虑症", "失眠症", "偏头痛", "癫痫",
    ],
    "症状": [
        "发热", "发烧", "头痛", "咳嗽", "咳痰", "乏力", "疲倦",
        "呕吐", "恶心", "胸闷", "气短", "呼吸困难", "心悸",
        "头晕", "眩晕", "水肿", "浮肿", "多饮多尿", "体重下降",
        "消瘦", "视力模糊", "腹痛", "腹泻", "便秘", "食欲不振",
        "纳差", "失眠", "入睡困难", "关节痛", "关节疼痛", "肌肉酸痛",
        "皮肤瘙痒", "尿频尿急", "尿痛", "血尿", "蛋白尿", "黄疸",
        "意识障碍", "昏迷", "抽搐", "惊厥", "盗汗", "寒战", "发冷",
        "胸痛", "背痛", "腰痛", "肩痛", "肢体麻木", "言语不清",
        "口眼歪斜", "偏瘫", "瘫痪", "耳鸣", "听力下降", "鼻塞",
        "流涕", "咽痛", "声音嘶哑", "口干", "口苦", "腹胀",
        "反酸", "烧心", "打嗝", "便血", "黑便", "皮肤瘀斑",
        "颈部肿块", "淋巴结肿大", "多汗", "怕热", "怕冷",
    ],
    "药物": [
        "二甲双胍", "格列本脲", "格列齐特", "阿卡波糖", "胰岛素",
        "阿司匹林", "布洛芬", "对乙酰氨基酚", "氨溴索", "乙酰半胱氨酸",
        "硝苯地平", "卡托普利", "依那普利", "氯沙坦", "缬沙坦",
        "美托洛尔", "比索洛尔", "阿托伐他汀", "瑞舒伐他汀", "辛伐他汀",
        "阿莫西林", "头孢克肟", "头孢呋辛", "头孢拉定", "左氧氟沙星",
        "阿奇霉素", "克拉霉素", "罗红霉素", "奥美拉唑", "泮托拉唑",
        "雷贝拉唑", "克拉霉素", "甲硝唑", "华法林", "氯吡格雷",
        "替格瑞洛", "地高辛", "硝酸甘油", "速尿", "呋塞米",
        "螺内酯", "氢氯噻嗪", "泼尼松", "甲泼尼龙", "地塞米松",
        "环磷酰胺", "甲氨蝶呤", "来氟米特", "羟氯喹", "柳氮磺吡啶",
        "连花清瘟胶囊", "连花清瘟", "蒲地蓝消炎片", "藿香正气水",
        "云南白药", "板蓝根", "感冒灵", "复方甘草片", "川贝枇杷膏",
        "阿卡波糖", "格列吡嗪", "罗格列酮", "西格列汀", "达格列净",
        "氨氯地平", "非洛地平", "贝那普利", "厄贝沙坦", "坎地沙坦",
        "吲达帕胺", "特拉唑嗪", "多沙唑嗪", "非那雄胺", "坦索罗辛",
    ],
    "检查": [
        "血常规", "尿常规", "大便常规", "便常规", "肝功能", "肝功",
        "肾功能", "肾功", "血糖", "空腹血糖", "餐后血糖", "糖化血红蛋白",
        "血脂", "血脂四项", "血脂六项", "心电图", "动态心电图",
        "X光", "胸部X线", "胸片", "CT", "计算机断层扫描", "MRI",
        "核磁共振", "B超", "超声", "彩超", "心脏彩超", "腹部B超",
        "胃镜", "肠镜", "结肠镜", "血压测量", "测血压", "血压监测",
        "眼底检查", "肺功能", "肺功能检查", "骨密度", "骨密度检查",
        "病理活检", "活检", "冠状动脉造影", "冠脉造影", "脑电图",
        "肌电图", "超声心动图", "心肌酶谱", "甲状腺功能", "甲功",
        "肿瘤标志物", "肿瘤标记物", "HPV检测", "TCT", "宫颈刮片",
        "心电图运动负荷试验", "平板运动试验", "C反应蛋白", "血沉",
        "凝血功能", "D-二聚体", "B型钠尿肽", "BNP",
    ],
}


def _normalize_dict() -> Dict[str, List[Tuple[str, str]]]:
    """
    构建归一化词典：以原始词为key，返回 [(类型, 原始词)] 列表
    
    同时构建首字索引加速匹配
    """
    normalized: Dict[str, List[Tuple[str, str]]] = {}
    first_char_index: Dict[str, List[str]] = {}
    
    for entity_type, terms in MEDICAL_DICT.items():
        for term in terms:
            term_norm = term.strip()
            if not term_norm:
                continue
            if term_norm not in normalized:
                normalized[term_norm] = []
            normalized[term_norm].append((entity_type, term_norm))
            
            # 建立首字索引
            first_char = term_norm[0]
            if first_char not in first_char_index:
                first_char_index[first_char] = []
            first_char_index[first_char].append(term_norm)
    
    return normalized, first_char_index


# 全局词典缓存（模块加载时初始化）
_ENTITY_DICT, _FIRST_CHAR_INDEX = _normalize_dict()

# 按词长降序排列，优先匹配长词
_DICT_SORTED = sorted(_ENTITY_DICT.keys(), key=len, reverse=True)


# ============================================================================
# Layer 2: 实体抽取器
# ============================================================================

def _extract_by_dict(text: str) -> List[Dict[str, Any]]:
    """
    基于词典的实体抽取（包含交叉匹配去重）
    
    使用首字索引加速，只遍历以词典中首字开头的位置
    """
    if not text:
        return []
    
    matches: List[Dict[str, Any]] = []
    text_len = len(text)
    matched_spans: Set[Tuple[int, int]] = set()
    
    i = 0
    while i < text_len:
        char = text[i]
        
        # 检查该字是否为某个词典词的首字
        candidate_terms = _FIRST_CHAR_INDEX.get(char, [])
        
        for term in sorted(candidate_terms, key=len, reverse=True):
            term_len = len(term)
            if i + term_len > text_len:
                continue
            
            # 检查文本是否匹配该词典词
            if text[i:i+term_len] == term:
                # 检查此区间是否已被更长的匹配覆盖
                span_covered = any(
                    s <= i and e >= i + term_len
                    for s, e in matched_spans
                )
                if not span_covered:
                    matches.append({
                        "type": _ENTITY_DICT[term][0][0],
                        "name": term,
                        "start": i,
                        "end": i + term_len
                    })
                    matched_spans.add((i, i + term_len))
                # 跳过已匹配的部分（贪心匹配最长）
                # 但仍然需要检查其他实体类型下是否有不同的匹配
                break
        
        i += 1
    
    return matches


def _extract_by_regex(text: str) -> List[Dict[str, Any]]:
    """
    基于正则的实体抽取（补充词典无法覆盖的情况）
    
    例如：数字+单位 类型的检查项
    """
    matches: List[Dict[str, Any]] = []
    
    # 模式1：数字+单位 类检查项（如 "13.2mmol/L" 推测为血糖）
    # 暂时不做复杂regex，保持轻量
    
    return matches


def _merge_results(dict_matches: List[Dict[str, Any]],
                   regex_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并多种来源的实体，按文本位置排序，去重"""
    all_entities = dict_matches + regex_matches
    
    # 按 start 排序
    all_entities.sort(key=lambda e: (e["start"], -e["end"]))
    
    # 去重（相同位置+相同类型+相同名称只保留一个）
    seen: Set[Tuple[int, int, str]] = set()
    merged: List[Dict[str, Any]] = []
    for ent in all_entities:
        key = (ent["start"], ent["end"], ent["type"])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    
    return merged


# ============================================================================
# Layer 3: 识别流水线引擎
# ============================================================================

PIPELINE_ORDER = [
    "validate_input",     # 1. 校验输入
    "extract_entities",   # 2. 实体抽取
    "build_output",       # 3. 构建输出
]

STEP_HANDLERS = {
    "validate_input":     "_step_validate_input",
    "extract_entities":   "_step_extract_entities",
    "build_output":       "_step_build_output",
}


# ============================================================================
# Layer 4: 主处理类
# ============================================================================

class KGEntityRecognizer:
    """
    医疗实体识别算子
    
    功能：
    1. 从医疗文本中识别预定义的4类实体（疾病/症状/药物/检查）
    2. 返回实体类型、名称、位置信息
    
    输入格式：
        {"text": "患者糖尿病多年，出现多饮多尿症状"}
        {"data": [{"text": "..."}, {"text": "..."}]}  # 批量模式
    
    输出格式：
        {"entities": [
            {"type": "疾病", "name": "糖尿病", "start": 2, "end": 5},
            {"type": "症状", "name": "多饮多尿", "start": 10, "end": 14}
        ]}
    
    处理顺序：validate_input → extract_entities → build_output
    """

    # =========================================================================
    # 常量定义
    # =========================================================================

    SUPPORTED_INPUT_KEYS = ["text", "data"]

    # =========================================================================
    # 公开接口
    # =========================================================================

    def validate(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        校验输入数据
        
        Args:
            input_data: 输入数据（dict 或 list）
            params: 参数（当前未使用）
            
        Returns:
            校验结果: {"valid": bool, "errors": [str], "warnings": [str]}
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        if input_data is None:
            errors.append("输入数据为空")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if isinstance(input_data, dict):
            has_valid_key = any(k in input_data for k in self.SUPPORTED_INPUT_KEYS)
            if not has_valid_key:
                warnings.append(
                    f"输入字典中未找到支持的key ({self.SUPPORTED_INPUT_KEYS})，"
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
        
        Args:
            input_data: 输入数据
            params: 参数字典（当前保留接口，未使用）
            
        Returns:
            处理结果
        """
        if params is None:
            params = {}
        
        pipeline_params = dict(params)
        pipeline_params.setdefault("data_key", "data")
        
        return self._run_pipeline(input_data, pipeline_params)

    def get_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取结果摘要
        
        Args:
            result: process() 的输出
            
        Returns:
            包含摘要信息的字典
        """
        entities = result.get("entities", [])
        if not entities:
            return {"entity_count": 0, "type_distribution": {}}
        
        type_dist = {}
        for ent in entities:
            t = ent.get("type", "未知")
            type_dist[t] = type_dist.get(t, 0) + 1
        
        return {
            "entity_count": len(entities),
            "type_distribution": type_dist,
            "entity_names": [e.get("name", "") for e in entities],
        }

    def get_schema(self) -> Dict[str, Any]:
        """
        获取算子的输入输出Schema定义
        
        Returns:
            Schema字典
        """
        return {
            "input": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "医疗文本（单条）"
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "批量数据（需包含text字段）"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "识别出的实体列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "实体类型: 疾病/症状/药物/检查"},
                                "name": {"type": "string", "description": "实体名称"},
                                "start": {"type": "integer", "description": "起始位置"},
                                "end": {"type": "integer", "description": "结束位置"}
                            }
                        }
                    },
                    "entity_count": {
                        "type": "integer",
                        "description": "实体总数"
                    }
                }
            }
        }

    # =========================================================================
    # 流水线引擎（内部）
    # =========================================================================

    def _run_pipeline(self, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行流水线：按 PIPELINE_ORDER 顺序执行每个步骤
        """
        state = {
            "input": input_data,
            "params": params,
            "errors": [],
            "warnings": [],
            "entities": [],
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
        """
        安全执行单个步骤，捕获异常
        """
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
        """步骤1：校验输入"""
        input_data = state["input"]
        
        validation = self.validate(input_data, state["params"])
        state["warnings"].extend(validation.get("warnings", []))
        
        if not validation.get("valid", False):
            state["errors"].extend(validation.get("errors", []))
            state["_fatal"] = True
            return state
        
        # 提取待处理的文本列表
        texts: List[str] = []
        
        if isinstance(input_data, dict):
            if "text" in input_data:
                text = input_data["text"]
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
            elif "data" in input_data:
                data_list = input_data["data"]
                if isinstance(data_list, list):
                    for item in data_list:
                        if isinstance(item, dict) and "text" in item:
                            t = item["text"]
                            if isinstance(t, str) and t.strip():
                                texts.append(t.strip())
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                elif isinstance(item, dict) and "text" in item:
                    t = item["text"]
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
        
        if not texts:
            state["warnings"].append("未提取到有效的文本输入")
        
        state["texts"] = texts
        return state

    def _step_extract_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """步骤2：实体抽取"""
        texts = state.get("texts", [])
        all_entities: List[Dict[str, Any]] = []
        
        for text in texts:
            dict_matches = _extract_by_dict(text)
            regex_matches = _extract_by_regex(text)
            entities = _merge_results(dict_matches, regex_matches)
            all_entities.extend(entities)
        
        # 去重（相同type+name只保留一个）
        seen: Set[Tuple[str, str]] = set()
        unique_entities: List[Dict[str, Any]] = []
        for ent in all_entities:
            key = (ent["type"], ent["name"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        state["entities"] = unique_entities
        return state

    def _step_build_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """步骤3：构建最终输出"""
        # 此步骤在 _build_final_output 中统一处理
        return state

    def _build_final_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建最终输出结果"""
        output = {
            "entities": state.get("entities", []),
            "entity_count": len(state.get("entities", [])),
        }
        
        if state.get("errors"):
            output["errors"] = state["errors"]
        if state.get("warnings"):
            output["warnings"] = state["warnings"]
        
        return output
