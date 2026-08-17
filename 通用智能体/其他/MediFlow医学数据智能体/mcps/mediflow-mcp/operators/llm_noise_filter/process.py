# -*- coding: utf-8 -*-
"""
LLM 语义噪声过滤算子：处理规则算子无法覆盖的语义级噪声。
  - 感叹词/语气词（哎呀妈呀、呜呜呜、卧槽）
  - 无关生活描述（昨天吃了排骨、跟隔壁老王下棋）
  - 系统导出废话（由HIS系统自动导出、图片链接已失效）
  - @mentions 通知类内容（@所有人 记得录入）
  - 填表人/时间戳等元数据（填表：李护士，时间2026-06-07）

设计原则：
  - 不修改医疗语义（症状/诊断/用药/检查描述一字不改）
  - 不做实体标准化（那是 MedicalTermNormalizer 的职责）
  - 当文本中没有上述噪声时，原文原样返回

触发建议（由 Agent 判断）：
  文本中含以下特征之一时调用：
    1. 感叹词/口语化表达：哎|呜|卧槽|太..了|好吧|嗷
    2. @mentions：@[^\s]
    3. 系统导出注释：由.*系统|自动导出|链接已失效|填表.*：
    4. 无关人名+动词：护士|医生\s*（非诊断语境）
"""
import os
import re
import time
from pathlib import Path
import difflib
from typing import Dict, Any

import requests
from loguru import logger

from datamate.core.base_op import Mapper

try:
    from ops.user.llm_noise_filter.noise_logger import (
        log_run, _scan_signals, NOISE_SIGNALS
    )
    from ops.user.llm_noise_filter.noise_rule_engine import NoiseRuleEngine
except ImportError:
    from noise_logger import log_run, _scan_signals, NOISE_SIGNALS
    from noise_rule_engine import NoiseRuleEngine

# 快速检测：是否存在语义噪声信号（若无则跳过 LLM 调用）
_COLLOQUIAL_RE = re.compile(
    r'哎[呀ya]|呜呜|卧槽|太[^\s]{0,4}了|嗷嗷|好吧好吧|OMG|WTF'
    r'|@[^\s@]{1,20}'                        # @mentions
    r'|由.{1,10}系统.{0,6}(导出|生成|产生)'   # 系统导出注释
    r'|图片链接已失效|该网页无法显示'
    r'|填表[时间：:].{0,30}[0-9]'            # 填表时间戳
    r'|记得把这个录入|交接班记录'
    r'|跟.{1,8}聊天'                          # 无关闲聊（"跟邻居聊天"）
    r'|邻居|煮饭|买菜|散步.{0,3}(遇到|碰见|看见)'  # 日常琐事
    r'|听.{1,6}说[^.。]{0,20}(可以|能|管用|有效)'  # 道听途说
    r'|去[大中心]{0,2}医院看看'               # 第三方随意建议
    r'|试.{0,3}(试|一下|看看)(?!表|剂|药|方|案)'  # 口语试探（排除"试表""试剂"）
    r'|不知道吃[什么啥](?!药)',                # 非医疗的不确定（"不知道吃什么好"）
    re.IGNORECASE
)

# 医疗数值 token（带单位的数字）：用于安全检查，确保 LLM 没删掉化验值/药量
_MED_VALUE_RE = re.compile(
    r'\d+\.?\d*\s*(?:mmol/L|mmHg|mg|kg|g|%|U/L|IU|ug/L|μg/L|umol/L|μmol/L|'
    r'ng/mL|mL|ml|次/分|单位|U)\b',
    re.IGNORECASE
)

_DETERMINISTIC_NOISE_PATTERNS = [
    ("his_export", re.compile(
        r"(?:HIS|EMR)?系统自动(?:导出|生成)|"
        r"HIS系统自动导出\s*填表时间[:：][^\n]{0,80}|"
        r"填表时间[:：][^\n]{0,80}填表人[:：][^\n]{0,80}",
        re.IGNORECASE,
    )),
    ("system_boilerplate", re.compile(r"【系统提示】[^\n]{0,120}(?:自动生成|内部流转)[^\n]*")),
    ("debug_metadata", re.compile(
        r"患者ID[:：]\s*TMP-[^\n]{0,80}|"
        r"设备[:：]\s*EMR-DEBUG[^\n]{0,80}|"
        r"操作员[:：]\s*admin_test",
        re.IGNORECASE,
    )),
    ("ocr_tail", re.compile(
        r"@{0,3}#{0,3}\s*OCR_CONFIDENCE\s*=\s*[0-9.]+\s*#{0,3}@{0,3}",
        re.IGNORECASE,
    )),
    ("ad_disclaimer", re.compile(
        r"(?:关注|注)?公众号领取健康资料|"
        r"广告合作请联系\s*\S*|"
        r"免责声明[:：][^\n]{0,200}|"
        r"您可以提供微信号[^\n，。；;]{0,80}|"
        r"微信号\s*[A-Za-z0-9_\-]{3,40}",
        re.IGNORECASE,
    )),
    ("view_more_tail", re.compile(
        r"查看更多关于[^\n]{0,180}(?:\.\.\.|…)",
        re.IGNORECASE,
    )),
]

def _med_tokens(text: str) -> set:
    """提取带单位的医疗数值（化验值、药量、生命体征），归一化空白后比较"""
    return {re.sub(r'\s+', '', m.group(0)) for m in _MED_VALUE_RE.finditer(text)}


def _deterministic_clean(text: str) -> tuple[str, list[str]]:
    """在调用大模型前删除高置信导出、广告和生成式噪声。"""
    removed = []
    for name, pattern in _DETERMINISTIC_NOISE_PATTERNS:
        matches = [m.group(0) for m in pattern.finditer(text)]
        if matches:
            removed.extend(f"{name}: {item[:160]}" for item in matches)
            text = pattern.sub("", text)
    return text, removed


PROMPT = """你是医疗文本清洗专家。以下文本来自医疗场景（病历、问诊记录、医学文档），其中可能夹杂了与医疗无关的内容。

请完成以下操作，直接输出清洗后的文本，不做任何解释：
1. 删除感叹词、语气词（如"哎呀妈呀"、"呜呜呜"、"太吓人了"、"卧槽"）
2. 删除与医疗无关的生活描述（如"昨天吃了排骨"、"跟隔壁老王下棋"）
3. 删除 @mentions 通知（如"@所有人 记得录入"）
4. 删除系统自动生成的废话（如"由HIS系统自动导出"、"图片链接已失效"、"填表人：李护士"）
5. 删除纯占位符行（如"暂无记录"独占一行时）
6. 删除在线问诊的寒暄套话：开场白（如"你好，根据你的描述"、"看了你的描述"）、
   结尾祝福（如"祝你健康"、"祝早日康复"）、安抚客套（如"不用太担心"、"问题不大"、
   "以上仅供参考"），但务必保留其中的实质医疗建议（诊断/用药/检查/转诊指征）

严格禁止：
- 不修改任何症状、诊断、用药、检查数值描述
- 症状的性状描述要保留（如"针扎样疼痛"、"刀割样痛"是有医学意义的体征，不是噪声）
- 不做内容总结或改写
- 若文本无上述噪声，原样返回

文本：
{text}"""


class LLMNoiseFilter(Mapper):

    def __init__(self, *args, **kwargs):
        super(LLMNoiseFilter, self).__init__(*args, **kwargs)
        self.ollama_url = kwargs.get("ollamaUrl", "https://api.deepseek.com/v1/chat/completions")
        self.model = kwargs.get("modelName", "deepseek-chat")
        self.timeout = int(kwargs.get("timeoutSeconds", 60))
        self.api_key = kwargs.get("apiKey") or os.environ.get("CCF_LLM_API_KEY", "")
        if not self.api_key:
            key_file = os.environ.get("CCF_LLM_API_KEY_FILE", "/run/secrets/ccf_llm_api_key")
            try:
                self.api_key = Path(key_file).read_text(encoding="utf-8").strip()
            except OSError:
                self.api_key = ""
        # 从外部 SQLite 知识库加载噪声规则（独立于算子代码）
        self._kb_re = None
        kb_path = kwargs.get("kbPath",
            "/opt/runtime/datamate/ops/user/llm_noise_filter/noise_kb.db")
        self._rule_engine = NoiseRuleEngine(kb_path)
        self._load_kb_rules(kb_path)

    def _load_kb_rules(self, kb_path):
        """从外部 SQLite 噪声知识库加载所有 active 规则，编译为一个正则。
        按 match_type 区分匹配策略（问题1的平衡点）：
          - exact:  精确子串匹配（广告/水印/固定模板）→ re.escape
          - regex:  正则模糊匹配（口语变体，如 吓死.{0,2}）→ 原样
        单条规则 try-compile 容错：坏规则被跳过，不拖垮整体编译。"""
        import sqlite3 as _sql
        try:
            conn = _sql.connect(kb_path)
            # 兼容旧库：无 match_type 列时默认按 regex 处理
            cols = [c[1] for c in conn.execute("PRAGMA table_info(noise_rules)").fetchall()]
            has_mt = "match_type" in cols
            sel = ("SELECT pattern, match_type FROM noise_rules "
                   "WHERE status='active' AND medical_safe=1") if has_mt else \
                  ("SELECT pattern, 'regex' FROM noise_rules "
                   "WHERE status='active' AND medical_safe=1")
            rows = conn.execute(sel).fetchall()
            conn.close()

            valid, skipped = [], 0
            for pattern, mtype in rows:
                if not pattern:
                    continue
                frag = re.escape(pattern) if mtype == "exact" else pattern
                try:
                    re.compile(frag)            # 单条验证语法
                    valid.append(f"(?:{frag})")
                except re.error:
                    skipped += 1                # 跳过坏规则
            if valid:
                self._kb_re = re.compile("|".join(valid), re.IGNORECASE)
                logger.info(f"LLMNoiseFilter: loaded {len(valid)} KB rules "
                            f"({skipped} skipped)")
        except Exception as e:
            logger.debug(f"LLMNoiseFilter: KB not loaded ({e})")

    def _needs_llm(self, text: str) -> bool:
        """快速判断是否存在语义噪声信号，无噪声则跳过 LLM"""
        if _COLLOQUIAL_RE.search(text):
            return True
        if self._rule_engine.has_match(text):
            return True
        if self._kb_re and self._kb_re.search(text):
            return True
        return False

    def _call_llm(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
            "stream": False,
            "temperature": 0.1,  # 低温保守，避免幻觉
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(self.ollama_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        original = text
        char_count = len(text)
        task_id = sample.get("instance_id", "") or sample.get("task_id", "")

        # 始终扫描噪声信号（_scan_signals 用于评分/日志的细分类别）
        signals = _scan_signals(text) if text.strip() else {}
        # 是否调 LLM：综合 noise_logger 细分信号 + 算子门控(含外部KB规则)
        # 关键：_needs_llm 包含 KB 规则，_scan_signals 不含 → 必须取并集
        has_signals = bool(signals) or (text.strip() and self._needs_llm(text))

        status = "clean（无噪声信号）"
        chars_removed = 0
        removed_segments = []

        rule_cleaned, rule_removed = self._rule_engine.clean(text)
        if rule_cleaned != text:
            removed_segments.extend(rule_removed)
            chars_removed += len(text) - len(rule_cleaned)
            text = rule_cleaned
            status = "cleaned(sqlite rules)"

        deterministic_cleaned, deterministic_removed = _deterministic_clean(text)
        if deterministic_cleaned != text:
            removed_segments.extend(deterministic_removed)
            chars_removed += len(text) - len(deterministic_cleaned)
            text = deterministic_cleaned
            status = "cleaned(deterministic rules)"

        signals = _scan_signals(text) if text.strip() else {}
        has_signals = bool(signals) or (text.strip() and self._needs_llm(text))

        if has_signals:
            try:
                cleaned = self._call_llm(text).strip()
                reason = self._reject_reason(text, cleaned)
                if reason:
                    logger.warning(f"LLMNoiseFilter: rejected ({reason})")
                    status = f"rejected（{reason}）"
                else:
                    chars_removed = char_count - len(cleaned)
                    # 提取被删片段：找出 LLM 输出和原文的差异行
                    removed_segments = self._extract_removed(text, cleaned)
                    text = cleaned
                    status = "cleaned"
            except Exception as e:
                logger.warning(f"LLMNoiseFilter LLM call failed: {e}")
                status = f"error（{e}）"
        elif not text.strip():
            status = "empty"

        # 写回清洗后文本
        sample[self.text_key] = text
        # 注入噪声元数据到 sample
        sample["_noise"] = {
            "status": status,
            "noise_score": round(len(signals) / max(1, sum(1 for _ in NOISE_SIGNALS)), 3),
            "signals_detected": list(signals.keys()),
            "categories": list(set(v["category"] for v in signals.values())),
            "chars_removed": chars_removed,
            "chars_before": char_count,
            "chars_after": len(text),
        }

        # 记录到噪声日志库（用于学习）
        try:
            log_run(
                task_id=task_id,
                file_name=sample.get(self.filename_key, ""),
                char_count=char_count,
                status=status,
                noise_signals=signals,
                chars_removed=chars_removed,
                removed_segments=removed_segments,
            )
        except Exception as log_err:
            logger.debug(f"noise_logger 记录失败: {log_err}")

        elapsed = time.time() - start
        summary = (f"LLMNoiseFilter[{status}] "
                   f"score={sample['_noise']['noise_score']:.2f} "
                   f"removed={chars_removed}chars "
                   f"cost={elapsed:.3f}s")

        logger.info(f"fileName: {sample[self.filename_key]}, {summary}")
        return sample

    def _extract_removed(self, original: str, cleaned: str) -> list:
        """提取被 LLM 删除的文本片段（行级 diff）"""
        import difflib
        orig_lines = original.splitlines()
        clean_lines = cleaned.splitlines()
        removed = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, orig_lines, clean_lines).get_opcodes():
            if tag == 'delete' or tag == 'replace':
                for idx in range(i1, i2):
                    removed.append(orig_lines[idx])
        return removed

    def _reject_reason(self, original: str, cleaned: str):
        """返回拒绝原因字符串；None 表示通过。
        只在「明显损坏」时拒绝：空/极短、异常增长（多半是LLM加了解释）、丢失医疗数值。"""
        if not cleaned:
            return "输出为空"
        if len(cleaned) < max(10, len(original) * 0.15):
            return f"输出过短{len(cleaned)}/{len(original)}"
        if len(cleaned) > len(original) * 1.15:
            return f"输出异常增长{len(cleaned)}/{len(original)}"
        orig_med = _med_tokens(original)
        if orig_med:
            lost = orig_med - _med_tokens(cleaned)
            if lost:
                return f"丢失医疗数值{sorted(lost)}"
        return None
