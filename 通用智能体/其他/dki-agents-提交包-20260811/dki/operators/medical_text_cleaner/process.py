"""MedicalTextCleaner —— 医疗文本组合清洗算子（DataMate 自定义算子规范）。

设计：优先复用 DataMate 内置清洗算子（HtmlTagCleaner / InvisibleCharactersCleaner /
FullWidthCharacterCleaner / ExtraSpaceCleaner / DuplicateSentencesFilter /
TraditionalChineseCleaner / AnonymizedPhoneNumber / AnonymizedIdNumber），
将多步清洗封装为一个面向医疗病历的组合算子；DataMate runtime 不可用时，
每一步自动回退到本文件内的等价正则实现（_fallback_*），保证独立可运行。
"""

import html as _html
import re
import time

from loguru import logger

from dki.dmcompat import DATAMATE_AVAILABLE as _DATAMATE, Mapper, OPERATORS

# 步骤名 -> DataMate 内置算子名
_BUILTIN_MAP = {
    "html": "HtmlTagCleaner",
    "invisible": "InvisibleCharactersCleaner",
    "fullwidth": "FullWidthCharacterCleaner",
    "space": "ExtraSpaceCleaner",
    "dedup_sent": "DuplicateSentencesFilter",
    "traditional": "TraditionalChineseCleaner",
}
_PII_BUILTINS = ["AnonymizedPhoneNumber", "AnonymizedIdNumber"]

# 注意顺序：invisible 必须先于 html（零宽字符可能嵌在标签内部使标签逃逸清理）
DEFAULT_STEPS = ["invisible", "html", "fullwidth", "space", "dedup_sent", "traditional"]


# ---------------------------------------------------------------- 回退实现
def _fallback_html(text: str) -> str:
    text = re.sub(r"<[^>\n]{1,120}>", "", text)
    return _html.unescape(text)


def _fallback_invisible(text: str) -> str:
    # 移除控制字符(保留 tab/newline/CR)与零宽、BOM 字符
    return re.sub("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\ufeff]", "", text)


def _fallback_fullwidth(text: str) -> str:
    # 仅转换全角数字/字母/空格与常见全角标点中的冒号（保留中文标点习惯）
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "　":
            out.append(" ")
        elif 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _fallback_space(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 中文字符/标点两侧的空格去除
    text = re.sub(r"(?<=[一-鿿，。：；！？、]) +(?=[一-鿿，。：；！？、])", "", text)
    text = re.sub(r" +(?=[，。：；！？、])", "", text)
    text = re.sub(r"(?<=[，。：；、]) +", "", text)
    return text


def _fallback_dedup_sent(text: str) -> str:
    def _key(p: str) -> str:
        # 剥离章节头前缀（如"现病史："）后比较，识别带头句与裸句的重复
        return re.sub(r"^[^。！？：]{1,8}：", "", p.strip())

    seen = set()
    out_lines = []
    for line in text.split("\n"):
        parts = re.split(r"(?<=[。！？])", line)
        kept = []
        for p in parts:
            key = _key(p)
            if not key or key in ("。", "！", "？"):
                kept.append(p)
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
        out_lines.append("".join(kept))
    # 整行级去重（保序）
    final, seen_lines = [], set()
    for line in out_lines:
        key = line.strip()
        if key and key in seen_lines:
            continue
        if key:
            seen_lines.add(key)
        final.append(line)
    return "\n".join(final)


_TRAD_TABLE = {"頭": "头", "醫": "医", "藥": "药", "檢": "检", "診": "诊",
               "斷": "断", "護": "护", "術": "术", "體": "体", "發": "发"}


def _fallback_traditional(text: str) -> str:
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        for k, v in _TRAD_TABLE.items():
            text = text.replace(k, v)
        return text


def _fallback_phone(text: str) -> str:
    return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "<tel>", text)


def _fallback_id(text: str) -> str:
    return re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)", "<id>", text)


_FALLBACKS = {
    "html": _fallback_html,
    "invisible": _fallback_invisible,
    "fullwidth": _fallback_fullwidth,
    "space": _fallback_space,
    "dedup_sent": _fallback_dedup_sent,
    "traditional": _fallback_traditional,
}


def _resolve_op(op_name: str):
    """从 OPERATORS 注册表解析算子类（自含实现，等价于平台 load_ops_module 的核心逻辑）。"""
    content = OPERATORS.modules.get(op_name)
    if content is None:
        raise KeyError(f"算子 {op_name} 未注册")
    if isinstance(content, str):
        import importlib

        mod = importlib.import_module(content)
        return getattr(mod, op_name)
    return content


class MedicalTextCleaner(Mapper):
    """组合式医疗文本清洗（步骤可配，复用 DataMate 内置算子）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        steps = kwargs.get("steps", None)
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split(",") if s.strip()]
        self.steps = steps or list(DEFAULT_STEPS)
        mask = kwargs.get("maskPii", True)
        self.mask_pii = str(mask).lower() not in ("false", "0", "no")
        self._sub_ops = self._build_sub_ops()

    def _build_sub_ops(self):
        """构建 (步骤名, callable) 列表；callable 输入输出均为 sample。"""
        pipeline = []
        if _DATAMATE:
            try:
                # 独立运行：通过 dki 集成层注册 DataMate 内置算子
                from dki.dmcompat import load_builtin_ops

                load_builtin_ops()
            except ImportError:
                pass  # 平台内运行（无 dki 包）：内置算子已由平台注册
        # 以注册表实况判定内置算子可用性
        builtin_ok = _DATAMATE and "HtmlTagCleaner" in OPERATORS.modules

        def make_builtin(op_name):
            sub = _resolve_op(op_name)(op_name=op_name)
            return lambda sample: sub.execute(sample)

        def make_fallback(fn):
            def _run(sample):
                sample[self.text_key] = fn(sample[self.text_key])
                return sample
            return _run

        for step in self.steps:
            if step not in _BUILTIN_MAP and step not in _FALLBACKS:
                logger.warning("未知清洗步骤 {}，跳过", step)
                continue
            used_builtin = False
            if builtin_ok:
                try:
                    pipeline.append((f"{step}(datamate)", make_builtin(_BUILTIN_MAP[step])))
                    used_builtin = True
                except Exception as e:
                    logger.warning("内置算子 {} 不可用（{}），使用回退实现", _BUILTIN_MAP.get(step), e)
            if not used_builtin:
                pipeline.append((f"{step}(fallback)", make_fallback(_FALLBACKS[step])))

        if self.mask_pii:
            if builtin_ok:
                try:
                    for op_name in _PII_BUILTINS:
                        pipeline.append((f"pii:{op_name}(datamate)", make_builtin(op_name)))
                except Exception as e:
                    logger.warning("内置脱敏算子不可用（{}），使用回退实现", e)
            # 兜底扫尾：内置算子对不合法号段/校验位不通过的串不处理，
            # 医疗场景要求宁可多脱敏——追加正则兜底（幂等）
            pipeline.append(("pii:phone(fallback)", make_fallback(_fallback_phone)))
            pipeline.append(("pii:id(fallback)", make_fallback(_fallback_id)))
        return pipeline

    def execute(self, sample):
        start = time.time()
        self.read_file_first(sample)
        applied = []
        # 先执行文本规范化步骤（非 PII）
        pii_ops = [(n, fn) for n, fn in self._sub_ops if n.startswith("pii:")]
        norm_ops = [(n, fn) for n, fn in self._sub_ops if not n.startswith("pii:")]
        for name, fn in norm_ops:
            sample = fn(sample)
            applied.append(name)
        # 收尾兜底（幂等）：零宽字符→残留标签→HTML实体→全角→空格，
        # 保证输出不变式不依赖内置算子的字符集差异
        text = sample[self.text_key]
        text = _fallback_invisible(text)
        text = _fallback_html(text)
        text = _html.unescape(text).replace("\xa0", " ")
        text = _fallback_fullwidth(text)
        text = _fallback_space(text)
        if "dedup_sent" in self.steps:
            # 内置算子按行去重；行内重复句由回退实现补齐
            text = _fallback_dedup_sent(text)
        sample[self.text_key] = text
        # PII 脱敏最后执行（避免 <tel>/<id> 掩码被 HTML 清扫误删）
        for name, fn in pii_ops:
            sample = fn(sample)
            applied.append(name)
        ext = sample.get(self.ext_params_key) or {}
        ext["clean_steps_applied"] = applied
        sample[self.ext_params_key] = ext
        logger.info("fileName: {}, method: MedicalTextCleaner costs {:.6f} s",
                    sample.get(self.filename_key), time.time() - start)
        return sample
