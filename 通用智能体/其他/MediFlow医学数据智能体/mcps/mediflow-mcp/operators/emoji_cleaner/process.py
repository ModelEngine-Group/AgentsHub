# -*- coding: utf-8 -*-
"""
Emoji/颜文字清洗算子：去除医疗文本中的 Emoji、颜文字、内部 URL 链接
"""
import re
import time
from typing import Dict, Any

from loguru import logger
from datamate.core.base_op import Mapper

# Unicode emoji 区间（覆盖主要 emoji block）
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0000200D"             # Zero Width Joiner
    "\U0000FE0F"             # Variation Selector-16
    "]+",
    flags=re.UNICODE
)

# 内部/无效 URL（医院内网链接、图片链接已失效等）
_URL_PATTERN = re.compile(
    r'https?://[^\s一-鿿]+',
    flags=re.UNICODE
)

# 多余标点堆叠（如 ！！！、？？？、……）
_PUNCT_REPEAT_PATTERN = re.compile(r'([！？!?…]{2,})', flags=re.UNICODE)


class EmojiCleaner(Mapper):

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        if not text.strip():
            return sample

        # 1. 去除 Emoji
        text = _EMOJI_PATTERN.sub("", text)
        # 2. 去除内部 URL
        text = _URL_PATTERN.sub("", text)
        # 3. 压缩重复感叹号/问号（≥3个→1个）
        text = _PUNCT_REPEAT_PATTERN.sub(lambda m: m.group(1)[0], text)
        # 4. 清理多余空白
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        sample[self.text_key] = text.strip()
        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"method: EmojiCleaner costs {time.time() - start:.6f} s"
        )
        return sample
