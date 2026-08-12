# -*- coding: utf-8 -*-
"""
URL/噪声清理算子：删除URL链接、HTML注释、HTML实体，不留占位符。
"""
import re
import time
from typing import Dict, Any

from loguru import logger
from datamate.core.base_op import Mapper

_URL_RE = re.compile(
    r'(?:https?|ftp|file)://[\-A-Za-z0-9\+&@\(\)#/%\?=\^~_|!:\,\.\;]+'
    r'[\-A-Za-z0-9\+&@#/%=\~_\|]',
    re.MULTILINE | re.UNICODE
)
_BARE_URL_RE = re.compile(
    r'\b(?:www|w{2,3})\s*[.．。]\s*[^\s，。；;、]{1,80}?\s*[.．。]\s*'
    r'(?:com|cn|net|org|top|xyz|wang)\b',
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
_OCR_BARE_URL_RE = re.compile(
    r'\b(?:www|w{2,3})\s*(?:[.．。]\s*)+'
    r'(?:[A-Za-z0-9_\-]{2,40}[瑚坶]'
    r'|[A-Za-z0-9_\-]{2,40}(?:[.．。]\s*)+'
    r'[A-Za-z](?:\s*[.．。]?\s*[A-Za-z]){0,4}'
    r'|[A-Za-z0-9_\-]{3,40})',
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
_EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b|'
    r'@[A-Za-z0-9.\-]+'
)
_LOST_LINK_RE = re.compile(r'(?:图片|资料|链接)?链接?已失效[:：]?\s*', re.IGNORECASE)
# HTML 注释：<!-- ... -->
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
# HTML 实体：&nbsp; &amp; &lt; &gt; &quot; 等
_HTML_ENTITY_MAP = {
    '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&apos;': "'", '&middot;': '·', '&bull;': '·',
    '&#160;': ' ', '&#38;': '&', '&#60;': '<', '&#62;': '>',
    # 双重编码实体：&amp;nbsp; 经由 HtmlTagCleaner 解 &amp;→& 后会变回 &nbsp;
    # 在此直接消灭，避免下游重新产生实体
    '&amp;nbsp;': ' ', '&amp;lt;': '<', '&amp;gt;': '>',
    '&amp;amp;': '&', '&amp;quot;': '"',
}
_HTML_ENTITY_RE = re.compile('|'.join(
    re.escape(k) for k in sorted(_HTML_ENTITY_MAP.keys(), key=len, reverse=True)
))
# 清理多余空格和空括号
_CLEANUP_RE = re.compile(r'[（(]\s*[）)]\s*|[ \t]{2,}')


class UrlRemover(Mapper):

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]
        text = _OCR_BARE_URL_RE.sub("", text)
        text = _URL_RE.sub("", text)
        text = _BARE_URL_RE.sub("", text)
        text = _EMAIL_RE.sub("", text)
        text = _LOST_LINK_RE.sub("", text)
        text = _HTML_COMMENT_RE.sub("", text)
        text = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITY_MAP[m.group(0)], text)
        text = _CLEANUP_RE.sub(" ", text).strip()
        sample[self.text_key] = text
        logger.info(
            f"fileName: {sample[self.filename_key]}, "
            f"method: UrlRemover costs {time.time() - start:.6f} s"
        )
        return sample
