"""MedicalSectionSplitter —— 病历章节结构化算子。

把清洗后的病历文本按章节名切分为结构化 dict，输出 JSON 写回 sample["text"]
（平台内 text 列可靠流转），并镜像到 ext_params["sections"]。
"""

import json
import re
import time

from loguru import logger

from dki.dmcompat import Mapper, OPERATORS

DEFAULT_SECTIONS = ["基本信息", "主诉", "现病史", "既往史", "查体",
                    "辅助检查", "诊断", "治疗方案", "医嘱"]


class MedicalSectionSplitter(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sections = kwargs.get("sections", None)
        if isinstance(sections, str):
            sections = [s.strip() for s in sections.split(",") if s.strip()]
        self.sections = sections or list(DEFAULT_SECTIONS)
        # 章节头模式：行首（或句首）的 "章节名：" / "章节名:"
        names = "|".join(re.escape(s) for s in self.sections)
        self._header_re = re.compile(rf"(?:^|\n)\s*({names})\s*[:：]", re.M)

    def execute(self, sample):
        start = time.time()
        self.read_file_first(sample)
        text = sample[self.text_key]

        matches = list(self._header_re.finditer(text))
        sections = {}
        if not matches:
            sections["全文"] = text.strip()
        else:
            # 章节头之前的引言归入"全文头"
            if matches[0].start() > 0:
                head = text[: matches[0].start()].strip()
                if head:
                    sections["全文头"] = head
            for i, m in enumerate(matches):
                name = m.group(1)
                seg_start = m.end()
                seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                content = text[seg_start:seg_end].strip()
                if name in sections:  # 重复章节头合并
                    sections[name] += "\n" + content
                else:
                    sections[name] = content

        payload = {"sections": sections, "raw_len": len(text)}
        sample[self.text_key] = json.dumps(payload, ensure_ascii=False)
        ext = sample.get(self.ext_params_key) or {}
        ext["sections"] = sections
        sample[self.ext_params_key] = ext
        logger.info("fileName: {}, method: MedicalSectionSplitter costs {:.6f} s",
                    sample.get(self.filename_key), time.time() - start)
        return sample
