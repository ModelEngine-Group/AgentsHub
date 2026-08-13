# CBLUE 真实医疗基准数据卡（CMeEE-V2 / CMeIE-V2）

> 用途：为任务二"知识抽取质量"提供**真实、可对标 SOTA** 的权威基准，补充合成受控语料
> 的同分布 F1。本项目只在本地下载用于训练/评测，**不重分发原始语料**；提交包按许可只保留
> 本数据卡与 `scripts/download_cblue.py` 复现脚本（`data/cblue/` 已列入 `.gitignore`）。

## 来源与许可
- 基准：CBLUE（Chinese Biomedical Language Understanding Evaluation），由中国中文信息学会
  医疗健康与生物信息处理专委会发起、阿里云天池承办，供**学术研究**使用。
- 官方入口：天池 https://tianchi.aliyun.com/dataset/95414 （接受数据协议后可下载）。
- 本项目下载渠道：HuggingFace 社区镜像（hf-mirror），`scripts/download_cblue.py` 逐文件
  resolve 下载并规范化：
  - CMeEE-V2（NER）来自 `xusenlin/cmeee`（train/validation parquet）。
  - CMeIE-V2（关系抽取）来自 `wubingheng/CMeIE-V2-NER`（train/dev jsonl）。
- 合规：仅本地训练/评测；不将原始数据纳入 git 或提交包；引用时标注 CBLUE 出处。

## 规模与切分（规范化后，实测）
- CMeEE-V2：train 15,000 / dev 5,000（dev 带金标，用作评测集）。
- CMeIE-V2：train 14,339 / dev 3,585。
- 划分口径：直接采用官方 train/validation(dev) 切分；**test 无公开金标**，不用于本地报数。
  评测只在官方 dev 上做（避免用无标签 test 造成不可复核的数字）。

## 实体/关系 schema
- CMeEE-V2 9 类实体（官方英文标签 → 中文）：
  `dis 疾病 / sym 临床表现 / dru 药物 / equ 医疗设备 / pro 医疗程序 / bod 身体部位 /
  ite 医学检验项目 / mic 微生物类 / dep 科室`。
  span 采用半开区间 `[start, end)`，实测 `text[start:end]==entity` 100% 对齐。
- CMeIE-V2：44 类关系谓词（如 临床表现/药物治疗/病因/辅助治疗/实验室检查/同义词…），
  三元组形如 `{subject, predicate, object}`。

## 与本项目 6 类 schema 的映射（用于统一叙事，不改动默认词典链路）
- `dis→Disease、sym→Symptom、dru→Drug、pro/ite→Exam/Treatment（按语义细分）、
  bod/equ/mic→保留原类（真实基准更细）、dep→Department`。
- 说明：CBLUE 类目比本项目 6 类更细，报基准 F1 时用 **CBLUE 原生 9 类**口径（与 SOTA 可比），
  不强行压到 6 类以免降低可比性；本项目默认 dict 主链与合成评测**保持不变**。

## 防泄漏与评测纪律
- 仅用官方 train 训练、官方 dev 评测；不做 dev 上调参后又拿 dev 报"最终"数字之外的夸大。
- 与本项目 final3/OOD 盲测集**物理隔离**（不同来源、不同用途），互不进入对方训练/选模。
- 报数口径：Micro-F1（实体级，与 CBLUE 官方一致）。**官方论文 PLM baseline 约 60.6–62.4，人类 67.0**（test 口径）；本仓库本地可复核报 **dev**。
- 产品 schema 映射（6 类）仅用于图谱落地；**CBLUE 评测始终用原生 9 类**：

| CMeEE | 产品类型 | 说明 |
|-------|----------|------|
| dis | Disease | |
| sym | Symptom | |
| dru | Drug | |
| ite | Exam | |
| pro | Treatment | |
| dep | Department | |
| bod / equ / mic | （评测保留，产品链可丢弃） | 见 `CMEEE_TO_PROJECT` |

详见 `docs/scoring_evidence_consistency.md`。
