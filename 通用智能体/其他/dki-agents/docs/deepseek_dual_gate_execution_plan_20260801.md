# dki-agents 双门槛修复执行书（供 DeepSeek 严格执行）

> 生效日期：2026-08-01  
> 性质：可执行计划，不是通过承诺。  
> 总原则：证据优先、远端单 NPU 测试、禁止篡改历史、禁止把开发集当盲测。

## 1. 最终目标

只有同时满足以下两项，才能生成新的最终提交候选：

1. CMeIE 官方 dev 的完整 text-to-SPO Micro-F1 大于或等于 0.50。
2. NL2SQL 一次性消费的独立 holdout 同时满足：

   - n_total 大于或等于 120；
   - n_hard 大于或等于 40，其中 hard、extra、extra-hard 合计；
   - 100 乘以 n_correct 严格大于 85 乘以 n_total；
   - 100 乘以 n_hard_correct 大于或等于 70 乘以 n_hard；
   - source、DB、evaluator、依赖、manifest、holdout 哈希全部匹配；
   - 不可变 consumption receipt 存在。

任意一项失败，最终状态只能是：

~~~
NOT_READY_FOR_SUBMISSION
~~~

## 2. 当前已知事实

| 项目 | 当前数值 | 正确解释 |
|---|---:|---|
| CMeEE 官方 dev | F1 0.613020 | 历史有效证据，必须字节不变 |
| CMeEE NPU 吞吐 | 326.098 docs/s | 历史有效证据 |
| CMeEE NPU/CPU | 17.7295x | 历史有效证据 |
| CMeIE 完整 text-to-SPO | F1 0.1911 | 未通过 |
| CMeIE gold-NER 条件化官方 dev | F1 0.4319，P 0.4268，R 0.4371 | 当前最优开发确认；不是端到端，仍低于 0.50 |
| CMeIE 激活谓词 | 19/44 | 长尾关系仍大量未激活 |
| NL2SQL 当前 hash-bound development | 102/180 = 0.5667，hard 67/128 = 0.5234 | 当前可验证链路，未通过 |
| NL2SQL AtomCode 旧回执 | 172/180 = 0.9556 | development_selection_used_re_evaluation，不是独立盲测 |

必须正确理解：

- CMeIE 的 gold-NER 条件化结果也只有 0.4319，所以只优化 NER 不能保证端到端通过；核心还包括关系分类、长尾谓词、候选噪声和 SPO 解码。
- NL2SQL 的第一优先级不是继续对旧题调参，而是解释 172/180 与 102/180 的源码、DB、输入和 evaluator 差异；旧回执不能当作当前性能或 blind 结果。

## 3. 路径、环境和硬约束

### 3.1 路径

本地项目根：

~~~
C:\Users\shenh\Desktop\比赛专用\ModelEngine开源项目贡献赛\dki-agents
~~~

固定远端 Python：

~~~
/root/dki-agents-resume-20260723/venv/bin/python
~~~

新产物只能写入：

~~~
公开本地证据：outputs/remote_runs/<RID>/
私有本地证据：outputs/private/<RID>/
远端实验根：/root/dki-agents-<RID>/
提交候选：dist/submission/<RID>/
~~~

### 3.2 运行 ID

每次纳入源码、配置、评估器或数据清单变更时，生成：

~~~
RID=YYYYMMDDTHHMMSSZ-g<HEAD前7位>-t<included-tree-SHA前12位>
~~~

目标目录已经存在时必须失败。绝不复用、绝不覆盖。

### 3.3 远端单 NPU 要求

所有训练、pytest、评估、性能测试都必须在远端单个逻辑 NPU 上运行：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=0
export DEVICE_ID=0
~~~

每个 RID 开始时保存以下完整输出：

~~~bash
npu-smi info
uname -a
cat /etc/os-release
/root/dki-agents-resume-20260723/venv/bin/python -V
/root/dki-agents-resume-20260723/venv/bin/python - <<'PY'
import torch
import torch_npu
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("npu_count=", torch.npu.device_count())
if torch.npu.device_count():
    print("device_name=", torch.npu.get_device_name(0))
PY
~~~

最终报告只使用实际返回的设备名称。现有历史运行出现过 Ascend910_9382；没有新验证时不得写成其他型号。

### 3.4 禁止事项

执行者绝对禁止：

- git reset、git checkout、git clean；
- 删除、覆盖或格式化已有未提交改动；
- 整仓提交、推送、切换分支；
- 覆盖 outputs/remote_runs/plan_v1/ 中任何文件；
- 覆盖历史模型、ZIP、manifest、receipt、OM、ONNX；
- 将密钥、.env、原始数据、模型、holdout、详细盲测结果打进 ZIP；
- 在本地执行功能、性能或 pytest；所有测试必须远端 NPU；
- 用 CMeIE 官方 dev 选择种子、模型、阈值或修改规则；
- 将 CMeIE 官方 dev 写成 blind；
- 使用超过 1B 参数的模型；
- 打印、全文搜索或泄露正式 holdout 的题目、gold SQL、预测 SQL、失败详情。

本地仅允许编辑源码、生成清单、计算哈希、同步远端、下载回执和打包。实际测试必须远端执行。

## 4. 统一证据协议

每个 RID 必须新建：

~~~
outputs/remote_runs/<RID>/progress.json
~~~

最小内容：

~~~json
{
  "rid": "<RID>",
  "phase": "<phase>",
  "status": "not_started|running|passed|failed|blocked",
  "git_head": "<HEAD>",
  "source_tree_sha256": "<SHA256>",
  "started_at": "<UTC>",
  "completed_at": null,
  "commands": [],
  "artifacts": [],
  "gate_result": {
  "next_action": ""
}
~~~

每条远端命令必须记录：

- 去敏后的命令；
- UTC 开始/结束时间；
- 完整 stdout/stderr；
- exit code；
- 输入/输出 SHA-256；
- Python、依赖、NPU、CANN 环境信息；
- pytest 的 JUnit XML、收集到的 node ID 和 skip 原因。

禁止只保存日志尾部或只保存返回码。

## 5. 阶段 0：冻结基线

### 5.1 工作树快照

写入当前 RID：

~~~powershell
git status --short
git rev-parse HEAD
git diff --stat
git diff -- README.md docs/model_card.md
~~~

不得使工作树变干净。

### 5.2 历史证据守卫

执行前和最终打包前都计算并比较 SHA-256：

~~~
docs/remote_npu_acceptance_20260725.md
outputs/remote_runs/plan_v1/submission/verification.json
outputs/remote_runs/plan_v1/cmeie_predicate_threshold_calibration_h05_s43.json
~~~

任一文件哈希变化，立即停止，不得生成新提交包。

### 5.3 included-tree

建立排序后的 included-files manifest。每项必须有：

~~~
relative_path
size
sha256
~~~

included-tree SHA 必须由该清单得出，而不能只使用 Git HEAD。合资格未跟踪的源码、测试、配置和文档必须纳入；private、模型、原始数据、密钥和远端归档必须排除。

### 5.4 远端隔离副本

将当前 included-tree 同步到：

~~~
/root/dki-agents-<RID>/
~~~

同步后比较远端/本地 manifest，必须完全一致。不得在历史远端目录直接做实验。

# 6. NL2SQL 工作流（优先执行）

## N0：解释旧 172/180 与当前 102/180

不得立即修改规则。首先创建：

~~~
outputs/remote_runs/<RID>/nl2sql/lineage_comparison.json
~~~

读取但不修改：

~~~
outputs/private/eval_v4b_receipt_20260801T0330/re_evaluation.json
~~~

旧回执已知绑定：

~~~
dataset_role=development_selection_used_re_evaluation
source_receipt=outputs/remote_runs/20260726T155404Z-g30fce24-t05cadfe54bd4/nl2sql/v4b_development_initial.json
database_sha256=e7721fbaf49a81cc157af453728f004edb4a08aed0f3c449e1e40036562455d8
engine_source_sha256=ec64e3ef5e2ad93d2a1380247a1e7b4fe6c292670e07fd6e145ffb59446463c7
~~~

同时记录当前链路的 SHA：

- dki/analysis/nl2sql.py；
- dki/analysis/nl2sql_contract.py；
- 所有 dki/analysis/nl2sql_semantic*.py；
- strict evaluator；
- development 输入；
- SQLite DB；
- requirements / dependency lock；
- included-tree；
- sqlglot 版本。

差异只能分类为：

~~~
SOURCE_DRIFT
DATABASE_DRIFT
DATASET_DRIFT
EVALUATOR_DRIFT
PROTOCOL_DRIFT
ENCODING_DRIFT
DEPENDENCY_DRIFT
NON_REPRODUCIBLE_HISTORICAL_RECEIPT
~~~

禁止用环境不同作为结论。

## N1：隔离复现矩阵

若能从现有 hash-bound 归档精确得到历史源码和历史 DB，在新的远端目录运行四格矩阵：

| 引擎 | DB | 必须记录 |
|---|---|---|
| 历史引擎 | 历史 DB | 正确数、coverage、所有 hash |
| 历史引擎 | 当前 DB | 正确数、coverage、所有 hash |
| 当前引擎 | 历史 DB | 正确数、coverage、所有 hash |
| 当前引擎 | 当前 DB | 正确数、coverage、所有 hash |

四种组合必须使用同一个 development 输入和同一个严格 evaluator。

禁止 git checkout 恢复旧树；只能从已有归档复制到新的隔离目录。

如果不能精确恢复历史源码或 DB，写入：

~~~
NON_REPRODUCIBLE_HISTORICAL_RECEIPT
~~~

随后禁止再把 172/180 当作当前性能。

N1 通过前必须回答：

1. 旧 172/180 的源码、DB、输入和 evaluator 分别是什么；
2. 当前 102/180 的源码、DB、输入和 evaluator 分别是什么；
3. 具体差异属于哪些分类；
4. 哪一条链是当前合规、可复现链路。

## N2：仅用 development 修复规则

旧 v4b 只能作为 development。所有调优都只能读取 development。

每一个错误类别必须严格按下列顺序处理：

1. 新增一个最小 regression fixture；
2. 先在远端运行，确认修复前 fixture 失败；
3. 实现最小、泛化的规则；
4. 重跑 fixture；
5. 运行全部 NL2SQL 测试；
6. 运行 SQL AST 安全测试；
7. 远端完整 development 评估；
8. 保存新的 hash-bound receipt。

禁止按以下内容硬编码：

- 题目 ID；
- 完整 question；
- 完整 gold SQL；
- 具体病人 ID；
- 单一疾病、地区、时间的专属答案。

优先排查顺序：

1. nl2sql.py 对 semantic 模块的 dispatcher、注册和优先级；
2. UTF-8 规范化与中文关键词匹配；
3. derived table / CTE 的别名、输出列和血缘；
4. 多表 JOIN、COUNT DISTINCT、聚合排序、TopK；
5. 时间边界、cohort、retention、窗口函数；
6. EXCEPT、INTERSECT、NOT EXISTS、相关子查询；
7. strict result contract：列数、列顺序、完整结果集、超过 200 行结果。

每条新增规则至少配一个正例和一个相邻负例。

development 晋级门：

~~~
overall >= 0.90
hard/extra/extra-hard >= 0.78
coverage >= 0.95
原44题 = 100%
已知120题 >= 0.9333
engine = rule
SQL安全测试全部通过
0 failure / 0 error / 0 xpass / 无新增 skip
~~~

不满足时，禁止生成或消费正式 holdout。

## N3：独立 holdout

正式 holdout 必须由独立策展方提供。独立策展方只能看到：

- 数据库 schema；
- 允许表；
- 数据库只读副本；
- JSONL 格式；
- SQL 安全规则；
- difficulty/category 定义。

独立策展方禁止看到：

- nl2sql.py 和 semantic 模块；
- development 题目；
- 现有测试题；
- 当前失败详情；
- 规则实现；
- 旧回执中的 detailed results；
- 调优记录。

如果执行者已经读取过引擎源码、开发题或失败样本，它不能充当独立策展方。

没有独立策展方时，停止并写：

~~~
BLOCKED_INDEPENDENT_HOLDOUT_MISSING
~~~

不得自己编题、自己调规则，然后将其称为 blind。

建议独立策展方先生成 300 题：

~~~
90 medium
150 hard
60 extra/extra-hard
~~~

最终冻结后至少保留：

~~~
holdout >= 120
hard/extra/extra-hard >= 40
development >= 150
~~~

题目必须覆盖 filtering、aggregation、grouping、TopK、multi-table join、temporal、KG、HAVING、nested/correlated、CTE、set operation、window function。

每条题必须有：

~~~
id
normalized question
gold_sql
difficulty
category
~~~

每条 gold SQL 必须：

- 只有一条 SQLite 只读查询；
- 仅访问 patients、visits、kg_entities、kg_triples；
- 能被 sqlglot 解析；
- 能在密封 DB 完整执行；
- 不含写操作、ATTACH、危险 PRAGMA 或非法表；
- 不依赖 200 行截断。

development/holdout 之间必须同时满足：

- 规范化 question 无交集；
- 近重复无交集；
- 规范化 AST skeleton 无交集。

## N4：密封和一次性消费

密封下列 hash：

- rule engine；
- 所有 semantic 模块；
- evaluator；
- blind protocol；
- 依赖锁和 sqlglot 版本；
- DB；
- task1/task2/task3 provenance；
- KG；
- included-tree；
- development manifest；
- holdout manifest 与 holdout 内容；
- engine=rule。

评估程序必须严格顺序执行：

1. 验证全部 hash；
2. 验证 DB provenance；
3. 验证 holdout 规模和 hard 数；
4. 验证 development/holdout 无文本、近重复、AST skeleton 交集；
5. 读取 holdout 前原子 create-once 写入 consumed-started.json；
6. 一次性运行所有题；
7. 每题只调用一次引擎；
8. overall/hard 从同一份详细结果派生；
9. detailed results 只写 outputs/private/<RID>/；
10. 公开回执不包含问题、gold SQL、预测 SQL 或错误样本。

marker 创建后，只要崩溃、环境错误或指标失败，该 holdout 都已消费；必须写 consumed-invalid 或 failed-attestation，不得重跑。下一轮必须使用新的独立 holdout。

# 7. CMeIE 工作流

## C0：历史拒绝路线

以下仅作 rejected ablation 记录。没有新的、可检验的假设时不得重复：

| 路线 | 结果 |
|---|---:|
| GPLinker seed 42/43 | F1 0 |
| predicate-span sparse rescore | F1 0.0804 |
| predicate-span BCE frozen4 | F1 0.2150 |
| CMeEE gazetteer/entity backend | F1 0.1162 / 0.1299 |
| pair gap 120 | F1 0.1294 |
| all-negative 旧端到端 | F1 0.3448 |
| 双模型 ensemble | F1 0.3915 |

逐谓词 0.05 阈值的 0.4319 仅是历史参考；新最终路线不得用 44 个自由阈值绕开泛化门。

## C1：family-disjoint split

建立 create-once manifest：

~~~
fit = 80%
calibration = 10%
outer = 10%
seed = 20260725
~~~

family 规则：

- 有 @：取 @ 前的规范化标题；
- 无 @：取规范化后的前 32 个有效字符；
- 完全重复和近重复文本必须同 family；
- family 和文本 hash 不得跨 split；
- 按 family 大小与 44 谓词频率做确定性分配。

manifest 必须记录：原始数据 SHA、行 SHA、family SHA、split 行数、谓词分布、交集检查、生成脚本 SHA、included-tree SHA。

outer 在模型、seed、阈值全部冻结之前禁止读取。

## C2：先抬高关系分类条件化上限

目的：使用 gold argument span 先证明关系分类器本身能够超过当前 0.4319。此阶段失败时，不得进入 OOF。

固定模型：

~~~
hfl/chinese-roberta-wwm-ext
约102M参数
max_length=320
~~~

每个实体对输入必须含：

- 原始句子；
- subject marker；
- object marker；
- subject/object 原生类型；
- 主客体方向；
- 字符距离；
- 同句标志。

为 marker 注册 tokenizer special tokens，保存 tokenizer hash。

输出必须有：

1. pair-existence 二分类头；
2. 44 维 predicate multi-label 头。

不得把多谓词对强制成单标签 softmax。

训练样本包含：

- 全部 gold 正实体对；
- 同句非 gold 对；
- 反向对；
- 类型错误对；
- 边界扰动对；
- 邻近 hard negative。

负样本课程：

~~~
epoch 1: 正负 1:1
epoch 2+: 正负最多 1:3
每句负样本上限 32
~~~

长尾处理只允许使用 fit 信息：

- class-balanced sampling；
- inverse-sqrt frequency 权重；
- 权重上限 20；
- asymmetric focal loss 或 class-balanced BCE；
- 基于 fit 谓词先验的 logit adjustment。

logit adjustment 仅从预声明网格选择：

~~~
tau = 0.0, 0.5, 1.0
~~~

训练参数：

~~~
seeds = 42,43,44
epochs = 8
batch_size = 8
gradient_accumulation = 4
effective_batch = 32
encoder_lr = 2e-5
head_lr = 1e-4
warmup = 0.10
weight_decay = 0.01
gradient_clip = 1.0
FP16/NPU
early_stopping_patience = 2
~~~

只在 calibration 搜索：

~~~
pair threshold = 0.30,0.40,0.50,0.60,0.70
predicate threshold = 0.30,0.40,0.50,0.60,0.70
schema constraint = off,on
~~~

共 50 个固定组合。选择顺序：F1、precision、较高全局阈值、schema on 优先。

C2 门：先跑 seed 42 和 43。gold argument 条件化 calibration 必须同时达到：

~~~
F1 >= 0.55
Precision >= 0.50
Recall >= 0.50
激活谓词 >= 30/44
~~~

若 seed 42、43 都低于 0.50，停止该架构；不得跑 seed 44、不得做 OOF、不得读 outer。

## C3：CMeIE 专用 argument-NER

仅在 C2 通过后开始。

固定模型：

~~~
hfl/chinese-roberta-wwm-ext
GlobalPointer
11 个 CMeIE 原生类型
max_length=320
~~~

禁止复用 CMeEE 九类型实体前端。

标签必须通过 fast tokenizer offset mapping 对齐。必须测试：

- 嵌套实体；
- 重复 mention；
- 同表面词多位置；
- 多谓词实体对；
- 同句多个合法 span；
- dropped-label 统计。

完整训练前，先用固定 64 条 fit 样本过拟合，最多 30 epoch。必须同时满足：

~~~
训练集 span F1 >= 0.98
dropped labels = 0
预测实体非零
无 NaN
无 optimizer step 跳过
~~~

失败时只允许检查 offset、sparse loss、mask、label index、pointer head、optimizer 和 FP16，禁止直接开始完整训练。

完整训练解冻策略：

~~~
epoch 1-2: encoder 冻结，head LR=5e-4
epoch 3-4: 解冻顶部4层，encoder LR=1e-5，head LR=1e-4
epoch 5-8: 全量解冻，encoder LR=2e-5，head LR=1e-4

batch_size=8
gradient_accumulation=4
effective_batch=32
warmup=0.10
weight_decay=0.01
head clip=1.0
encoder clip=0.5
FP16/NPU
patience=2
~~~

C3 门：calibration 必须同时达到：

~~~
typed mention Precision >= 0.70
typed mention Recall >= 0.88
typed mention F1 >= 0.75
gold argument-pair candidate recall >= 0.85
候选总量 <= 当前 CMeEE 候选量的 2 倍
~~~

任一不满足，禁止 OOF、禁止 outer；保存 rejected receipt。

## C4：family-disjoint OOF hard negatives

只有 C2 和 C3 都通过才能执行。

在 fit 内建立五折 article-family-disjoint OOF。每折模型只能预测未参与本折训练的样本。

收集：

- 漏检实体；
- 错边界实体；
- 错类型实体；
- 非 gold 实体；
- 反向实体对；
- 同句非 gold 对；
- 高置信错误谓词。

使用这些 OOF 噪声重训 C2 的关系分类器。最终只允许：

- 一个全局 pair threshold；
- 一个全局 predicate threshold；
- schema on/off。

禁止 44 维独立阈值搜索。

C4 门：

~~~
完整 calibration text-to-SPO F1 >= 0.45
Precision >= 0.42
Recall >= 0.42
激活谓词 >= 30/44
候选量 <= 基线 2 倍
~~~

## C5：outer 和官方 dev

顺序不可改变：

1. 完成 seed 42、43、44；
2. 每个 seed 在 calibration 冻结阈值；
3. 按 calibration 选择最终 seed；
4. 保存模型、阈值、数据、tokenizer、源码、评估器 hash；
5. 一次性读取 outer；
6. outer 后不得根据成绩更换 seed 或阈值；
7. outer 通过后，只执行一次官方 dev 开发确认。

outer 门：

~~~
三 seed median F1 >= 0.50
最差 seed F1 >= 0.48
每个 seed Precision >= 0.45
每个 seed Recall >= 0.45
family/hash 无泄漏
~~~

最终 CMeIE 门：

~~~
官方 dev 完整 text-to-SPO F1 >= 0.50
~~~

失败后保留默认 rule 链路，不得伪造通过。

## C6：最后备用路线

只有 C2 的关系分类器明确失败时，才允许重新审计 GPLinker。

任何全量训练前，必须在固定 32 或 64 条样本上证明：

~~~
训练集 SPO F1 >= 0.98
entity/head/tail 三头均有非零正例
解码结果非空
dropped labels = 0
无 NaN
~~~

不满足则判为实现或标签错误，禁止全量训练和多 seed 消耗。

# 8. 测试要求

每个重要阶段都必须保存：

- pytest collection node ID；
- 完整 stdout/stderr；
- JUnit XML；
- 每个 skip 的 node ID 和理由；
- exit code；
- 环境指纹。

要求：

~~~
原有 node ID 不能减少
0 failure
0 error
0 xpass
无新增 skip
新测试只能增加 passed
~~~

必须补足测试：

### CMeIE

- family、重复文本、近重复不跨 split；
- split manifest 确定性；
- offset 对齐；
- 嵌套实体、重复 mention、多谓词；
- 跨句拒绝；
- multi-label predicate loss；
- 全局阈值与 hash 绑定；
- 空候选不推理；
- OOF 不预测自身训练样本；
- create-once；
- 候选数上限。

### NL2SQL

- 缺字段、重复 ID、重复 question；
- 非法 difficulty/category；
- 多语句、写 SQL、非法表、非法 AST；
- AST skeleton 跨 split；
- source/DB/evaluator/manifest 篡改；
- create-once 并发竞争；
- marker 前后崩溃；
- 每题仅一次引擎调用；
- 0.85、0.70 边界；
- 超过 200 行结果；
- 严格列数；
- private 数据绝不进入 ZIP。

# 9. 发布门和最终报告

只有以下全部为真才能构建新的 clean-room RC：

~~~
CMeIE 官方 dev 完整 text-to-SPO F1 >= 0.50
NL2SQL 独立 holdout overall > 0.85
NL2SQL 独立 holdout hard >= 0.70
NL2SQL holdout >= 120
NL2SQL hard >= 40
CMeEE 历史证据 hash 未变化
远端全仓测试零失败
manifest/secret/private-data 检查全部通过
~~~

新 ZIP 必须 create-once，且禁止包含：

~~~
outputs/private/
holdout
gold SQL
详细 NL2SQL 结果
原始 CBLUE 数据
模型 checkpoint
OM/ONNX
.env
SSH 密钥
远端归档
未登记文件
符号链接
路径穿越项
~~~

最终报告必须包含：

1. 实际 NPU、驱动、CANN、torch、torch_npu、Python、OS 信息；
2. CMeIE 的 gold-pair、gold-NER 条件化、完整 text-to-SPO 三栏指标；
3. CMeIE 的模型、tokenizer、split、阈值、源码和评估器 hash；
4. NL2SQL 的聚合数、coverage、abstention 和绑定 hash；
5. CMeEE 历史证据的前后 hash 比较；
6. pytest node ID、passed/skipped/failure/error/xpass；
7. 新 RC 的 SHA256SUMS；
8. 唯一最终结论：READY_FOR_SUBMISSION 或 NOT_READY_FOR_SUBMISSION。

# 10. 不可改变的执行顺序

1. 冻结当前工作树和历史证据；
2. 创建 RID、included-tree 和远端隔离副本；
3. 完成 NL2SQL 172/180 与 102/180 血缘对照；
4. 仅用 development 修复 NL2SQL；
5. development 通过后，等待独立策展 holdout；
6. 密封并一次性执行 NL2SQL blind；
7. 冻结 CMeIE family-disjoint split；
8. 先验证 CMeIE gold-argument 关系分类器；
9. C2 通过后才训练 11 类型 argument-NER；
10. C3 通过后才做五折 OOF hard negatives；
11. calibration 通过后才读取 outer；
12. outer 通过后只执行一次官方 dev；
13. 两个门都通过后运行远端全仓测试；
14. 比对历史证据 hash；
15. create-once 构建 clean-room RC；
16. 生成最终报告。

任何阶段失败，必须停在该阶段并写失败回执；不得跳过门槛、复用旧结果，或把 development 写成 blind。


# 11. 给 DeepSeek 的启动提示词

将以下提示词与本文件路径一起发送给 DeepSeek：

~~~
你是 dki-agents 的执行代理。请先完整阅读：
C:\Users\shenh\Desktop\比赛专用\ModelEngine开源项目贡献赛\dki-agents\docs\deepseek_dual_gate_execution_plan_20260801.md

随后严格按该文档开始实际执行，不要只复述计划，也不要自行降低任何门槛。

硬性规则：
1. 保留当前所有未提交和未跟踪文件；禁止 git reset、git checkout、git clean、整仓提交、覆盖历史产物。
2. 所有功能、训练、评估、pytest 和性能测试必须在远端单逻辑 NPU 0 上执行；固定使用 /root/dki-agents-resume-20260723/venv/bin/python。
3. 每次新实验必须生成新 RID，使用新的本地证据目录、远端隔离目录和 create-once 文件。
4. 先执行阶段 0 和 NL2SQL 的 N0/N1 血缘对照；在解释 172/180 与 102/180 的绑定差异前，不允许调 NL2SQL 规则。
5. CMeIE 必须先通过 C2 的 gold-argument 条件化关系分类门，再进行 C3 的 11 类型 argument-NER；未通过不得做 OOF 或读取 outer。
6. CMeIE 官方 dev 只能在 outer、模型和阈值冻结后执行一次开发确认；不得作为选参数据，也不得称为 blind。
7. NL2SQL 正式 holdout 必须由未接触引擎源码和 development 的独立策展方提供。缺少独立策展方时，明确写 BLOCKED_INDEPENDENT_HOLDOUT_MISSING；不得自己造题再称为盲测。
8. 每个阶段都保存完整命令、stdout、stderr、exit code、哈希、JUnit XML、node ID、skip 原因和 progress.json。
9. 失败时停在对应 gate，写 rejected 或 failed receipt；不得跳过、重跑已消费 holdout、复用旧结果或伪造通过。
10. 最终只允许输出 READY_FOR_SUBMISSION 或 NOT_READY_FOR_SUBMISSION，并给出证据路径和精确指标。

现在开始执行阶段 0。每完成一个 gate，先报告：RID、修改文件、远端命令、哈希、测试结果、是否通过和下一步；然后继续下一个允许阶段。
~~~

