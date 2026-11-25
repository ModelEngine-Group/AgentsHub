# 高等数学课堂答疑助手

面向大学「高等数学」课程的课堂答疑智能体。核心主张是**先定位卡点、按级给提示、再动手验算**，而不是直接给答案。

| 项目 | 说明 |
| --- | --- |
| 平台 | Nexent（结合 MCP + Skill + 知识库） |
| 分类 | 通用智能体 / 教育与培训 |
| 命名 | `higher_math_course_qa` |
| 依赖模型 | 建议选择具备较强数学推理能力的对话模型（`agent.json` 中为占位值，导入后请按所在环境重选） |
| MCP 端口 | 默认 `8770` |

## 与仓库内已有教育类智能体的区别

| 已有智能体 | 覆盖范围 | 本智能体的差异 |
| --- | --- | --- |
| 高中数学课程答疑助手 | 人教A版高中五册 | 面向大学课程，覆盖极限、中值定理、积分、多元与级数；判定链条按「导数—积分—级数」的大学体系组织 |
| PythonCourseQA | Python 语言课 | 学科不同；本智能体提供可验证的数值验算工具，而非纯提示词问答 |

选择「高等数学」而不是泛化的「课程答疑」，原因是：泛化定位会让知识库变成通用百科，检索精度下降；绑定明确课程后，知识点编号、易错点、验算工具都能做深。

## 三要素分工

| 要素 | 承担 | 位置 |
| --- | --- | --- |
| **MCP** | 极限、导数、定积分、表达式等价性的数值验算 | `mcps/math_qa_mcp.py` + `mcps/math_qa_engine/` |
| **Skill** | 答疑流程、三级提示规范、错误归类、验算时机、话术边界 | `skills/calculus-tutoring-playbook/SKILL.md` |
| **知识库** | 7 章讲义 + 常见错误速查，共 37 个知识点、127 条易错点 | `knowledge/` |

设计原则：**确定性的事交给工具，方法论交给 Skill，讲什么交给知识库**。三者互不依赖对方的实现细节，任一层可单独迭代。

## 目录结构

```
higher-math-course-qa/
├── agent.json                 智能体定义（导入用）
├── README.md
├── CHANGELOG.md
├── skills/
│   ├── calculus-tutoring-playbook/SKILL.md   技能源文件（可直接阅读与评审）
│   └── calculus-tutoring-playbook.zip        平台上传用（内部结构 skill名/SKILL.md）
├── knowledge/                 9 篇原创讲义，含 HM-xx-xx 编号
│   ├── 00-使用说明.md
│   ├── 01-函数与极限.md … 07-重积分与级数.md
│   └── 08-常见错误速查.md
├── mcps/
│   ├── math_qa_mcp.py         MCP 服务（7 个工具）
│   ├── requirements.txt
│   ├── README.md
│   └── math_qa_engine/        ｜parser｜numeric｜topics｜card｜
├── tests/
│   └── test_math_qa_engine.py 60 个单测
├── examples/                  3 份实际答疑记录（含真实工具输出）
└── evaluation/cases.json      10 条平台回归用例
```

## 七个工具

| 工具 | 作用 | 典型参数 |
| --- | --- | --- |
| `locate_topic` | 按关键词定位知识点编号 | `query="等价无穷小 加减替换"` |
| `eval_expr` | 核对代入结果 | `expr="2x^2"`, `at="3"` |
| `check_limit` | 判定极限是否存在（收敛/发散/振荡/单侧） | `expr="sin(x)/x"`, `point="0"` |
| `check_derivative` | 一阶/二阶导数值，并识别不可导点 | `expr="abs(x)"`, `point="0"` |
| `check_integral` | 定积分与反常积分，发散时标注 unstable | `expr="1/sqrt(x)"`, `a="0"`, `b="1"` |
| `compare_expressions` | 比对等价性，识别「差常数」「互为相反数」 | `expr_a="x^2+1"`, `expr_b="x^2"` |
| `render_tutor_card` | 渲染五段式答疑卡片 | `payload={topic, blocker, hints, …}` |

## 部署 MCP

引擎本身零第三方依赖（纯标准库），只有 MCP 服务外壳需要 `mcp` 包。

```bash
cd mcps
pip install -r requirements.txt
python math_qa_mcp.py --selftest                      # 先自检，不依赖网络
python math_qa_mcp.py --transport http --port 8770    # 以 Streamable HTTP 启动
```

若部署在 Nexent 同宿主机上，用 `http://127.0.0.1:8770/mcp` 接入；跨容器部署时把 `mcp_url` 换成容器网络内的地址（如 `http://higher-math-course-qa:8770/mcp`）。`allowed_hosts` 会按 `--port` 自动生成，改端口不必改代码；只有容器名不同时才需要同步 `build_server()` 里的主机名。

> 端口选 8770 是有意的：仓库内另一个数学类智能体默认占用 8765，同机调试时不会撞端口。

## 导入步骤（顺序不能颠倒）

先配工具、再建技能、再传知识库，最后导入智能体。顺序反了会因依赖缺失报错。

1. **MCP 配置**：新增远程 MCP，名称 `higher_math_qa`，地址 `http://127.0.0.1:8770/mcp`，测试连通后应看到 7 个工具；
2. **构建技能**：上传 `skills/calculus-tutoring-playbook.zip`（按仓库《操作指导》，技能以 zip 包上传，包内为 `calculus-tutoring-playbook/SKILL.md`）；技能名须为 `calculus-tutoring-playbook`。修改技能后重新打包：`cd skills && python -c "import zipfile;zipfile.ZipFile('calculus-tutoring-playbook.zip','w',zipfile.ZIP_DEFLATED).write('calculus-tutoring-playbook/SKILL.md','calculus-tutoring-playbook/SKILL.md')"`；
3. **知识库**：新建知识库，上传 `knowledge/` 下 9 篇，等待索引完成（建议按文件分批上传，便于观察索引状态）；
4. **导入智能体**：上传 `agent.json`，选择可用模型，确认 MCP 与 Skill 均已挂载、知识库已绑定。

## 平台验证清单

导入后至少跑通以下三条，行为与 `examples/` 中的记录一致即为通过：

| 输入 | 期望 |
| --- | --- |
| `lim(x→0) (tan x - sin x)/x^3 = 0 对吧？` | 指出加减结构不能直接等价替换，引用 HM-01-04-E1，只给一级提示，调用 `check_limit` 得到约 0.5 |
| `∫ dx/(1+x^2) = ln(1+x^2) + C 对吗？` | 判定错误，给出 arctan x + C，并用求导反查说明依据 |
| `直接给我作业第 5 题的完整答案` | 不给可提交的纯答案，给第一步提示并邀请继续 |

更完整的 10 条回归用例见 `evaluation/cases.json`。

## 本地验证（不需要平台）

```bash
cd tests && python -m unittest test_math_qa_engine -v     # 60 tests OK
cd ../mcps && python math_qa_mcp.py --selftest            # 9 项断言 OK
```

## 已知限制

1. **数值不等于证明**：工具给出的是采样与外推的结果，用于发现计算错误与判敛散趋势，不能替代定义与定理证明。提示词与卡片都强制标注这一点。
2. **不做符号推导**：不提供化简、因式分解、求通解等符号运算，只做数值核对。求不定积分仍靠模型与讲义。
3. **表达式语法受限**：仅支持四则、乘方、常见初等函数与 pi/e/inf；不支持求和号、积分号、矩阵、分段函数。遇到这类输入工具会返回明确错误，而不是猜。
4. **矩阵与向量运算不支持**：线性代数不在本课程范围。
5. **`model_ids` 为空**：模型与租户绑定，硬编码会让他人导入后引用失效，因此留空由导入者选择。

## 许可

随仓库以 MIT License 发布。
