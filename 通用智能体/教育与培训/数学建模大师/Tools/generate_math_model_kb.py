from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent.parent


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def build_paper_library() -> None:
    base = ROOT / "KB_优秀论文库"
    base.mkdir(parents=True, exist_ok=True)

    readme = dedent(
        """
        # KB_优秀论文库

        本库用于训练智能体学习优秀论文的结构组织、摘要写法、模型组合和图表表达。

        ## 内容结构
        - `00_使用说明.md`：如何使用本库进行论文反向学习
        - `01_官方来源索引.md`：可追溯的竞赛官方入口与年度页面
        - `02_论文拆解模板.md`：统一拆解模板，便于沉淀结构化样本
        - `03_写作风格观察清单.md`：摘要、正文、图表、附录观察清单
        - `04_高质量论文样本清单.csv`：推荐优先阅读入口（链接索引）

        ## 说明
        1. 优先收录官方/组委会/赛事站点的公开入口，避免版权不清晰的二次转载。
        2. 若某赛事未公开完整论文全文，保留“结果页 + 官方展示页 + 评审点评”用于结构学习。
        3. 可以按年份持续补充，建议每年更新一次。
        """
    )
    write_file(base / "README.md", readme)

    usage = dedent(
        """
        # 使用说明

        ## 推荐工作流
        1. 在 `04_高质量论文样本清单.csv` 中选 3-5 篇同类赛题论文。
        2. 每篇使用 `02_论文拆解模板.md` 拆解为结构化笔记。
        3. 将摘要写法、模型链路、图表表达填入你自己的“写作策略库”。
        4. 用 `03_写作风格观察清单.md` 做一次质量复查，形成可复用模板。

        ## 给智能体的提示词建议
        - “先输出论文骨架，再补每章要点，最后填图表和附录代码位置。”
        - “摘要中按：问题 -> 方法 -> 结果 -> 结论 的四句结构写作。”
        - “模型组合优先采用主模型 + 对照模型 + 稳健性检验结构。”
        """
    )
    write_file(base / "00_使用说明.md", usage)

    source_index = dedent(
        """
        # 官方来源索引（更新时间：2026-05-23）

        ## 1) 国赛优秀论文（全国大学生数学建模竞赛）
        - 论文展示总入口（中国大学生在线）  
          https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/qkt_sxjm_lw_lwzs.shtml
        - 2025 论文展示入口  
          https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/2025qgdxssxjmjslwzs/
        - 2025 A题论文展示示例  
          https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2025qgdxssxjmjslwzs_2025atlw/251101/2022729.shtml
        - 2025 C题论文展示示例  
          https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2025qgdxssxjmjslwzs_2025ctlw/251101/2022740.shtml

        ## 2) MCM/ICM Outstanding Papers（官方结果与规则入口）
        - COMAP MCM/ICM 主入口  
          https://www.comap.com/contests/mcm-icm
        - Contest Rules / Instructions  
          https://www.contest.comap.org/undergraduate/contests/mcm/instructions.html
        - 2025 结果总入口（含各题结果 PDF）  
          https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/index.html
        - 2025 Problem A 结果 PDF  
          https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/2025_MCM_Problem_A_Results.pdf
        - 2025 Problem B 结果 PDF  
          https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/2025_MCM_Problem_B_Results.pdf

        ## 3) 华为杯研究生数学建模（官方竞赛平台）
        - 中国研究生数学建模竞赛主页（当届通知、赛题与公告）  
          https://cpipc.acge.org.cn/cw/hp/4
        - 赛事介绍页  
          https://cpipc.acge.org.cn/cw/contestIntrol/4
        - 竞赛新闻列表（含年度邀请函）  
          https://cpipc.acge.org.cn/cw/contestNews/list/4/1

        ## 4) 亚太杯优秀论文（APMCM）
        - APMCM 官方站  
          https://apmcm.org/
        - 中文入口  
          https://www.apmcm.org/?language=cn
        - 英文入口  
          https://apmcm.org/?language=en

        ## 5) MathorCup 优秀论文/赛事资料
        - MathorCup 官方站  
          https://www.mathorcup.org/
        - 赛题与评奖栏目  
          https://mathorcup.org/cooperation
        - 奖项/公告栏目  
          https://mathorcup.org/prize

        ## 6) 深圳杯优秀论文（及评审点评）
        - “深圳杯”数学建模挑战赛入口（m2ct）  
          https://www.m2ct.org/modular-list.jsp?menuType=bfInstruc&pageType=smxly
        - 2025 深圳杯评审总结点评  
          https://www.cmathc.org.cn/mcm/news/394.html
        - 2017 深圳杯优秀论文公开展示通知（含附件）  
          https://www.mcm.edu.cn/upload_cn/node/431/OWbhyOK2aae9070a2ac76a31390df8661d8c8672.pdf
        - 中国大学生在线深圳杯论文展示示例  
          https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_szbsxjmtzslw/210720/1702257.shtml

        ## 使用建议
        - 对“无全文公开”的赛事，优先抓取：题目、结果、评审点评、获奖名单，并结合你自己的高分论文样本补齐全文学习库。
        - 推荐每年新增一个年份子目录，避免历史链接混乱。
        """
    )
    write_file(base / "01_官方来源索引.md", source_index)

    paper_template = dedent(
        """
        # 论文拆解模板

        ## 基础信息
        - 竞赛名称：
        - 年份与题号：
        - 论文链接：
        - 奖项层级：
        - 拆解日期：

        ## 一、问题重述（不超过 5 行）
        - 背景：
        - 核心目标：
        - 关键约束：

        ## 二、摘要结构拆解
        - 第 1 句（问题定义）：
        - 第 2 句（模型与方法）：
        - 第 3 句（关键结果）：
        - 第 4 句（结论与价值）：

        ## 三、模型组合链路
        - 数据预处理：
        - 主模型：
        - 辅助模型：
        - 稳健性/敏感性分析：
        - 误差评估：

        ## 四、数学表达质量
        - 关键符号定义是否完整：
        - 公式编号与引用是否一致：
        - 推导是否有跳步：

        ## 五、图表表达
        - 图表类型与用途映射：
        - 图题/表题规范性：
        - 视觉可读性：

        ## 六、可复用写作句式
        - 摘要句式：
        - 模型章节句式：
        - 结论章节句式：

        ## 七、可迁移到本题的要点
        - 可直接复用：
        - 需改造复用：
        - 不建议复用：
        """
    )
    write_file(base / "02_论文拆解模板.md", paper_template)

    style_checklist = dedent(
        """
        # 写作风格观察清单

        ## 摘要
        - 是否明确“问题-方法-结果-结论”四段式信息？
        - 是否出现空泛叙述而无量化结果？
        - 是否在摘要中交代模型创新点？

        ## 引言与问题分析
        - 是否把业务问题转化为可建模问题？
        - 假设是否写明边界与合理性来源？

        ## 模型与求解
        - 变量、参数、集合定义是否统一？
        - 推导是否给出关键中间式？
        - 算法流程是否可复现？

        ## 图表
        - 图轴、单位、图例是否完整？
        - 图注是否自解释（不依赖正文）？
        - 是否避免“花哨但信息密度低”的图？

        ## 结论与讨论
        - 是否有结果解释而非仅报数？
        - 是否明确模型局限与改进方向？

        ## 附录
        - 核心代码是否可运行？
        - 数据来源是否可追溯？
        """
    )
    write_file(base / "03_写作风格观察清单.md", style_checklist)

    csv_manifest = dedent(
        """
        category,competition,year_or_type,title_or_entry,url,note
        国赛优秀论文,全国大学生数学建模竞赛,总入口,论文展示主页,https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/qkt_sxjm_lw_lwzs.shtml,官方
        国赛优秀论文,全国大学生数学建模竞赛,2025,2025论文展示入口,https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/2025qgdxssxjmjslwzs/,官方
        国赛优秀论文,全国大学生数学建模竞赛,2025A,2025A题论文示例,https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2025qgdxssxjmjslwzs_2025atlw/251101/2022729.shtml,官方
        MCM/ICM Outstanding Papers,MCM/ICM,总入口,COMAP赛事主页,https://www.comap.com/contests/mcm-icm,官方
        MCM/ICM Outstanding Papers,MCM/ICM,规则,Contest Rules,https://www.contest.comap.org/undergraduate/contests/mcm/instructions.html,官方
        MCM/ICM Outstanding Papers,MCM/ICM,2025,结果总入口,https://www.contest.comap.org/undergraduate/contests/mcm/contests/2025/results/index.html,官方
        华为杯研究生数学建模优秀论文,中国研究生数学建模竞赛,总入口,竞赛主页,https://cpipc.acge.org.cn/cw/hp/4,官方
        华为杯研究生数学建模优秀论文,中国研究生数学建模竞赛,新闻,竞赛新闻列表,https://cpipc.acge.org.cn/cw/contestNews/list/4/1,官方
        亚太杯优秀论文,APMCM,总入口,APMCM官网,https://apmcm.org/,官方
        亚太杯优秀论文,APMCM,中文入口,APMCM中文站,https://www.apmcm.org/?language=cn,官方
        MathorCup优秀论文,MathorCup,总入口,MathorCup官网,https://www.mathorcup.org/,官方
        MathorCup优秀论文,MathorCup,栏目,赛题与评奖,https://mathorcup.org/cooperation,官方
        深圳杯优秀论文,深圳杯,赛事入口,m2ct赛事实页,https://www.m2ct.org/modular-list.jsp?menuType=bfInstruc&pageType=smxly,官方
        深圳杯优秀论文,深圳杯,评审,2025评审点评,https://www.cmathc.org.cn/mcm/news/394.html,官方
        深圳杯优秀论文,深圳杯,历史优秀论文,2017公开展示通知,https://www.mcm.edu.cn/upload_cn/node/431/OWbhyOK2aae9070a2ac76a31390df8661d8c8672.pdf,官方
        """
    ).strip()
    write_file(base / "04_高质量论文样本清单.csv", csv_manifest)


def method_block(
    name: str,
    scene: str,
    idea: str,
    formula: str,
    steps: list[str],
    params: list[str],
    py_code: str,
    pros: list[str],
    cons: list[str],
    writing_tpl: str,
) -> str:
    steps_md = "\n".join([f"{idx + 1}. {s}" for idx, s in enumerate(steps)])
    params_md = "\n".join([f"- {p}" for p in params])
    pros_md = "\n".join([f"- {p}" for p in pros])
    cons_md = "\n".join([f"- {c}" for c in cons])

    return dedent(
        f"""
        # {name}

        ## 适用场景
        {scene}

        ## 基本思想
        {idea}

        ## 数学公式
        {formula}

        ## 建模步骤
        {steps_md}

        ## 参数说明
        {params_md}

        ## Python 代码
        ```python
        {py_code}
        ```

        ## 优缺点
        优点：
        {pros_md}

        缺点：
        {cons_md}

        ## 论文表述模板
        {writing_tpl}
        """
    )


def build_method_library() -> None:
    base = ROOT / "KB_建模方法库"
    base.mkdir(parents=True, exist_ok=True)

    readme = dedent(
        """
        # KB_建模方法库

        每个方法文档统一包含：
        - 适用场景
        - 基本思想
        - 数学公式
        - 建模步骤
        - 参数说明
        - Python代码
        - 优缺点
        - 论文表述模板

        建议：在实际论文中采用“主模型 + 对照模型 + 稳健性分析”写法，避免单模型孤立叙述。
        """
    )
    write_file(base / "README.md", readme)

    common_tpl = "本文针对{target}问题，构建了{name}模型。首先{preprocess}，然后{core}，最后通过{verify}验证模型有效性。结果表明，该方法在{result}方面表现良好。"

    methods = [
        {
            "file": "01_AHP_层次分析法.md",
            "name": "AHP 层次分析法",
            "scene": "多指标决策、指标权重需要专家知识表达、样本数据不足时。",
            "idea": "将复杂问题分解为目标层-准则层-方案层，通过两两比较构造判断矩阵，计算权重并进行一致性检验。",
            "formula": "判断矩阵 A=(a_ij)，权重向量 w 满足 A w = λ_max w；一致性指标 CI=(λ_max-n)/(n-1)，一致性比率 CR=CI/RI。",
            "steps": ["建立层次结构", "构造判断矩阵", "求特征向量并归一化得到权重", "进行一致性检验", "综合评分排序"],
            "params": ["a_ij：第 i 指标相对第 j 指标的重要性", "RI：随机一致性指标", "CR<0.1 通常认为通过一致性检验"],
            "code": "import numpy as np\n\nA = np.array([[1,3,5],[1/3,1,2],[1/5,1/2,1]], dtype=float)\nvals, vecs = np.linalg.eig(A)\nidx = np.argmax(vals.real)\nw = np.abs(vecs[:, idx].real)\nw = w / w.sum()\nlambda_max = vals[idx].real\nn = A.shape[0]\nCI = (lambda_max - n) / (n - 1)\nRI = 0.58\nCR = CI / RI\nprint('weights=', w, 'CR=', CR)",
            "pros": ["结构清晰，便于解释", "融合专家经验", "可处理定性指标"],
            "cons": ["主观性较强", "指标多时判断矩阵构造成本高"],
            "tpl": common_tpl.format(target="综合评价", name="AHP", preprocess="对指标进行层次划分", core="通过判断矩阵求得各层权重", verify="一致性检验", result="决策排序与解释性"),
        },
        {
            "file": "02_熵权法.md",
            "name": "熵权法",
            "scene": "多指标评价中需要客观赋权、指标离散程度差异明显时。",
            "idea": "利用信息熵衡量指标信息量，离散程度越大，权重越高。",
            "formula": "标准化后 p_ij = x_ij / Σ_i x_ij；熵值 e_j = -k Σ_i p_ij ln p_ij；差异系数 d_j=1-e_j；权重 w_j=d_j/Σ_j d_j。",
            "steps": ["指标正向化与标准化", "计算指标熵值", "计算差异系数与权重", "加权汇总评分"],
            "params": ["k=1/ln(n)", "p_ij 为第 i 样本在第 j 指标的占比", "n 为样本数"],
            "code": "import numpy as np\n\nX = np.array([[8, 70, 0.8],[6, 90, 0.6],[9, 60, 0.9]], dtype=float)\nX = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)\nP = X / (X.sum(axis=0, keepdims=True) + 1e-12)\nk = 1 / np.log(X.shape[0])\nE = -k * np.sum(P * np.log(P + 1e-12), axis=0)\nD = 1 - E\nW = D / D.sum()\nscore = X @ W\nprint('weights=', W, 'score=', score)",
            "pros": ["客观性强", "实现简单", "适合批量评价"],
            "cons": ["对异常值敏感", "忽略指标因果结构"],
            "tpl": common_tpl.format(target="多指标评价", name="熵权法", preprocess="完成指标无量纲化", core="按信息熵计算客观权重", verify="与主观权重进行对比", result="区分样本优劣"),
        },
        {
            "file": "03_TOPSIS.md",
            "name": "TOPSIS",
            "scene": "方案排序、绩效评价、风险评估等需要比较“最优解接近度”的场景。",
            "idea": "计算各方案与正理想解、负理想解的距离，选择最接近正理想、最远离负理想的方案。",
            "formula": "D_i^+=sqrt(Σ_j (v_ij-v_j^+)^2), D_i^-=sqrt(Σ_j (v_ij-v_j^-)^2), C_i=D_i^-/(D_i^+ + D_i^-)。",
            "steps": ["指标标准化", "乘以权重得到加权矩阵", "确定正负理想解", "计算距离与贴近度", "按贴近度排序"],
            "params": ["v_j^+：正理想解", "v_j^-：负理想解", "C_i：贴近度，越大越优"],
            "code": "import numpy as np\n\nX = np.array([[80, 0.90, 7],[75, 0.95, 8],[90, 0.85, 6]], dtype=float)\nW = np.array([0.4, 0.35, 0.25])\nXn = X / np.sqrt((X**2).sum(axis=0))\nV = Xn * W\nV_pos = np.array([V[:,0].max(), V[:,1].max(), V[:,2].min()])\nV_neg = np.array([V[:,0].min(), V[:,1].min(), V[:,2].max()])\nD_pos = np.sqrt(((V - V_pos)**2).sum(axis=1))\nD_neg = np.sqrt(((V - V_neg)**2).sum(axis=1))\nC = D_neg / (D_pos + D_neg)\nprint('C=', C)",
            "pros": ["逻辑直观", "可与多种赋权方法结合", "适合排序任务"],
            "cons": ["权重依赖明显", "指标相关性强时可能失真"],
            "tpl": common_tpl.format(target="方案优选", name="TOPSIS", preprocess="进行标准化并确定权重", core="计算与正负理想解距离", verify="采用不同权重做鲁棒性对比", result="区分优劣方案"),
        },
        {
            "file": "04_灰色关联分析.md",
            "name": "灰色关联分析",
            "scene": "小样本、信息不完备条件下的因素关联度分析。",
            "idea": "通过序列几何形状相似度度量变量间关联程度，适合样本少、噪声大的情形。",
            "formula": "关联系数 ξ_i(k)=(Δ_min+ρΔ_max)/(Δ_i(k)+ρΔ_max)，关联度 r_i=平均_k ξ_i(k)。",
            "steps": ["确定参考序列与比较序列", "无量纲化处理", "计算绝对差序列", "计算关联系数与关联度", "按关联度排序"],
            "params": ["ρ：分辨系数，常取 0.5", "Δ_i(k)：对应点绝对差"],
            "code": "import numpy as np\n\nx0 = np.array([10,12,15,14,18], dtype=float)\nX = np.array([[9,11,14,13,16],[11,13,16,15,19]], dtype=float)\nx0 = (x0 - x0.min()) / (x0.max() - x0.min())\nXn = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True)-X.min(axis=1, keepdims=True)+1e-12)\nDelta = np.abs(Xn - x0)\nd_min, d_max = Delta.min(), Delta.max()\nrho = 0.5\nxi = (d_min + rho*d_max) / (Delta + rho*d_max)\nr = xi.mean(axis=1)\nprint('relation=', r)",
            "pros": ["适合小样本", "对分布要求低", "解释性较好"],
            "cons": ["无量纲化方式影响结果", "仅反映相关不代表因果"],
            "tpl": common_tpl.format(target="因素影响", name="灰色关联分析", preprocess="构建参考序列和比较序列", core="计算灰色关联系数与关联度", verify="更换分辨系数做稳健性分析", result="识别关键影响因子"),
        },
        {
            "file": "05_GM11.md",
            "name": "GM(1,1)",
            "scene": "短期预测、小样本时间序列、趋势较单调的场景。",
            "idea": "通过一次累加生成（AGO）削弱随机波动，建立一阶微分方程进行预测。",
            "formula": "x^(1)(k+1)= (x^(0)(1)-b/a)e^{-ak} + b/a，x^(0)(k)=x^(1)(k)-x^(1)(k-1)。",
            "steps": ["原序列 AGO", "构造背景值与矩阵 B", "最小二乘估计参数 a,b", "还原预测序列", "残差检验"],
            "params": ["a：发展系数", "b：灰作用量", "后验差比 C、 小误差概率 P 用于检验"],
            "code": "import numpy as np\n\nx0 = np.array([32, 35, 39, 44, 50], dtype=float)\nx1 = np.cumsum(x0)\nz1 = 0.5 * (x1[1:] + x1[:-1])\nB = np.column_stack((-z1, np.ones_like(z1)))\nY = x0[1:]\na, b = np.linalg.lstsq(B, Y, rcond=None)[0]\n\ndef x1_hat(k):\n    return (x0[0] - b/a) * np.exp(-a*(k-1)) + b/a\n\npred = [x1_hat(k) - x1_hat(k-1) for k in range(2, 8)]\nprint('a,b=', a, b)\nprint('pred=', pred)",
            "pros": ["样本需求低", "实现简单", "对短期趋势有效"],
            "cons": ["对突变数据不稳", "不适合复杂季节性序列"],
            "tpl": common_tpl.format(target="短期趋势预测", name="GM(1,1)", preprocess="对原序列进行AGO累加", core="估计灰微分方程参数", verify="通过残差与后验差比检验", result="短期预测精度"),
        },
        {
            "file": "06_PCA_主成分分析.md",
            "name": "PCA 主成分分析",
            "scene": "高维降维、指标压缩、消除多重共线性。",
            "idea": "将原变量线性变换为相互正交的主成分，保留大部分方差。",
            "formula": "协方差矩阵 S 的特征分解 S=QΛQ^T，贡献率 η_i=λ_i/Σλ_i。",
            "steps": ["数据标准化", "求协方差矩阵", "特征分解", "按累计贡献率选主成分", "计算主成分得分"],
            "params": ["累计贡献率常用阈值 80%-95%", "主成分载荷反映变量贡献"],
            "code": "import numpy as np\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nX = np.random.rand(100, 6)\nXz = StandardScaler().fit_transform(X)\npca = PCA(n_components=0.9)\nZ = pca.fit_transform(Xz)\nprint('n_components=', pca.n_components_)\nprint('explained=', pca.explained_variance_ratio_)\nprint('scores_shape=', Z.shape)",
            "pros": ["降维有效", "减少噪声", "利于后续建模"],
            "cons": ["可解释性随降维降低", "线性假设限制明显"],
            "tpl": common_tpl.format(target="高维指标降维", name="PCA", preprocess="完成标准化处理", core="提取累计贡献率较高的主成分", verify="比较降维前后模型性能", result="降维与稳健性"),
        },
        {
            "file": "07_聚类分析.md",
            "name": "聚类分析",
            "scene": "无监督分组、人群细分、区域分型。",
            "idea": "依据样本相似性将对象划分到若干簇，使簇内相似度高、簇间差异大。",
            "formula": "K-means目标函数：min Σ_i Σ_{x in C_i} ||x-μ_i||^2。",
            "steps": ["特征选择与标准化", "确定聚类数 k", "训练聚类模型", "评估聚类效果", "解释各簇特征"],
            "params": ["k：聚类数量", "轮廓系数用于评估聚类质量"],
            "code": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nfrom sklearn.preprocessing import StandardScaler\n\nX = np.random.rand(120, 4)\nXz = StandardScaler().fit_transform(X)\nmodel = KMeans(n_clusters=3, n_init=20, random_state=42)\nlabels = model.fit_predict(Xz)\nscore = silhouette_score(Xz, labels)\nprint('silhouette=', score)",
            "pros": ["可发现潜在结构", "适合探索性分析"],
            "cons": ["对尺度敏感", "聚类数选择依赖经验"],
            "tpl": common_tpl.format(target="样本分群", name="聚类分析", preprocess="构建特征并标准化", core="通过无监督聚类得到类别结构", verify="使用轮廓系数等指标评估", result="类别区分度"),
        },
        {
            "file": "08_回归分析.md",
            "name": "回归分析",
            "scene": "解释变量影响、预测连续变量、建立定量关系。",
            "idea": "建立自变量与因变量的函数关系，估计参数并进行显著性分析。",
            "formula": "线性回归：y=β_0 + Σ_j β_j x_j + ε；最小二乘解 β=(X^TX)^(-1)X^Ty。",
            "steps": ["数据清洗与特征工程", "划分训练/测试集", "拟合回归模型", "评估与诊断", "解释系数并预测"],
            "params": ["R^2、RMSE、MAE 为常用指标", "需关注多重共线性与异方差"],
            "code": "import numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\n\nX = np.random.rand(200, 5)\ny = X @ np.array([2.1, -1.3, 0.8, 0.0, 1.5]) + np.random.randn(200) * 0.2\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = LinearRegression().fit(X_tr, y_tr)\npred = model.predict(X_te)\nprint('R2=', r2_score(y_te, pred), 'RMSE=', mean_squared_error(y_te, pred, squared=False))",
            "pros": ["解释性强", "成熟可靠"],
            "cons": ["对异常值敏感", "线性模型对非线性关系刻画有限"],
            "tpl": common_tpl.format(target="连续变量预测", name="回归分析", preprocess="完成特征工程与数据划分", core="估计回归参数并进行显著性分析", verify="通过残差与交叉验证检验", result="预测精度与解释性"),
        },
        {
            "file": "09_时间序列预测.md",
            "name": "时间序列预测",
            "scene": "按时间顺序观测的数据预测，如销量、负荷、客流、价格。",
            "idea": "挖掘序列中的趋势、季节性与周期性结构，利用历史信息预测未来。",
            "formula": "加法分解：y_t = T_t + S_t + R_t；乘法分解：y_t = T_t * S_t * R_t。",
            "steps": ["时间索引与频率校验", "平稳性与季节性识别", "构建模型", "滚动预测与回测", "误差评估"],
            "params": ["常用误差：MAPE、RMSE", "需关注结构突变点"],
            "code": "import pandas as pd\nfrom statsmodels.tsa.seasonal import seasonal_decompose\n\nidx = pd.date_range('2022-01-01', periods=36, freq='M')\nseries = pd.Series(range(36), index=idx) + pd.Series([0,1,2]*12, index=idx)\nres = seasonal_decompose(series, model='additive', period=12)\nprint(res.trend.dropna().head())",
            "pros": ["符合时间逻辑", "可解释趋势与季节成分"],
            "cons": ["对异常冲击敏感", "长期预测误差累积"],
            "tpl": common_tpl.format(target="时序变化", name="时间序列预测", preprocess="识别趋势与季节性", core="构建时序模型并滚动预测", verify="进行回测误差分析", result="中短期预测"),
        },
        {
            "file": "10_ARIMA.md",
            "name": "ARIMA",
            "scene": "单变量时间序列、无复杂外生变量时的经典预测场景。",
            "idea": "通过差分使序列平稳，结合自回归与移动平均项建模。",
            "formula": "ARIMA(p,d,q)：Φ(B)(1-B)^d y_t = Θ(B)ε_t。",
            "steps": ["平稳性检验与差分", "识别 p,q（ACF/PACF）", "拟合ARIMA", "残差白噪声检验", "预测与区间估计"],
            "params": ["p：AR阶数", "d：差分阶数", "q：MA阶数"],
            "code": "import numpy as np\nimport pandas as pd\nfrom statsmodels.tsa.arima.model import ARIMA\n\nidx = pd.date_range('2020-01-01', periods=80, freq='M')\ny = pd.Series(np.cumsum(np.random.randn(80)) + 50, index=idx)\nmodel = ARIMA(y, order=(1,1,1)).fit()\nforecast = model.forecast(steps=6)\nprint(forecast)",
            "pros": ["理论成熟", "样本需求相对可控"],
            "cons": ["参数选择依赖经验", "难处理强非线性"],
            "tpl": common_tpl.format(target="单变量时序", name="ARIMA", preprocess="完成平稳化与阶数识别", core="拟合ARIMA并检验残差", verify="滚动窗口回测", result="短期预测"),
        },
        {
            "file": "11_随机森林.md",
            "name": "随机森林",
            "scene": "中小样本分类/回归，非线性关系明显，需要较强泛化能力。",
            "idea": "通过Bootstrap采样构建多棵决策树，采用Bagging集成降低方差。",
            "formula": "集成预测：\\hat{f}(x)=1/B Σ_{b=1}^B f_b(x)。",
            "steps": ["数据预处理", "训练随机森林", "调参与交叉验证", "特征重要性分析", "测试集评估"],
            "params": ["n_estimators：树数量", "max_depth：树深度", "max_features：分裂特征数"],
            "code": "import numpy as np\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error\n\nX = np.random.rand(300, 8)\ny = np.sin(X[:,0]*3) + X[:,1]**2 + np.random.randn(300)*0.05\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\nrf = RandomForestRegressor(n_estimators=300, random_state=42)\nrf.fit(X_tr, y_tr)\npred = rf.predict(X_te)\nprint('RMSE=', mean_squared_error(y_te, pred, squared=False))",
            "pros": ["鲁棒性好", "可处理非线性与交互作用", "对异常值不太敏感"],
            "cons": ["可解释性弱于线性模型", "模型体积较大"],
            "tpl": common_tpl.format(target="非线性回归/分类", name="随机森林", preprocess="完成数据清洗与特征编码", core="训练多树集成模型", verify="通过交叉验证与特征重要性分析", result="预测精度与稳定性"),
        },
        {
            "file": "12_XGBoost.md",
            "name": "XGBoost",
            "scene": "结构化数据预测、对精度要求高、需要处理复杂非线性与稀疏特征。",
            "idea": "基于梯度提升框架逐步拟合残差，并加入正则化抑制过拟合。",
            "formula": "目标函数 Obj = Σ_i l(y_i, ŷ_i) + Σ_k Ω(f_k)，其中 Ω 控制树复杂度。",
            "steps": ["准备训练数据", "设定损失函数与参数", "迭代训练弱学习器", "早停与调参", "解释与评估"],
            "params": ["eta：学习率", "max_depth：树深度", "subsample：样本采样率", "colsample_bytree：特征采样率"],
            "code": "import numpy as np\nimport xgboost as xgb\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error\n\nX = np.random.rand(400, 10)\ny = 3*X[:,0] - 2*X[:,1] + np.sin(4*X[:,2]) + np.random.randn(400)*0.1\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)\nmodel.fit(X_tr, y_tr)\npred = model.predict(X_te)\nprint('RMSE=', mean_squared_error(y_te, pred, squared=False))",
            "pros": ["精度高", "抗过拟合能力较强", "支持缺失值处理"],
            "cons": ["参数较多", "训练与解释成本较高"],
            "tpl": common_tpl.format(target="高精度预测", name="XGBoost", preprocess="完成特征构建和样本划分", core="基于梯度提升迭代学习残差", verify="通过早停和交叉验证抑制过拟合", result="泛化误差"),
        },
        {
            "file": "13_LSTM.md",
            "name": "LSTM",
            "scene": "长序列依赖明显的时序预测、序列建模任务。",
            "idea": "通过输入门、遗忘门、输出门控制信息流，缓解RNN梯度消失问题。",
            "formula": "f_t=σ(W_f[h_{t-1},x_t]+b_f)，i_t=σ(...)，\\tilde{C_t}=tanh(...)，C_t=f_t*C_{t-1}+i_t*\\tilde{C_t}。",
            "steps": ["构造滑动窗口样本", "数据归一化", "搭建LSTM网络", "训练与验证", "反归一化输出预测"],
            "params": ["look_back：窗口长度", "hidden_size：隐藏层维度", "epochs：训练轮次"],
            "code": "import numpy as np\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense\n\nX = np.random.rand(200, 12, 3)\ny = np.random.rand(200)\nmodel = Sequential([\n    LSTM(32, input_shape=(12, 3)),\n    Dense(1)\n])\nmodel.compile(optimizer='adam', loss='mse')\nmodel.fit(X, y, epochs=5, batch_size=16, verbose=0)\nprint('trained')",
            "pros": ["能刻画长依赖", "适合复杂时序模式"],
            "cons": ["训练成本较高", "参数敏感，调参复杂"],
            "tpl": common_tpl.format(target="复杂时序预测", name="LSTM", preprocess="构建滑动窗口并归一化", core="训练门控循环网络提取时序特征", verify="用滚动窗口评估外推能力", result="非线性时序拟合"),
        },
        {
            "file": "14_线性规划.md",
            "name": "线性规划",
            "scene": "资源分配、生产计划、运输问题等目标与约束均线性的优化任务。",
            "idea": "在一组线性约束条件下，求线性目标函数最优值。",
            "formula": "min c^Tx, s.t. Ax<=b, x>=0。",
            "steps": ["定义决策变量", "建立目标函数", "建立约束条件", "调用求解器", "结果敏感性分析"],
            "params": ["c：目标系数", "A,b：约束矩阵与向量"],
            "code": "import numpy as np\nfrom scipy.optimize import linprog\n\nc = [-3, -5]\nA = [[1, 0], [0, 2], [3, 2]]\nb = [4, 12, 18]\nres = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')\nprint('x=', res.x, 'obj=', -res.fun)",
            "pros": ["理论成熟", "求解高效", "解释性强"],
            "cons": ["线性假设限制较强"],
            "tpl": common_tpl.format(target="资源优化配置", name="线性规划", preprocess="完成变量与约束定义", core="建立线性目标并调用求解器", verify="进行约束扰动敏感性分析", result="资源利用效率"),
        },
        {
            "file": "15_整数规划.md",
            "name": "整数规划",
            "scene": "选址、排班、装箱、路径等决策变量必须取整数的场景。",
            "idea": "在线性（或非线性）优化框架下加入整数约束，求可行最优解。",
            "formula": "min c^Tx, s.t. Ax<=b, x∈Z^n。",
            "steps": ["定义整数/0-1变量", "建立目标与约束", "调用MILP求解器", "验证可行性", "分析业务解释"],
            "params": ["binary/integer 变量类型", "Big-M 参数需谨慎设置"],
            "code": "import pulp\n\nmodel = pulp.LpProblem('ip', pulp.LpMaximize)\nx1 = pulp.LpVariable('x1', lowBound=0, cat='Integer')\nx2 = pulp.LpVariable('x2', lowBound=0, cat='Integer')\nmodel += 3*x1 + 5*x2\nmodel += x1 <= 4\nmodel += 2*x2 <= 12\nmodel += 3*x1 + 2*x2 <= 18\nmodel.solve(pulp.PULP_CBC_CMD(msg=False))\nprint('x1,x2=', x1.value(), x2.value(), 'obj=', pulp.value(model.objective))",
            "pros": ["贴合离散决策", "可表达复杂业务规则"],
            "cons": ["计算复杂度高", "大规模问题求解时间长"],
            "tpl": common_tpl.format(target="离散决策优化", name="整数规划", preprocess="将业务规则离散化建模", core="引入整数约束求最优", verify="对约束和参数进行稳健性分析", result="可执行决策方案"),
        },
        {
            "file": "16_非线性规划.md",
            "name": "非线性规划",
            "scene": "目标/约束为非线性，如收益饱和、能耗非线性、复杂工程设计。",
            "idea": "在可行域中搜索非线性目标函数最优解。",
            "formula": "min f(x), s.t. g_i(x)<=0, h_j(x)=0。",
            "steps": ["确定变量与可行域", "构建非线性目标与约束", "选择求解器", "多初值求解", "结果验证"],
            "params": ["初值选择影响收敛", "局部最优风险需控制"],
            "code": "import numpy as np\nfrom scipy.optimize import minimize\n\ndef obj(x):\n    return (x[0]-1)**2 + (x[1]-2)**2 + np.sin(x[0]*x[1])\n\ncons = ({'type': 'ineq', 'fun': lambda x: 3 - x[0] - x[1]},)\nres = minimize(obj, x0=np.array([0.5, 0.5]), constraints=cons)\nprint('x=', res.x, 'f=', res.fun)",
            "pros": ["表达能力强", "适配真实复杂问题"],
            "cons": ["可能陷入局部最优", "对初值与求解器敏感"],
            "tpl": common_tpl.format(target="复杂优化", name="非线性规划", preprocess="完成变量和约束非线性表达", core="采用数值优化算法求解", verify="多初值对比与可行性检查", result="目标改进幅度"),
        },
        {
            "file": "17_多目标优化.md",
            "name": "多目标优化",
            "scene": "成本-效益、效率-公平、风险-收益等多目标冲突问题。",
            "idea": "同时优化多个目标函数，求帕累托最优解集并做决策折中。",
            "formula": "min (f_1(x), f_2(x),...,f_m(x))，x∈Ω。",
            "steps": ["定义多目标函数", "构建约束", "求帕累托前沿", "偏好决策", "方案解释"],
            "params": ["加权法权重", "ε-约束阈值", "帕累托支配关系"],
            "code": "import numpy as np\n\n# 简化：网格搜索近似帕累托前沿\nxs = np.linspace(0, 5, 200)\npts = []\nfor x in xs:\n    f1 = x**2\n    f2 = (x-4)**2\n    pts.append((x, f1, f2))\n\npareto = []\nfor p in pts:\n    dominated = False\n    for q in pts:\n        if q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]):\n            dominated = True\n            break\n    if not dominated:\n        pareto.append(p)\nprint('pareto_size=', len(pareto))",
            "pros": ["适合权衡冲突目标", "输出决策空间更全面"],
            "cons": ["需要后续偏好决策", "计算复杂度较高"],
            "tpl": common_tpl.format(target="多目标权衡", name="多目标优化", preprocess="建立多个目标函数", core="求取帕累托最优解集", verify="比较不同偏好下的方案稳定性", result="折中最优方案"),
        },
        {
            "file": "18_遗传算法.md",
            "name": "遗传算法",
            "scene": "复杂非凸、组合优化、不可导目标函数求解。",
            "idea": "模拟自然选择，通过选择、交叉、变异迭代搜索最优解。",
            "formula": "适应度函数 Fitness(x) 引导群体进化。",
            "steps": ["编码与初始化种群", "计算适应度", "选择交叉变异", "迭代进化", "输出最优个体"],
            "params": ["pop_size：种群规模", "pc：交叉率", "pm：变异率", "gen：迭代代数"],
            "code": "import random\n\ndef fitness(x):\n    return -(x-3.7)**2 + 10\n\npop = [random.uniform(0, 10) for _ in range(30)]\nfor _ in range(80):\n    pop = sorted(pop, key=fitness, reverse=True)\n    elite = pop[:10]\n    new_pop = elite[:]\n    while len(new_pop) < 30:\n        a, b = random.sample(elite, 2)\n        child = (a + b) / 2\n        if random.random() < 0.2:\n            child += random.uniform(-0.5, 0.5)\n        child = min(10, max(0, child))\n        new_pop.append(child)\n    pop = new_pop\nbest = max(pop, key=fitness)\nprint('best=', best, 'fitness=', fitness(best))",
            "pros": ["全局搜索能力较强", "适合黑箱优化"],
            "cons": ["参数敏感", "收敛速度不稳定"],
            "tpl": common_tpl.format(target="复杂组合优化", name="遗传算法", preprocess="完成编码与初始种群生成", core="通过选择交叉变异迭代搜索", verify="多次独立运行比较稳定性", result="全局近优解"),
        },
        {
            "file": "19_模拟退火.md",
            "name": "模拟退火",
            "scene": "离散优化、路径规划、排程等局部最优陷阱明显的问题。",
            "idea": "在高温时接受劣解以跳出局部最优，降温后逐步收敛。",
            "formula": "接受概率 P=exp(-ΔE/T)（当 ΔE>0 时）。",
            "steps": ["初始化解与温度", "邻域扰动生成新解", "按概率接受", "降温迭代", "终止输出最优"],
            "params": ["T0：初始温度", "alpha：降温系数", "L：每温度迭代次数"],
            "code": "import math\nimport random\n\ndef f(x):\n    return (x-2.4)**2 + math.sin(5*x)\n\nx = random.uniform(-5, 5)\nbest = x\nT = 10.0\nwhile T > 1e-3:\n    for _ in range(50):\n        xn = x + random.uniform(-0.3, 0.3)\n        de = f(xn) - f(x)\n        if de < 0 or random.random() < math.exp(-de / T):\n            x = xn\n            if f(x) < f(best):\n                best = x\n    T *= 0.92\nprint('best=', best, 'f=', f(best))",
            "pros": ["实现简单", "可跳出局部最优"],
            "cons": ["参数设置依赖经验", "计算耗时可能较长"],
            "tpl": common_tpl.format(target="全局优化", name="模拟退火", preprocess="设置初始解与温度", core="依据Metropolis准则迭代搜索", verify="重复实验检验稳定性", result="全局近优性能"),
        },
        {
            "file": "20_粒子群算法.md",
            "name": "粒子群算法",
            "scene": "连续优化、参数寻优、神经网络超参搜索。",
            "idea": "粒子在解空间内根据个体最优与群体最优更新速度和位置。",
            "formula": "v_i^{t+1}=ωv_i^t+c1r1(pbest_i-x_i^t)+c2r2(gbest-x_i^t)，x_i^{t+1}=x_i^t+v_i^{t+1}。",
            "steps": ["初始化粒子位置速度", "计算适应度", "更新pbest/gbest", "更新速度位置", "迭代终止"],
            "params": ["ω：惯性权重", "c1,c2：学习因子", "粒子数与迭代数"],
            "code": "import numpy as np\n\nnp.random.seed(42)\nN = 30\nx = np.random.uniform(-5, 5, size=N)\nv = np.zeros(N)\npbest = x.copy()\n\ndef f(z):\n    return (z-1.8)**2 + np.sin(3*z)\n\ngbest = pbest[np.argmin(f(pbest))]\nfor _ in range(100):\n    r1, r2 = np.random.rand(N), np.random.rand(N)\n    v = 0.7*v + 1.4*r1*(pbest - x) + 1.4*r2*(gbest - x)\n    x = x + v\n    better = f(x) < f(pbest)\n    pbest[better] = x[better]\n    gbest = pbest[np.argmin(f(pbest))]\nprint('gbest=', gbest, 'f=', f(gbest))",
            "pros": ["参数相对较少", "收敛速度较快"],
            "cons": ["后期易早熟", "高维复杂问题性能波动"],
            "tpl": common_tpl.format(target="连续参数优化", name="粒子群算法", preprocess="初始化粒子群", core="按群体协同机制更新粒子", verify="多次运行比较最优值分布", result="求解效率与精度"),
        },
        {
            "file": "21_蒙特卡洛模拟.md",
            "name": "蒙特卡洛模拟",
            "scene": "不确定性传播、风险评估、积分估计、复杂系统仿真。",
            "idea": "通过大量随机采样近似求解解析难解问题。",
            "formula": "E[f(X)] ≈ (1/N)Σ_{i=1}^N f(x_i)。",
            "steps": ["定义随机变量分布", "随机采样", "计算目标函数", "统计结果分布", "置信区间估计"],
            "params": ["N：采样次数", "随机种子用于复现"],
            "code": "import numpy as np\n\nnp.random.seed(42)\nN = 100000\nx = np.random.rand(N)\ny = np.random.rand(N)\ninside = (x**2 + y**2) <= 1\npi_est = 4 * inside.mean()\nprint('pi_est=', pi_est)",
            "pros": ["通用性强", "适合复杂不确定系统"],
            "cons": ["采样量大时代价高", "收敛速度相对较慢"],
            "tpl": common_tpl.format(target="不确定性分析", name="蒙特卡洛模拟", preprocess="建立随机输入分布", core="进行大规模随机仿真", verify="比较采样规模对结果稳定性的影响", result="风险估计"),
        },
        {
            "file": "22_灵敏度分析.md",
            "name": "灵敏度分析",
            "scene": "评估模型对参数扰动的响应，识别关键参数。",
            "idea": "改变输入参数并观察输出变化，衡量参数对模型结果的影响强度。",
            "formula": "局部灵敏度 S_i = (ΔY/Y) / (ΔX_i/X_i)。",
            "steps": ["确定基准参数", "设定扰动范围", "逐一或联合扰动", "计算输出变化", "排序关键参数"],
            "params": ["扰动幅度常取 ±5%、±10%", "可采用OAT或全局灵敏度方法"],
            "code": "import numpy as np\n\ndef model(x1, x2):\n    return 2*x1**2 + 0.5*x2\n\nx1, x2 = 10.0, 8.0\ny0 = model(x1, x2)\nfor rate in [0.05, 0.1]:\n    y1 = model(x1*(1+rate), x2)\n    s1 = ((y1 - y0) / y0) / rate\n    y2 = model(x1, x2*(1+rate))\n    s2 = ((y2 - y0) / y0) / rate\n    print('rate=', rate, 'S_x1=', s1, 'S_x2=', s2)",
            "pros": ["帮助识别关键参数", "提升模型解释性"],
            "cons": ["局部法受基准点影响", "全局法计算量大"],
            "tpl": common_tpl.format(target="参数稳健性", name="灵敏度分析", preprocess="确定参数基准与扰动方案", core="计算输出对输入扰动的响应", verify="对比不同扰动幅度结果", result="关键参数识别"),
        },
        {
            "file": "23_误差分析.md",
            "name": "误差分析",
            "scene": "预测模型评估、测量误差分解、模型比较。",
            "idea": "通过多种误差指标与残差图诊断模型偏差、方差与异常点。",
            "formula": "MAE=(1/n)Σ|y_i-ŷ_i|，RMSE=sqrt((1/n)Σ(y_i-ŷ_i)^2)，MAPE=(1/n)Σ|((y_i-ŷ_i)/y_i)|。",
            "steps": ["计算误差指标", "绘制残差分布与残差-拟合图", "识别异常点", "比较模型误差", "给出改进建议"],
            "params": ["MAE 对异常值不太敏感", "RMSE 对大误差更敏感", "MAPE 在 y 接近0时不稳定"],
            "code": "import numpy as np\n\ny = np.array([10, 20, 30, 25, 40], dtype=float)\nyhat = np.array([12, 18, 29, 27, 41], dtype=float)\nmae = np.mean(np.abs(y - yhat))\nrmse = np.sqrt(np.mean((y - yhat)**2))\nmape = np.mean(np.abs((y - yhat) / (y + 1e-12)))\nprint('MAE=', mae, 'RMSE=', rmse, 'MAPE=', mape)",
            "pros": ["可量化模型效果", "支持模型对比与诊断"],
            "cons": ["单一指标可能误导", "需结合业务阈值解释"],
            "tpl": common_tpl.format(target="模型效果评估", name="误差分析", preprocess="构建真实值-预测值对照", core="计算多维误差指标并做残差诊断", verify="在不同样本子集比较误差稳定性", result="误差可控性"),
        },
    ]

    for m in methods:
        content = method_block(
            name=m["name"],
            scene=m["scene"],
            idea=m["idea"],
            formula=m["formula"],
            steps=m["steps"],
            params=m["params"],
            py_code=m["code"],
            pros=m["pros"],
            cons=m["cons"],
            writing_tpl=m["tpl"],
        )
        write_file(base / m["file"], content)


def build_code_library() -> None:
    base = ROOT / "KB_代码模板库"
    base.mkdir(parents=True, exist_ok=True)

    readme = dedent(
        """
        # KB_代码模板库

        用途：
        - 给 `coding_solver` 快速调用算法模板
        - 给 `chart_maker` 快速生成图表数据与结果

        注意：
        1. 模板优先保证可读与可改，非极限性能写法。
        2. 每个模板都保留函数入口，便于你在智能体中按需拼接。
        """
    )
    write_file(base / "README.md", readme)

    templates = {
        "01_数据读取模板.py": dedent(
            """
            import pandas as pd


            def load_data(path: str, file_type: str = "csv") -> pd.DataFrame:
                if file_type == "csv":
                    return pd.read_csv(path)
                if file_type in {"xlsx", "xls"}:
                    return pd.read_excel(path)
                if file_type == "tsv":
                    return pd.read_csv(path, sep="\\t")
                raise ValueError(f"unsupported file_type: {file_type}")


            if __name__ == "__main__":
                df = load_data("data.csv", "csv")
                print(df.head())
            """
        ),
        "02_数据清洗模板.py": dedent(
            """
            import pandas as pd


            def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out.columns = [c.strip() for c in out.columns]
                out = out.drop_duplicates()
                out = out.replace({"": None, "NA": None, "N/A": None})
                return out


            if __name__ == "__main__":
                demo = pd.DataFrame({" a ": [1, 1, None], "b": ["NA", "x", "x"]})
                print(basic_clean(demo))
            """
        ),
        "03_缺失值处理模板.py": dedent(
            """
            import pandas as pd
            from sklearn.impute import SimpleImputer


            def fill_missing(df: pd.DataFrame, strategy_num: str = "median", strategy_cat: str = "most_frequent") -> pd.DataFrame:
                out = df.copy()
                num_cols = out.select_dtypes(include=["number"]).columns
                cat_cols = [c for c in out.columns if c not in num_cols]

                if len(num_cols) > 0:
                    imp_num = SimpleImputer(strategy=strategy_num)
                    out[num_cols] = imp_num.fit_transform(out[num_cols])
                if len(cat_cols) > 0:
                    imp_cat = SimpleImputer(strategy=strategy_cat)
                    out[cat_cols] = imp_cat.fit_transform(out[cat_cols])
                return out
            """
        ),
        "04_异常值处理模板.py": dedent(
            """
            import pandas as pd


            def clip_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
                out = df.copy()
                num_cols = out.select_dtypes(include=["number"]).columns
                for col in num_cols:
                    q1 = out[col].quantile(0.25)
                    q3 = out[col].quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - factor * iqr
                    high = q3 + factor * iqr
                    out[col] = out[col].clip(lower=low, upper=high)
                return out
            """
        ),
        "05_可视化模板.py": dedent(
            """
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
            plt.rcParams["axes.unicode_minus"] = False


            def quick_plot(df: pd.DataFrame, x: str, y: str) -> None:
                sns.set_theme(style="whitegrid")
                plt.figure(figsize=(8, 5))
                sns.lineplot(data=df, x=x, y=y, marker="o")
                plt.title("趋势图")
                plt.tight_layout()
                plt.show()
            """
        ),
        "06_AHP_代码.py": dedent(
            """
            import numpy as np


            def ahp_weights(matrix: np.ndarray, ri: float = 0.58):
                vals, vecs = np.linalg.eig(matrix)
                idx = np.argmax(vals.real)
                w = np.abs(vecs[:, idx].real)
                w = w / w.sum()
                n = matrix.shape[0]
                ci = (vals[idx].real - n) / (n - 1)
                cr = ci / ri if ri > 0 else 0
                return w, cr
            """
        ),
        "07_熵权法代码.py": dedent(
            """
            import numpy as np


            def entropy_weight(X: np.ndarray):
                Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
                P = Xn / (Xn.sum(axis=0, keepdims=True) + 1e-12)
                k = 1 / np.log(X.shape[0])
                E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
                D = 1 - E
                W = D / D.sum()
                score = Xn @ W
                return W, score
            """
        ),
        "08_TOPSIS_代码.py": dedent(
            """
            import numpy as np


            def topsis(X: np.ndarray, W: np.ndarray, benefit_mask: np.ndarray):
                X = X.astype(float)
                Xn = X / np.sqrt((X ** 2).sum(axis=0))
                V = Xn * W

                V_pos = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
                V_neg = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))

                D_pos = np.sqrt(((V - V_pos) ** 2).sum(axis=1))
                D_neg = np.sqrt(((V - V_neg) ** 2).sum(axis=1))
                C = D_neg / (D_pos + D_neg + 1e-12)
                return C
            """
        ),
        "09_PCA_代码.py": dedent(
            """
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA


            def run_pca(X, variance_ratio: float = 0.9):
                Xz = StandardScaler().fit_transform(X)
                pca = PCA(n_components=variance_ratio)
                Z = pca.fit_transform(Xz)
                return Z, pca
            """
        ),
        "10_聚类代码.py": dedent(
            """
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score


            def run_kmeans(X, k: int = 3):
                Xz = StandardScaler().fit_transform(X)
                model = KMeans(n_clusters=k, n_init=20, random_state=42)
                labels = model.fit_predict(Xz)
                score = silhouette_score(Xz, labels)
                return labels, score, model
            """
        ),
        "11_回归预测代码.py": dedent(
            """
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


            def train_regressor(X, y):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                metrics = {
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": mean_squared_error(y_test, pred, squared=False),
                    "R2": r2_score(y_test, pred),
                }
                return model, metrics
            """
        ),
        "12_ARIMA_代码.py": dedent(
            """
            import pandas as pd
            from statsmodels.tsa.arima.model import ARIMA


            def fit_arima(series: pd.Series, order=(1, 1, 1), steps: int = 6):
                model = ARIMA(series, order=order).fit()
                forecast = model.forecast(steps=steps)
                return model, forecast
            """
        ),
        "13_优化模型代码.py": dedent(
            """
            from scipy.optimize import linprog


            def solve_lp(c, A_ub, b_ub):
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
                return res
            """
        ),
        "14_遗传算法代码.py": dedent(
            """
            import random


            def genetic_optimize(fitness, low=0.0, high=1.0, pop_size=40, generations=100):
                pop = [random.uniform(low, high) for _ in range(pop_size)]
                for _ in range(generations):
                    pop = sorted(pop, key=fitness, reverse=True)
                    elite = pop[: pop_size // 3]
                    children = elite[:]
                    while len(children) < pop_size:
                        a, b = random.sample(elite, 2)
                        child = (a + b) / 2
                        if random.random() < 0.2:
                            child += random.uniform(-0.1, 0.1) * (high - low)
                        child = max(low, min(high, child))
                        children.append(child)
                    pop = children
                best = max(pop, key=fitness)
                return best, fitness(best)
            """
        ),
        "15_模拟退火代码.py": dedent(
            """
            import math
            import random


            def simulated_annealing(obj, x0, t0=10.0, alpha=0.95, t_min=1e-3, step=0.1, inner=60):
                x = x0
                best = x
                T = t0
                while T > t_min:
                    for _ in range(inner):
                        xn = x + random.uniform(-step, step)
                        de = obj(xn) - obj(x)
                        if de < 0 or random.random() < math.exp(-de / T):
                            x = xn
                            if obj(x) < obj(best):
                                best = x
                    T *= alpha
                return best, obj(best)
            """
        ),
        "16_灵敏度分析代码.py": dedent(
            """
            import numpy as np


            def oat_sensitivity(model_func, base_params: dict, perturb=0.1):
                base_output = model_func(**base_params)
                result = {}
                for k, v in base_params.items():
                    new_params = dict(base_params)
                    new_params[k] = v * (1 + perturb)
                    new_output = model_func(**new_params)
                    s = ((new_output - base_output) / base_output) / perturb
                    result[k] = s
                return result
            """
        ),
        "17_误差分析代码.py": dedent(
            """
            import numpy as np


            def calc_errors(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12)))
                return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            """
        ),
        "18_Word_LaTeX_导出代码.py": dedent(
            """
            from pathlib import Path


            def export_markdown(markdown_text: str, output_path: str) -> None:
                Path(output_path).write_text(markdown_text, encoding="utf-8")


            def export_latex_table(df, output_path: str) -> None:
                latex_str = df.to_latex(index=False)
                Path(output_path).write_text(latex_str, encoding="utf-8")
            """
        ),
    }

    for file_name, content in templates.items():
        write_file(base / file_name, content)


def build_chart_library() -> None:
    base = ROOT / "KB_图表模板库"
    base.mkdir(parents=True, exist_ok=True)

    readme = dedent(
        """
        # KB_图表模板库

        用途：供 `chart_maker` 直接复用论文图表样式与出图脚本。

        目录建议：
        - `00_图表风格规范.md`
        - `01_流程图模板.md`
        - `02_指标体系图模板.md`
        - `03_模型框架图模板.md`
        - `04_预测拟合图模板.py`
        - `05_误差分析图模板.py`
        - `06_灵敏度分析图模板.py`
        - `07_热力图模板.py`
        - `08_雷达图模板.py`
        - `09_表格格式模板.md`
        - `10_Matplotlib中文论文图表模板.py`
        """
    )
    write_file(base / "README.md", readme)

    style_doc = dedent(
        """
        # 优秀论文图表样式规范

        ## 基础规范
        - 字体：中文建议 `SimHei/Microsoft YaHei`，英文 `Times New Roman`。
        - 线宽：主线 1.8-2.2，辅助线 1.0-1.2。
        - 分辨率：论文提交图建议 `dpi>=300`。
        - 颜色：控制在 4-6 个主色，保证灰度打印可区分。

        ## 标注规范
        - 图题在图下方，表题在表上方。
        - 坐标轴必须含单位。
        - 图例尽量放在不遮挡数据区域。

        ## 信息密度建议
        - 每张图只讲一个核心结论。
        - 若图中包含多结论，拆分为子图(a)(b)(c)。
        """
    )
    write_file(base / "00_图表风格规范.md", style_doc)

    flowchart = dedent(
        """
        # 流程图模板（Mermaid）

        ```mermaid
        flowchart TD
            A[问题定义] --> B[数据获取与清洗]
            B --> C[指标构建]
            C --> D[模型建立]
            D --> E[参数求解]
            E --> F[结果分析]
            F --> G[灵敏度与误差分析]
            G --> H[结论与建议]
        ```
        """
    )
    write_file(base / "01_流程图模板.md", flowchart)

    indicator_diagram = dedent(
        """
        # 指标体系图模板（Mermaid）

        ```mermaid
        graph TD
            T[总目标] --> C1[一级指标A]
            T --> C2[一级指标B]
            T --> C3[一级指标C]
            C1 --> A1[二级指标A1]
            C1 --> A2[二级指标A2]
            C2 --> B1[二级指标B1]
            C2 --> B2[二级指标B2]
            C3 --> C31[二级指标C1]
            C3 --> C32[二级指标C2]
        ```
        """
    )
    write_file(base / "02_指标体系图模板.md", indicator_diagram)

    framework = dedent(
        """
        # 模型框架图模板（Mermaid）

        ```mermaid
        flowchart LR
            X[输入数据] --> P[预处理模块]
            P --> M1[主模型]
            P --> M2[辅助模型]
            M1 --> E[评估模块]
            M2 --> E
            E --> O[决策输出]
        ```
        """
    )
    write_file(base / "03_模型框架图模板.md", framework)

    pred_plot = dedent(
        """
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


        def plot_pred_fit(y_true, y_pred, title="预测拟合图"):
            plt.figure(figsize=(8, 5), dpi=160)
            plt.plot(y_true, label="真实值", linewidth=2)
            plt.plot(y_pred, label="预测值", linewidth=2, linestyle="--")
            plt.xlabel("样本序号")
            plt.ylabel("目标值")
            plt.title(title)
            plt.legend()
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
        """
    )
    write_file(base / "04_预测拟合图模板.py", pred_plot)

    err_plot = dedent(
        """
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


        def plot_residual(y_true, y_pred, title="误差分析图"):
            residual = np.asarray(y_true) - np.asarray(y_pred)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
            axes[0].scatter(y_pred, residual, alpha=0.8)
            axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
            axes[0].set_title("残差-预测值")
            axes[0].set_xlabel("预测值")
            axes[0].set_ylabel("残差")

            axes[1].hist(residual, bins=20, alpha=0.85, edgecolor="black")
            axes[1].set_title("残差分布")
            axes[1].set_xlabel("残差")
            axes[1].set_ylabel("频数")
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        """
    )
    write_file(base / "05_误差分析图模板.py", err_plot)

    sens_plot = dedent(
        """
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


        def plot_sensitivity(sens_dict, title="灵敏度分析图"):
            keys = list(sens_dict.keys())
            vals = [sens_dict[k] for k in keys]
            plt.figure(figsize=(8, 4), dpi=160)
            plt.bar(keys, vals)
            plt.axhline(0, color="black", linewidth=1)
            plt.title(title)
            plt.ylabel("灵敏度系数")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.show()
        """
    )
    write_file(base / "06_灵敏度分析图模板.py", sens_plot)

    heatmap = dedent(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


        def plot_heatmap(matrix, title="热力图"):
            plt.figure(figsize=(7, 5), dpi=160)
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(title)
            plt.tight_layout()
            plt.show()
        """
    )
    write_file(base / "07_热力图模板.py", heatmap)

    radar = dedent(
        """
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False


        def plot_radar(labels, values, title="雷达图"):
            n = len(labels)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            values = np.concatenate([values, [values[0]]])
            angles = np.concatenate([angles, [angles[0]]])
            fig = plt.figure(figsize=(6, 6), dpi=160)
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(title)
            plt.tight_layout()
            plt.show()
        """
    )
    write_file(base / "08_雷达图模板.py", radar)

    table_style = dedent(
        """
        # 表格格式模板

        ## 三线表建议
        - 仅保留顶线、表头线、底线。
        - 小数位统一（一般 2-4 位）。
        - 指标命名统一中英文风格，不混杂简称和全称。

        ## 结果表推荐列
        - 指标名
        - 模型A
        - 模型B
        - 提升率
        - 备注
        """
    )
    write_file(base / "09_表格格式模板.md", table_style)

    mpl_cn = dedent(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 160


        def setup_paper_style():
            sns.set_theme(
                context="paper",
                style="whitegrid",
                palette="deep",
                font="sans-serif",
                rc={
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                },
            )
        """
    )
    write_file(base / "10_Matplotlib中文论文图表模板.py", mpl_cn)


def build_scoring_library() -> None:
    base = ROOT / "KB_论文评分库"
    base.mkdir(parents=True, exist_ok=True)

    readme = dedent(
        """
        # KB_论文评分库

        用途：
        - 给 `reviewer` 做结构化审稿
        - 给 `paper_writer` 做写作前自检

        说明：
        - CUMCM 官方“评阅工作规范”“论文格式规范”以组委会最新版为准。
        - MCM/ICM 以 COMAP Rules/Instructions 与结果分级机制为准。
        """
    )
    write_file(base / "README.md", readme)

    score_std = dedent(
        """
        # 数学建模竞赛评分标准（实用版）

        ## 通用评分维度（建议用于内部自评）
        1. 问题理解与重述（15%）
        2. 假设合理性与变量定义（10%）
        3. 模型构建质量（25%）
        4. 求解与结果可靠性（20%）
        5. 结果分析与讨论（10%）
        6. 图表与论文表达（10%）
        7. 创新性与可推广性（10%）

        ## 官方依据（建议优先阅读）
        - CUMCM 赛区评阅工作规范（2023修订稿入口）  
          https://www.mcm.edu.cn/html_cn/node/011a3fefdb4951a8cb595400f44ec3df.html
        - CUMCM 全国奖项评阅工作规范（2023修订稿入口）  
          https://www.mcm.edu.cn/html_cn/node/b1f48689659f0660e80a2d6279d7b37d.html
        - COMAP Rules/Instructions  
          https://www.contest.comap.org/undergraduate/contests/mcm/instructions.html
        """
    )
    write_file(base / "01_数学建模竞赛评分标准.md", score_std)

    format_spec = dedent(
        """
        # 国赛论文格式规范

        ## 最新可用官方链接（2026-05-23检索）
        - 全国大学生数学建模竞赛论文格式规范（2026年修订稿 PDF）
          https://www.mcm.edu.cn/upload_cn/node/775/cQMeL0YY905244c8bd4b9af832f1699446d8385e.pdf

        ## 自检要点
        - 摘要页是否独立且不超过一页
        - 页码是否按规范位置连续编号
        - 电子版论文与纸质版内容一致
        - 支撑材料（代码、数据、附录）是否齐全
        """
    )
    write_file(base / "02_国赛论文格式规范.md", format_spec)

    abstract_std = dedent(
        """
        # 摘要评价标准

        ## 四段式摘要建议
        1. 问题背景与目标（1句）
        2. 方法体系与关键模型（1-2句）
        3. 核心结果（必须量化，1句）
        4. 结论与应用价值（1句）

        ## 评分关注点
        - 是否“先结果后方法”或“先方法后结果”逻辑清晰
        - 是否出现空话（如“效果很好”但无数值）
        - 是否体现创新点或模型优势
        """
    )
    write_file(base / "03_摘要评价标准.md", abstract_std)

    minus_points = dedent(
        """
        # 常见扣分点

        1. 变量、符号未定义或前后不一致
        2. 公式堆砌但缺乏解释
        3. 图表无单位、无图题、图例混乱
        4. 仅给结果，不给求解过程或验证过程
        5. 代码不可运行或关键步骤缺失
        6. 结论与前文模型结果不对应
        7. 参考文献格式混乱，引用不规范
        8. 附录堆大量无关代码影响可读性
        """
    )
    write_file(base / "04_常见扣分点.md", minus_points)

    comments_doc = dedent(
        """
        # 优秀论文评语模板

        ## 正向评语
        - 论文问题重述准确，建模目标清晰，假设边界明确。
        - 模型组合合理，主辅模型衔接自然，结果验证充分。
        - 图表表达规范，能有效支撑结论，工程解释较强。

        ## 改进建议评语
        - 建议补充参数敏感性分析，增强模型稳健性论证。
        - 建议在附录增加关键代码注释与运行说明，提升可复现性。
        - 建议对异常样本单独讨论，避免均值掩盖结构性偏差。
        """
    )
    write_file(base / "05_优秀论文评语.md", comments_doc)

    chart_norm = dedent(
        """
        # 图表规范

        - 图题完整：图号 + 图名 + 必要单位
        - 轴标签完整：名称 + 单位
        - 图例不遮挡关键数据区
        - 多图对比时坐标范围应可比
        - 建议统一配色与字号，提升整体专业性
        """
    )
    write_file(base / "06_图表规范.md", chart_norm)

    formula_norm = dedent(
        """
        # 公式规范

        - 关键公式应编号（如(1)(2)(3)）
        - 首次出现的符号必须定义
        - 推导中省略步骤应给出解释性文字
        - 向量、矩阵、标量符号风格统一
        """
    )
    write_file(base / "07_公式规范.md", formula_norm)

    ref_norm = dedent(
        """
        # 参考文献规范

        ## 建议格式（GB/T 7714 常用）
        - 期刊：[序号] 作者. 题名[J]. 刊名, 年, 卷(期): 起止页码.
        - 图书：[序号] 作者. 书名[M]. 出版地: 出版社, 年.
        - 学位论文：[序号] 作者. 题名[D]. 学校, 年.
        - 网络资源：[序号] 作者/机构. 标题[EB/OL]. URL, 访问日期.

        ## 自检
        - 引文与文末条目一一对应
        - 年份、卷期、页码完整
        - URL 可访问
        """
    )
    write_file(base / "08_参考文献规范.md", ref_norm)

    appendix_norm = dedent(
        """
        # 附录代码规范

        - 必须包含：运行环境、依赖库、主入口脚本
        - 关键函数应有简短注释（输入/输出）
        - 数据读取路径建议相对路径
        - 随机过程建议固定随机种子以便复现
        - 附录中仅保留关键代码，过长脚本可外链仓库
        """
    )
    write_file(base / "09_附录代码规范.md", appendix_norm)


def build_root_readme() -> None:
    content = dedent(
        """
        # MathMoudelMaster 知识库

        已生成以下 5 个知识库目录：
        - `KB_优秀论文库`
        - `KB_建模方法库`
        - `KB_代码模板库`
        - `KB_图表模板库`
        - `KB_论文评分库`

        推荐导入顺序（Nexent）：
        1. 先导入 `KB_优秀论文库` + `KB_论文评分库`（写作与评审策略）
        2. 再导入 `KB_建模方法库`（方法知识）
        3. 最后导入 `KB_代码模板库` + `KB_图表模板库`（执行层）
        """
    )
    write_file(ROOT / "README.md", content)


def main() -> None:
    build_root_readme()
    build_paper_library()
    build_method_library()
    build_code_library()
    build_chart_library()
    build_scoring_library()
    print("Knowledge base generation completed.")


if __name__ == "__main__":
    main()
