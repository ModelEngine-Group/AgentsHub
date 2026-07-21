"""
Offline Review Engine - 离线论文评审，不需要API也能给出专业评审
基于规则+知识库，分析论文结构、内容质量、国赛标准
"""
import re
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════
# 知识库
# ═══════════════════════════════════════════════════════

MODEL_KB = {
    "optimization": {
        "name": "优化类",
        "models": [
            {"name": "线性规划", "method": "linprog/scipy.optimize", "keywords": ["线性规划", "目标函数", "约束条件", "单纯形"]},
            {"name": "整数规划", "method": "pulp/gurobipy", "keywords": ["整数规划", "0-1规划", "指派问题"]},
            {"name": "非线性规划", "method": "scipy.optimize.minimize", "keywords": ["非线性", "梯度", "拉格朗日"]},
            {"name": "多目标优化", "method": "NSGA-II/pymoo", "keywords": ["多目标", "Pareto", "pareto最优"]},
            {"name": "动态规划", "method": "自定义DP", "keywords": ["动态规划", "状态转移", "最优子结构"]},
            {"name": "图论最短路径", "method": "networkx/dijkstra", "keywords": ["最短路径", "Dijkstra", "Floyd"]},
        ],
    },
    "statistics": {
        "name": "统计类",
        "models": [
            {"name": "回归分析", "method": "sklearn.linear_model", "keywords": ["回归", "拟合", "最小二乘"]},
            {"name": "方差分析", "method": "scipy.stats", "keywords": ["方差分析", "ANOVA", "F检验"]},
            {"name": "主成分分析", "method": "sklearn.decomposition.PCA", "keywords": ["PCA", "主成分", "降维"]},
            {"name": "聚类分析", "method": "sklearn.cluster", "keywords": ["聚类", "K-means", "层次聚类"]},
        ],
    },
    "prediction": {
        "name": "预测类",
        "models": [
            {"name": "时间序列(ARIMA)", "method": "statsmodels.tsa", "keywords": ["ARIMA", "时间序列", "平稳性"]},
            {"name": "灰色预测(GM)", "method": "自定义GM(1,1)", "keywords": ["灰色预测", "GM(1,1)", "累加生成"]},
            {"name": "神经网络预测", "method": "torch/sklearn.neural_network", "keywords": ["神经网络", "BP", "LSTM"]},
            {"name": "马尔可夫链", "method": "自定义转移矩阵", "keywords": ["马尔可夫", "转移概率", "状态转移"]},
            {"name": "曲线拟合", "method": "scipy.optimize.curve_fit", "keywords": ["曲线拟合", "指数拟合", "多项式拟合"]},
        ],
    },
    "physics": {
        "name": "物理/工程类",
        "models": [
            {"name": "微分方程", "method": "scipy.integrate.solve_ivp", "keywords": ["微分方程", "ODE", "PDE", "龙格库塔"]},
            {"name": "有限元分析", "method": "fenics/fipy", "keywords": ["有限元", "FEM", "网格"]},
            {"name": "蒙特卡洛模拟", "method": "numpy.random", "keywords": ["蒙特卡洛", "随机模拟", "概率"]},
        ],
    },
}

REVIEW_RUBRIC = {
    "math_modeling": {
        "abstract": {"max": 20, "checks": [
            ("五要素", r"问题|方法|模型|算法|结果|结论", 8, "摘要必须包含：问题、方法、模型、算法、结果、结论"),
            ("字数", None, 4, "摘要300-500字为佳"),
            ("数据", r"\d+\.?\d*%?|提高了?\d+|降低了?\d+|优化了?\d+", 4, "摘要应包含关键数据结果"),
            ("创新点", r"创新|改进|提出|新[的方]|首次", 4, "摘要应突出创新点"),
        ]},
        "model": {"max": 30, "checks": [
            ("假设合理性", r"假设|假定|设[定为]|不妨设", 6, "需要明确的模型假设"),
            ("符号说明", r"符号|其中|表示|定义|设.*?[=＝]", 5, "需要规范的符号说明"),
            ("公式推导", r"[=＝→⇒]|公式|方程|约束|目标函数|min|max|s\.t\.", 10, "需要完整的数学推导"),
            ("创新性", r"改进|优化|提出|新[的方]|结合|融合", 5, "模型应有创新点"),
            ("适用性", r"适用|推广|局限|条件|范围", 4, "应讨论模型适用范围"),
        ]},
        "algorithm": {"max": 25, "checks": [
            ("算法选择", r"算法|求解|遗传|粒子群|模拟退火|蚁群|梯度|牛顿|迭代", 8, "需要明确的求解算法"),
            ("算法描述", r"步骤|流程|输入|输出|初始化|终止|收敛", 8, "需要清晰的算法描述"),
            ("复杂度", r"复杂度|O\(|时间.*?空间|计算量|效率", 4, "应分析算法复杂度"),
            ("可运行代码", r"import|def |class |for |while |return|print|function\s|\bend;?|disp\(", 5, "应提供可运行的代码"),
            ("MATLAB代码", r"function\s|\bend;?|\bdisp\(|\bfprintf\(|\bplot\(|\bsyms?\b|MATLAB", 3, "包含MATLAB代码，体现了工程实践能力"),
            ("LaTeX公式", r"\\begin\{|\\frac|\\int|\\sum|\\prod|\\sqrt|\$\$", 3, "包含LaTeX数学公式，体现学术规范"),
        ]},
        "results": {"max": 20, "checks": [
            ("结果正确", r"结果|输出|最优|解[为是]|值[为是]", 6, "需要展示求解结果"),
            ("图表", r"图\s*\d|表\s*\d|Figure|Table|如图|如表|见图|见表", 6, "需要用图表展示结果"),
            ("灵敏度分析", r"灵敏|敏感|稳健|鲁棒|扰动|参数.*?变化", 4, "应进行灵敏度分析"),
            ("对比", r"对比|比较|对照|优于|高于|低于|提升", 4, "应有对比分析"),
        ]},
        "writing": {"max": 15, "checks": [
            ("结构完整", r"摘[要关]|关键词|问题重述|模型假设|符号说明|模型建立|参考文献|附录", 5, "论文结构应完整"),
            ("参考文献", r"\[\d+\]|参考文献|引用|\d{4}.*?(?:et al|等人|等)\.", 5, "需要规范的参考文献"),
            ("格式规范", r"第[一二三四五]章|§|\\section|\\subsection|\\begin\{", 5, "应有清晰的章节划分"),
            ("LaTeX排版", r"\\documentclass|\\usepackage|\\begin\{document\}|\\maketitle", 3, "使用LaTeX排版，体现专业性"),
        ]},
    },
}

COMMON_MISTAKES = [
    {"pattern": r"摘[要关].*?(?:问题|背景|研究)", "msg": "摘要以背景开头，应直接陈述问题和方法", "fix": "摘要应按：问题→方法→模型→结果→结论 的顺序"},
    {"pattern": r"(?:假设|假定)\s*[：:]?\s*$", "msg": "假设部分为空或过于简略", "fix": "列出3-5条合理的简化假设，每条说明理由"},
    {"pattern": r"模型\s*建立", "msg": "模型建立部分可能缺少数学公式", "fix": "用数学符号定义变量，写出目标函数和约束条件"},
    {"pattern": r"(?:结果|结论)\s*[：:]?\s*$", "msg": "结果部分可能缺少具体数据", "fix": "给出具体的数值结果，用表格或图表展示"},
    {"pattern": r"(?:图|表)\s*\d", "msg": "图表可能缺少文字分析", "fix": "每张图/表后用1-2句话分析其含义"},
    {"pattern": r"参考文献\s*[：:]?\s*$", "msg": "参考文献部分为空", "fix": "引用5-10篇相关文献，格式统一"},
    {"pattern": r"(?:代码|程序)", "msg": "提到了代码但可能未放入附录", "fix": "将核心代码放入附录，确保可运行"},
]


class OfflineReviewEngine:
    """离线论文评审引擎 - 不需要API，基于规则+知识库"""

    def analyze(self, content: str) -> Dict:
        """深度分析论文内容"""
        content = content.strip()
        if len(content) < 30:
            return {"error": "内容过短", "word_count": len(content)}

        # 分段
        paragraphs = [p.strip() for p in re.split(r'\n+', content) if p.strip()]

        # 检测章节
        sections = self._detect_sections(content)

        # 检测公式
        formulas = self._extract_formulas(content)

        # 检测方法
        methods = self._extract_methods(content)

        # 检测图表引用
        figure_refs = re.findall(r'图\s*\d|Figure|如图|见图', content)
        table_refs = re.findall(r'表\s*\d|Table|如表|见表', content)

        # 检测参考文献
        refs = re.findall(r'\[\d+\]|参考文献|引用', content)

        # 检测问题类型
        problem_type = self._detect_problem_type(content)

        return {
            "word_count": len(content),
            "char_count": len(content.replace(" ", "").replace("\n", "")),
            "paragraph_count": len(paragraphs),
            "sections": sections,
            "section_count": len(sections),
            "formula_count": len(formulas),
            "formulas": formulas[:5],
            "methods": methods,
            "figure_refs": len(figure_refs),
            "table_refs": len(table_refs),
            "ref_count": len(refs),
            "problem_type": problem_type,
            "has_abstract": bool(re.search(r'摘[要关]', content[:1000])),
            "has_model": bool(re.search(r'模型建立|模型[的建]|建模', content)),
            "has_algorithm": bool(re.search(r'算法|求解|遗传|粒子群|模拟退火|优化', content)),
            "has_results": bool(re.search(r'结果|输出|最优|结论', content)),
            "has_references": bool(re.search(r'参考文献|\[\d+\]', content)),
            "has_code": bool(re.search(r'import|def |class |for .* in|while |return ', content)),
            "has_matlab": bool(re.search(r'function\s|end;?|\belseif\b|\bdisp\(|\bfprintf\(|\bplot\(|\bscatter\(|\bsolve\(|\bsyms?\b|MATLAB|matlab|\.m\b', content)),
            "has_latex": bool(re.search(r'\\begin\{|\\end\{|\\frac|\\int|\\sum|\\prod|\\sqrt|\\alpha|\\beta|\\gamma|\$\$|\\[a-zA-Z]+\{', content)),
            "has_python_code": bool(re.search(r'import (numpy|scipy|sklearn|matplotlib|pandas|torch|tensorflow)|def \w+\(|class \w+:', content)),
            "matlab_blocks": len(re.findall(r'function\s|\bend;?|\bdisp\(|\bfprintf\(|\bplot\(', content)),
            "latex_formulas": len(re.findall(r'\\[a-zA-Z]+\{|\$\$|\\frac|\\int|\\sum', content)),
            "python_blocks": len(re.findall(r'import (numpy|scipy|sklearn|matplotlib)|def \w+\(', content)),
        }

    def review(self, content: str, paper_type: str = "math_modeling") -> Dict:
        """完整评审 - 返回评分+反馈+改进建议"""
        content = content.strip()
        if len(content) < 30:
            return {
                "total_score": 0,
                "grade": "无法评审",
                "message": "论文内容过短或无法提取文本。可能是扫描件PDF，请上传可选中文字的PDF或直接粘贴文本。",
                "dimensions": [],
                "feedback": [],
                "improvements": [],
            }

        analysis = self.analyze(content)
        rubric = REVIEW_RUBRIC.get(paper_type, REVIEW_RUBRIC["math_modeling"])

        dimensions = []
        total_score = 0
        all_feedback = []

        for dim_name, dim_config in rubric.items():
            dim_score = 0
            dim_max = dim_config["max"]
            dim_feedback = []

            for check_name, pattern, points, hint in dim_config["checks"]:
                if pattern and re.search(pattern, content):
                    dim_score += points
                elif pattern:
                    dim_feedback.append(f"✗ {check_name}：{hint}")

            dim_score = min(dim_score, dim_max)
            total_score += dim_score
            dimensions.append({
                "name": self._dim_cn(dim_name),
                "score": dim_score,
                "max_score": dim_max,
                "percentage": round(dim_score / dim_max * 100) if dim_max else 0,
            })
            all_feedback.extend(dim_feedback)

        # 检测常见错误
        for mistake in COMMON_MISTAKES:
            if re.search(mistake["pattern"], content):
                all_feedback.append(f"⚠ {mistake['msg']}。建议：{mistake['fix']}")

        # 内容深度加分
        depth_bonus = 0
        if analysis["word_count"] > 3000: depth_bonus += 3
        if analysis["word_count"] > 5000: depth_bonus += 3
        if analysis["word_count"] > 8000: depth_bonus += 2
        if analysis["formula_count"] > 3: depth_bonus += 2
        if analysis["figure_refs"] > 2: depth_bonus += 2
        if analysis["ref_count"] > 2: depth_bonus += 1
        if analysis["has_code"]: depth_bonus += 2
        if analysis.get("has_matlab"): depth_bonus += 3
        if analysis.get("has_latex"): depth_bonus += 2
        if analysis.get("has_python_code"): depth_bonus += 2
        total_score = min(100, total_score + depth_bonus)

        # 等级
        if total_score >= 90: grade = "优秀 (国奖水平)"
        elif total_score >= 80: grade = "良好 (省奖水平)"
        elif total_score >= 70: grade = "中等"
        elif total_score >= 60: grade = "及格"
        else: grade = "不及格"

        # 改进建议（按优先级）
        improvements = self._generate_improvements(analysis, all_feedback)

        # 模型推荐
        recommendations = self._recommend_models(analysis)

        return {
            "total_score": total_score,
            "grade": grade,
            "analysis_summary": (
                f"论文共 {analysis['word_count']} 字，{analysis['section_count']} 个章节，"
                f"{analysis['formula_count']} 个公式，{analysis['figure_refs']} 处图表引用，"
                f"{analysis['ref_count']} 处参考文献。"
                f"检测到问题类型：{analysis['problem_type']['name'] if analysis['problem_type'] else '未知'}。"
                f"主要方法：{', '.join(analysis['methods'][:3]) if analysis['methods'] else '未检测到'}。"
                f"代码语言：{'MATLAB' if analysis.get('has_matlab') else ''}"
                f"{' + Python' if analysis.get('has_python_code') else ''}"
                f"{' + LaTeX公式' if analysis.get('has_latex') else ''}"
                f"{'（未检测到代码）' if not analysis.get('has_matlab') and not analysis.get('has_python_code') else ''}"
            ),
            "dimensions": dimensions,
            "feedback": all_feedback,
            "improvements": improvements,
            "recommendations": recommendations,
            "analysis": analysis,
        }

    def _detect_sections(self, content: str) -> Dict[str, str]:
        """检测论文章节"""
        patterns = {
            "摘要": r'(?:摘\s*要|Abstract)[：:\s]*(.*?)(?=关键词|关键字|$)',
            "问题重述": r'(?:问题重述|一[、.]?\s*问题|1[、.\s]?\s*问题)(.*?)(?=问题分析|二[、.]|$)',
            "问题分析": r'(?:问题分析|二[、.]?\s*问题|2[、.\s]?\s*问题)(.*?)(?=模型假设|三[、.]|$)',
            "模型假设": r'(?:模型假设|三[、.]?\s*模型|3[、.\s]?\s*模型)(.*?)(?=符号说明|模型建立|$)',
            "符号说明": r'(?:符号说明|符号|四[、.])(.*?)(?=模型建立|五[、.]|$)',
            "模型建立": r'(?:模型建立|模型的建立|五[、.]?\s*模型|4[、.\s]?\s*模型|5[、.\s]?\s*模型)(.*?)(?=模型求解|模型检验|灵敏度|$)',
            "模型求解": r'(?:模型求解|求解|六[、.])(.*?)(?=结果分析|模型检验|灵敏度|七[、.]|$)',
            "结果分析": r'(?:结果分析|七[、.])(.*?)(?=模型评价|参考文献|$)',
            "模型评价": r'(?:模型评价|模型推广|八[、.])(.*?)(?=参考文献|$)',
            "参考文献": r'(?:参考文献)(.*?)(?=$)',
        }
        sections = {}
        for name, pattern in patterns.items():
            m = re.search(pattern, content, re.DOTALL)
            if m:
                sections[name] = m.group(1).strip()[:500]
        return sections

    def _extract_formulas(self, content: str) -> List[str]:
        """提取数学公式"""
        patterns = [
            r'\$\$(.*?)\$\$',
            r'\$([^$]+)\$',
            r'[a-zA-Z]\s*[=]\s*[a-zA-Z0-9+\-*/^() ]+',
            r'(?:min|max)\s*\(',
        ]
        formulas = []
        for p in patterns:
            try:
                matches = re.findall(p, content)
                formulas.extend(matches)
            except re.error:
                pass
        return list(set([f.strip() for f in formulas if f.strip()]))[:10]

    def _extract_methods(self, content: str) -> List[str]:
        """提取方法/算法"""
        all_methods = []
        for ptype, pdata in MODEL_KB.items():
            for model in pdata["models"]:
                for kw in model["keywords"]:
                    if kw in content:
                        all_methods.append(model["name"])
                        break
        return list(set(all_methods))

    def _detect_problem_type(self, content: str) -> Dict:
        """检测问题类型"""
        scores = {}
        for ptype, pdata in MODEL_KB.items():
            score = 0
            for model in pdata["models"]:
                for kw in model["keywords"]:
                    if kw in content:
                        score += 1
            scores[ptype] = score

        if max(scores.values(), default=0) == 0:
            return {"type": "unknown", "name": "未知类型", "confidence": 0}

        best = max(scores, key=scores.get)
        return {
            "type": best,
            "name": MODEL_KB[best]["name"],
            "confidence": min(100, scores[best] * 15),
        }

    def _recommend_models(self, analysis: Dict) -> List[Dict]:
        """推荐模型"""
        ptype = analysis.get("problem_type", {}).get("type", "unknown")
        if ptype in MODEL_KB:
            return MODEL_KB[ptype]["models"][:3]
        # 默认推荐通用模型
        return [
            {"name": "回归分析", "method": "sklearn", "keywords": []},
            {"name": "聚类分析", "method": "sklearn", "keywords": []},
        ]

    def _generate_improvements(self, analysis: Dict, feedback: List[str]) -> List[Dict]:
        """生成改进建议"""
        improvements = []
        if not analysis["has_abstract"]:
            improvements.append({"priority": "高", "action": "添加摘要", "detail": "摘要必须包含：问题、方法、模型、算法、结果、结论"})
        if analysis["formula_count"] < 3:
            improvements.append({"priority": "高", "action": "增加数学公式", "detail": "用LaTeX格式写出目标函数、约束条件、求解公式"})
        if analysis["figure_refs"] < 2:
            improvements.append({"priority": "中", "action": "添加图表", "detail": "用matplotlib生成结果可视化图，用表格展示关键数据"})
        if analysis["ref_count"] < 3:
            improvements.append({"priority": "中", "action": "补充参考文献", "detail": "引用5-10篇相关论文，格式统一"})
        if analysis["word_count"] < 3000:
            improvements.append({"priority": "中", "action": "扩充内容", "detail": f"当前{analysis['word_count']}字，建议至少5000字"})
        if not analysis["has_code"]:
            improvements.append({"priority": "低", "action": "添加代码附录", "detail": "将核心算法代码放入附录，确保可运行"})
        return improvements

    def _dim_cn(self, name: str) -> str:
        mapping = {"abstract": "摘要", "model": "模型建立", "algorithm": "求解方法", "results": "结果分析", "writing": "论文写作"}
        return mapping.get(name, name)


# Singleton
offline_engine = OfflineReviewEngine()
