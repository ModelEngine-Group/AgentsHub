#!/usr/bin/env python3
"""
data_knowledge_insight_demo
=============================
基于 Nexent + DataMate 的"数据—知识—洞察"智能体 Demo 一键运行入口。

用法:
    python run_demo.py           # 显示帮助
    python run_demo.py task1     # 运行任务一：数据处理智能体
    python run_demo.py task2     # 运行任务二：医疗知识图谱生成与问答
    python run_demo.py task3     # 运行任务三：数据分析与可视化
    python run_demo.py all      # 运行完整闭环
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CLI_DIR = PROJECT_ROOT / "cli"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HELP_TEXT = """
data_knowledge_insight_demo - 基于 Nexent + DataMate 的"数据—知识—洞察"智能体 Demo

用法:
    python run_demo.py           显示本帮助
    python run_demo.py task1    运行任务一：数据处理智能体
    python run_demo.py task2    运行任务二：医疗知识图谱生成与问答
    python run_demo.py task3    运行任务三：数据分析与可视化
    python run_demo.py all      运行完整闭环（task1 -> task2 -> task3）

输出目录: outputs/
    cleaned_records.csv      任务一输出：清洗后的医疗记录
    cleaned_text.json       任务一输出：清洗后的医疗文本
    medical_triples.json    任务二输出：医疗三元组
    medical_kg.json         任务二输出：医疗知识图谱
    medical_kg.html         任务二输出：医疗知识图谱可视化
    charts/                任务三输出：分析图表
    analysis_report.md      任务三输出：分析报告

前置要求:
    依赖已通过 uv install 安装，运行前确保 Python 环境激活。
"""


def ensure_outputs_dir():
    """确保 outputs 目录存在"""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    charts_dir = OUTPUTS_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)


def run_task1():
    """运行任务一：数据处理智能体"""
    print("\n" + "=" * 60)
    print("任务一：数据处理智能体")
    print("=" * 60)
    try:
        from pipelines.task1_pipeline import run as task1_run
        task1_run()
    except Exception as e:
        print(f"[ERROR] 任务一运行失败: {e}")
        print("[INFO] 任务一尚未实现，将创建占位输出...")
        _create_task1_placeholder()


def run_task2():
    """运行任务二：医疗知识图谱生成与问答"""
    print("\n" + "=" * 60)
    print("任务二：医疗知识图谱生成与问答")
    print("=" * 60)
    try:
        from pipelines.task2_pipeline import run as task2_run
        task2_run()
    except Exception as e:
        print(f"[ERROR] 任务二运行失败: {e}")
        print("[INFO] 任务二尚未实现，将创建占位输出...")
        _create_task2_placeholder()


def run_task3():
    """运行任务三：数据分析与可视化"""
    print("\n" + "=" * 60)
    print("任务三：数据分析与可视化")
    print("=" * 60)
    try:
        from pipelines.task3_pipeline import run as task3_run
        task3_run()
    except Exception as e:
        print(f"[ERROR] 任务三运行失败: {e}")
        print("[INFO] 任务三尚未实现，将创建占位输出...")
        _create_task3_placeholder()


def run_all():
    """运行完整闭环"""
    print("\n" + "=" * 60)
    print("运行完整闭环：数据 → 知识 → 洞察")
    print("=" * 60)
    run_task1()
    run_task2()
    run_task3()
    print("\n" + "=" * 60)
    print("完整闭环运行完成！")
    print("=" * 60)
    _print_outputs()


def _create_task1_placeholder():
    """创建任务一占位输出"""
    ensure_outputs_dir()
    import json
    placeholder = {
        "status": "placeholder",
        "message": "任务一数据处理智能体正在实现中",
        "sample_output": {
            "total_records": 15,
            "cleaned_fields": ["patient_id", "age", "gender", "diagnosis", "medicine"]
        }
    }
    out_path = OUTPUTS_DIR / "task1_status.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)
    print(f"[PLACEHOLDER] 任务一状态已写入: {out_path}")


def _create_task2_placeholder():
    """创建任务二占位输出"""
    ensure_outputs_dir()
    import json
    placeholder = {
        "status": "placeholder",
        "message": "任务二医疗知识图谱正在实现中",
        "sample_entities": ["肺炎", "冠心病", "偏头痛"],
        "sample_relations": ["has_symptom", "treated_by"]
    }
    out_path = OUTPUTS_DIR / "task2_status.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)
    print(f"[PLACEHOLDER] 任务二状态已写入: {out_path}")


def _create_task3_placeholder():
    """创建任务三占位输出"""
    ensure_outputs_dir()
    import json
    placeholder = {
        "status": "placeholder",
        "message": "任务三数据分析与可视化正在实现中"
    }
    out_path = OUTPUTS_DIR / "task3_status.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)
    print(f"[PLACEHOLDER] 任务三状态已写入: {out_path}")


def _print_outputs():
    """打印当前输出文件列表"""
    print("\n当前 outputs 目录内容：")
    for p in sorted(OUTPUTS_DIR.rglob("*")):
        rel = p.relative_to(OUTPUTS_DIR)
        marker = "[DIR] " if p.is_dir() else "[FILE] "
        print(f"  {marker}{rel}")


def main():
    ensure_outputs_dir()

    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print(HELP_TEXT)
        return

    cmd = sys.argv[1].lower()
    if cmd == "task1":
        run_task1()
    elif cmd == "task2":
        run_task2()
    elif cmd == "task3":
        run_task3()
    elif cmd == "all":
        run_all()
    else:
        print(f"未知命令: {cmd}")
        print("运行 python run_demo.py 查看帮助")


if __name__ == "__main__":
    main()
