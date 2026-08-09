# -*- coding: utf-8 -*-
"""
DataLoader 算子综合测试脚本
基于 CCF 比赛要求设计，覆盖功能、编码、容错、性能等维度

运行方式：python test_data_loader.py
"""

import sys
import os
import time
import json
from datetime import datetime

# 添加算子路径
_OPERATORS_DIR = r"D:\PythonProject\ModelEngine\operators"
if _OPERATORS_DIR not in sys.path:
    sys.path.insert(0, _OPERATORS_DIR)

# 使用 try-except 处理导入，兼容 PyCharm 和命令行
try:
    from data_loader.process import DataLoaderMapper
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"请检查路径: {_OPERATORS_DIR}")
    raise

# ==================== 测试配置 ====================

TEST_DATA_DIR = "D:/PythonProject/OperatorTest/test_data"
REPORT_DIR = "D:/PythonProject/OperatorTest/test_results"

# 测试结果收集
test_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total": 0,
    "passed": 0,
    "failed": 0,
    "details": []
}

# ==================== 测试工具函数 ====================

def log(msg, level="INFO"):
    """统一日志输出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "📋", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
    print(f"[{timestamp}] {icons.get(level, '📋')} {msg}")


def assert_check(condition, success_msg, fail_msg):
    """断言检查"""
    global test_results
    test_results["total"] += 1

    if condition:
        test_results["passed"] += 1
        test_results["details"].append({"status": "PASS", "msg": success_msg})
        log(success_msg, "PASS")
        return True
    else:
        test_results["failed"] += 1
        test_results["details"].append({"status": "FAIL", "msg": fail_msg})
        log(fail_msg, "FAIL")
        return False


def save_report():
    """保存测试报告"""
    report_path = os.path.join(
        REPORT_DIR,
        f"data_loader_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    return report_path


# ==================== 测试用例 ====================

# ===== 1. 功能正确性测试 =====

def test_single_csv_load():
    """测试1：单文件 CSV 加载"""
    log("=" * 50)
    log("测试1：单文件 CSV 加载")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "sample_utf8.csv")],
        file_formats=["csv"]
    )
    result = loader({})

    assert_check(
        result["execute_result"] == True,
        f"✅ 加载成功，共 {result['count']} 条数据",
        f"❌ 加载失败: {result.get('errors', ['未知错误'])}"
    )

    assert_check(
        result["count"] == 5,
        f"✅ 数据条数正确: 5 条",
        f"❌ 数据条数错误: 期望 5，实际 {result['count']}"
    )

    assert_check(
        result["quality_score"] > 0,
        f"✅ 质量评分正常: {result['quality_score']}",
        f"❌ 质量评分为 0"
    )


def test_json_load():
    """测试2：JSON 文件加载"""
    log("=" * 50)
    log("测试2：JSON 文件加载")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "sample.json")],
        file_formats=["json"]
    )
    result = loader({})

    assert_check(
        result["execute_result"] == True,
        f"✅ JSON 加载成功，共 {result['count']} 条",
        f"❌ JSON 加载失败"
    )

    assert_check(
        result["count"] == 3,
        f"✅ JSON 数据正确: 3 条",
        f"❌ JSON 数据错误"
    )


def test_single_json_object():
    """测试3：单对象 JSON 加载"""
    log("=" * 50)
    log("测试3：单对象 JSON 加载")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "sample_single.json")],
        file_formats=["json"]
    )
    result = loader({})

    assert_check(
        result["count"] == 1,
        f"✅ 单对象 JSON 正确转为数组: 1 条",
        f"❌ 单对象处理错误"
    )


def test_batch_multi_format():
    """测试4：多格式批量加载"""
    log("=" * 50)
    log("测试4：多格式批量加载")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"),
            os.path.join(TEST_DATA_DIR, "sample.json")
        ],
        file_formats=["csv", "json"],
        merge_strategy="concat"
    )
    result = loader({})

    assert_check(
        result["success_count"] == 2,
        f"✅ 多格式加载成功: {result['success_count']} 个文件",
        f"❌ 多格式加载失败"
    )

    assert_check(
        result["count"] == 8,  # 5 + 3
        f"✅ 数据合并正确: {result['count']} 条",
        f"❌ 数据合并错误: 期望 8，实际 {result['count']}"
    )


# ===== 2. 编码兼容性测试 =====

def test_encoding_gbk():
    """测试5：GBK 编码支持"""
    log("=" * 50)
    log("测试5：GBK 编码支持")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "sample_gbk.csv")],
        file_formats=["csv"],
        encoding="gbk"
    )
    result = loader({})

    assert_check(
        result["execute_result"] == True,
        f"✅ GBK 编码加载成功",
        f"❌ GBK 编码加载失败"
    )


def test_encoding_auto_detect():
    """测试6：编码自动检测"""
    log("=" * 50)
    log("测试6：编码自动检测（UTF-8）")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "sample_utf8.csv")],
        file_formats=["csv"],
        encoding="utf-8"
    )
    result = loader({})

    assert_check(
        result["execute_result"] == True,
        f"✅ UTF-8 自动检测成功",
        f"❌ UTF-8 自动检测失败"
    )


# ===== 3. 合并策略测试 =====

def test_merge_concat():
    """测试7：concat 合并策略"""
    log("=" * 50)
    log("测试7：concat 合并策略")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"),
            os.path.join(TEST_DATA_DIR, "sample_mixed.csv")
        ],
        file_formats=["csv", "csv"],
        merge_strategy="concat"
    )
    result = loader({})

    assert_check(
        result["count"] == 10,  # 5 + 5
        f"✅ concat 合并正确: {result['count']} 条",
        f"❌ concat 合并错误"
    )


def test_merge_union():
    """测试8：union 去重合并策略"""
    log("=" * 50)
    log("测试8：union 去重合并策略")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"),
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv")  # 重复文件
        ],
        file_formats=["csv", "csv"],
        merge_strategy="union"
    )
    result = loader({})

    assert_check(
        result["count"] == 5,  # 去重后应该还是 5
        f"✅ union 去重正确: {result['count']} 条（去重成功）",
        f"❌ union 去重错误: {result['count']} 条"
    )


# ===== 4. 容错与降级测试 =====

def test_file_not_exist():
    """测试9：文件不存在容错"""
    log("=" * 50)
    log("测试9：文件不存在容错")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=["D:/not_exist_file.csv"],
        file_formats=["csv"],
        max_errors=3,
        fail_fast=False
    )
    result = loader({})

    assert_check(
        result["execute_result"] == True,  # 容错模式下不是失败
        f"✅ 容错成功，整体仍返回成功",
        f"❌ 容错失败"
    )

    assert_check(
        result["error_count"] >= 1,
        f"✅ 错误被正确记录: {result['error_count']} 个错误",
        f"❌ 错误未被记录"
    )


def test_degraded_mode():
    """测试10：快速失败模式 + 降级策略"""
    log("=" * 50)
    log("测试10：快速失败模式 + 降级策略")
    log("=" * 50)

    # 测试 fail_fast=True 时，降级在循环中被触发
    loader = DataLoaderMapper(
        file_paths=[
            "bad1.csv",                                    # 失败: error_count=1
            "bad2.csv",                                    # 失败: error_count=2
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"), # 成功: error_count 重置
            "bad3.csv",                                    # 失败: error_count=1
            "bad4.csv",                                    # 失败: error_count=2
            "bad5.csv",                                    # 失败: error_count=3 → fail_fast 触发
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv")  # 这个不会执行
        ],
        file_formats=["csv", "csv", "csv", "csv", "csv", "csv", "csv"],
        max_errors=3,
        fail_fast=True
    )
    result = loader({})

    # fail_fast=True 时，遇到连续3个错误会停止处理
    assert_check(
        result["success_count"] <= 2,
        f"✅ fail_fast 生效: 只处理了 {result['success_count']} 个文件（遇到错误后停止）",
        f"❌ fail_fast 未生效"
    )

    # 验证错误被正确记录
    assert_check(
        result["error_count"] >= 2,
        f"✅ 错误正确记录: {result['error_count']} 个错误",
        f"❌ 错误记录错误"
    )

    log(f"   成功文件: {result['success_count']} 个")
    log(f"   错误文件: {result['error_count']} 个")
    log(f"   降级原因: {result.get('degraded_reason', 'N/A')}")


def test_fail_fast_mode():
    """测试11：快速失败模式"""
    log("=" * 50)
    log("测试11：快速失败模式")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[
            "bad1.csv",
            "bad2.csv",
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv")
        ],
        file_formats=["csv", "csv", "csv"],
        max_errors=1,
        fail_fast=True
    )
    result = loader({})

    # 快速失败应该在第1个错误后停止
    assert_check(
        result["success_count"] <= 1,
        f"✅ fail_fast 模式生效: 只处理了 {result['success_count']} 个文件",
        f"❌ fail_fast 模式未生效"
    )


# ===== 5. 分块处理测试 =====

def test_chunk_processing():
    """测试12：大文件分块处理"""
    log("=" * 50)
    log("测试12：大文件分块处理")
    log("=" * 50)

    start_time = time.time()

    loader = DataLoaderMapper(
        file_paths=[os.path.join(TEST_DATA_DIR, "large_file.csv")],
        file_formats=["csv"],
        chunk_size=2000  # 每块 2000 行
    )
    result = loader({})

    elapsed = (time.time() - start_time) * 1000

    assert_check(
        result["execute_result"] == True,
        f"✅ 大文件分块加载成功",
        f"❌ 大文件加载失败"
    )

    assert_check(
        result["count"] == 10000,
        f"✅ 分块加载数据完整: {result['count']} 条",
        f"❌ 数据不完整: {result['count']} 条"
    )

    assert_check(
        loader._total_rows_loaded == 10000,
        f"✅ 分块计数器正确: {loader._total_rows_loaded} 行",
        f"❌ 分块计数器错误"
    )

    log(f"   执行耗时: {result['execution_time_ms']}ms")
    log(f"   吞吐量: {result['count'] / (result['execution_time_ms']/1000):.0f} 条/秒")


# ===== 6. 质量评分测试 =====

def test_quality_score():
    """测试13：质量评分逻辑"""
    log("=" * 50)
    log("测试13：质量评分逻辑")
    log("=" * 50)

    loader = DataLoaderMapper(
        file_paths=[
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"),
            os.path.join(TEST_DATA_DIR, "sample.json")
        ],
        file_formats=["csv", "json"]
    )
    result = loader({})

    # 完全成功时质量评分应该 >= 0.7
    assert_check(
        result["quality_score"] >= 0.7,
        f"✅ 质量评分正常: {result['quality_score']}",
        f"❌ 质量评分过低: {result['quality_score']}"
    )

    # 评分应该在 0-1 之间
    assert_check(
        0 <= result["quality_score"] <= 1,
        f"✅ 评分范围正确: {result['quality_score']}",
        f"❌ 评分超出范围"
    )


# ===== 7. 边界测试 =====

def test_empty_path_list():
    """测试14：空路径列表边界"""
    log("=" * 50)
    log("测试14：空路径列表边界")
    log("=" * 50)

    try:
        loader = DataLoaderMapper(
            file_paths=[],
            file_formats=[]
        )
        log("❌ 应该抛出 ValueError", "FAIL")
        test_results["total"] += 1
        test_results["failed"] += 1
    except ValueError as e:
        assert_check(
            "file_paths 不能为空" in str(e),
            f"✅ 空路径正确抛出异常: ValueError",
            f"❌ 异常信息不正确"
        )


def test_format_detection():
    """测试15：格式自动检测"""
    log("=" * 50)
    log("测试15：格式自动检测")
    log("=" * 50)

    # 不指定格式，让算子自动检测
    loader = DataLoaderMapper(
        file_paths=[
            os.path.join(TEST_DATA_DIR, "sample_utf8.csv"),
            os.path.join(TEST_DATA_DIR, "sample.json")
        ],
        file_formats=[]  # 不指定
    )
    result = loader({})

    assert_check(
        result["success_count"] == 2,
        f"✅ 格式自动检测成功: {result['success_count']} 个文件",
        f"❌ 格式自动检测失败"
    )


# ==================== 测试执行入口 ====================

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 DataLoader 算子综合测试")
    print("=" * 60)
    print(f"📁 测试数据目录: {TEST_DATA_DIR}")
    print(f"📁 测试报告目录: {REPORT_DIR}")
    print("=" * 60)
    print()

    # 清空之前的测试结果
    global test_results
    test_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": 0,
        "passed": 0,
        "failed": 0,
        "details": []
    }

    # 执行所有测试
    test_functions = [
        # 功能测试
        ("功能正确性", [
            test_single_csv_load,
            test_json_load,
            test_single_json_object,
            test_batch_multi_format,
        ]),
        # 编码测试
        ("编码兼容性", [
            test_encoding_gbk,
            test_encoding_auto_detect,
        ]),
        # 合并策略
        ("合并策略", [
            test_merge_concat,
            test_merge_union,
        ]),
        # 容错降级
        ("容错与降级", [
            test_file_not_exist,
            test_degraded_mode,
            test_fail_fast_mode,
        ]),
        # 性能测试
        ("性能与分块", [
            test_chunk_processing,
        ]),
        # 质量评分
        ("质量评分", [
            test_quality_score,
        ]),
        # 边界测试
        ("边界测试", [
            test_empty_path_list,
            test_format_detection,
        ]),
    ]

    for category, tests in test_functions:
        print(f"\n{'─' * 50}")
        print(f"📦 {category}")
        print(f"{'─' * 50}")
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                log(f"测试异常: {str(e)}", "FAIL")
                test_results["total"] += 1
                test_results["failed"] += 1

    # 输出总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"总测试数: {test_results['total']}")
    print(f"通过: {test_results['passed']} ✅")
    print(f"失败: {test_results['failed']} ❌")

    if test_results['failed'] == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {test_results['failed']} 个测试失败，请检查！")

    # 保存报告
    report_path = save_report()
    print(f"\n📄 测试报告已保存: {report_path}")
    print("=" * 60)

    return test_results


if __name__ == "__main__":
    run_all_tests()
