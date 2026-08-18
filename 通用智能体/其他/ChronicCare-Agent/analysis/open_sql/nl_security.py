"""Natural-language preflight for explicit SQL injection and data-exfiltration requests."""
from __future__ import annotations

import re

PATTERNS = (
    r"\b(drop|delete|update|insert|alter|create|attach|pragma|vacuum)\b",
    r"load_extension|readfile|sqlite_master|/etc/passwd|api\s*key|环境变量|系统文件|动态库",
    r"删除.*表|写入|修改.*数据库|关闭.*query_only|无限递归|笛卡尔积|on\s+1\s*=\s*1",
    r"绕过.*白名单|secret_column|执行两条语句|触发器|shell\s*函数|导出数据库",
    r"读取.*(?:系统|用户).*(?:文件|目录)|(?:系统|用户).*文件",
    r"(?:把|将).*(?:表|数据库|记录).*(?:更新|修改|设置)|(?:更新|修改|设置).*(?:表|数据库)",
)


def classify_nl_security(question: str) -> dict:
    text = str(question or "").lower()
    matched = [pattern for pattern in PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]
    return {
        "safe": not matched,
        "code": None if not matched else "NL_SECURITY_POLICY_REJECTED",
        "reason": None if not matched else "请求包含写操作、系统访问、绕过或资源滥用意图，未进入 SQL 生成阶段。",
        "matched_rule_count": len(matched),
    }
