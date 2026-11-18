# 多人约练与场馆排程助手知识库

## 区间运算 GROUP-INTERVAL

空闲与占用使用左闭右开区间，重叠与相邻空闲先合并，再做多人及场馆交集，扣除已占用时段。start/end 统一为 UTC，display_time 按指定的 display_timezone 计算，默认 UTC；答复直接引用展示字段并保留时区。

## 日历与执行 GROUP-CALENDAR

ICS 是候选安排，STATUS 为 TENTATIVE，不代表完成预约。字段格式参考 https://www.rfc-editor.org/rfc/rfc5545 。price_cents 是本次时段总费用。场馆状态和费用以实际预约时再次核验为准。
