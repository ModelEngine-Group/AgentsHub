# Open SQL 安全对抗评测

- 数据集版本：`1.0.0`
- 总样例：`120`
- 合法查询通过率：`100.00%`
- 危险查询阻断率：`100.00%`
- 总体安全判定正确率：`100.00%`
- sqlglot：`30.12.0`

## 分类结果

| 类别 | 总数 | 正确 | 通过率 |
| --- | ---: | ---: | ---: |
| bypass | 15 | 15 | 100.00% |
| dangerous_function | 15 | 15 | 100.00% |
| illegal_join | 15 | 15 | 100.00% |
| legal | 40 | 40 | 100.00% |
| non_whitelist_field | 15 | 15 | 100.00% |
| write_ddl | 20 | 20 | 100.00% |

## 失败案例

无。

危险 SQL 只进入 AST Guard，不送入 SQLite。合法 SQL 通过 Guard 后，仍必须通过只读 URI、query_only、SQLite authorizer、progress handler、行数和时间限制。
