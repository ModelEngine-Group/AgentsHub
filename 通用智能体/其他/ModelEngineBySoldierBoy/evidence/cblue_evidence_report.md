# CBLUE 数据证据报告

- 生成时间：2026-07-02T23:05:53+08:00
- 数据模式：`official_ready`
- official_ready：是
- source：`data/external/cblue`

## 阻塞项

- 未发现阻塞项。

## 官方归档文件

| 文件 | 找到 | 匹配数 |
| --- | --- | ---: |
| `CMeIE.zip/CMeIE-V2.zip` | 是 | 1 |
| `CMeEE.zip/CMeEE-V2.zip` | 是 | 1 |

## 解压后文件

| 文件 | 找到 | 字节 |
| --- | --- | ---: |
| `CMeIE_train.jsonl/CMeIE-V2_train.jsonl` | 是 | 15585410 |
| `CMeIE_dev.jsonl/CMeIE-V2_dev.jsonl` | 是 | 3903712 |
| `CMeEE_train.json/CMeEE-V2_train.json` | 是 | 13036448 |
| `CMeEE_dev.json/CMeEE-V2_dev.json` | 是 | 4189100 |

## 当前转换报告

| 数据集 | 文档数 | 记录数 | 输入 |
| --- | ---: | ---: | --- |
| CBLUE-CMeIE | 300 | 1224 | `data/external/cblue/CMeIE-V2/CMeIE-V2_train.jsonl`<br>`data/external/cblue/CMeIE-V2/CMeIE-V2_dev.jsonl` |
| CBLUE-CMeEE | 300 | 793 | `data/external/cblue/CMeEE-V2/CMeEE-V2_train.json`<br>`data/external/cblue/CMeEE-V2/CMeEE-V2_dev.json` |

## 官方数据重跑命令

```bash
bash scripts/run_final_acceptance.sh --cblue-source <official_train_20220729_dir_or_zip> --cblue-limit 300 --run-integration --knowledge-base-id <DataMate知识库ID>
```

说明：只有取得官方 CBLUE 原始包并且 `mode=official_ready` 后，才应对外宣称官方 CBLUE 验收通过。
