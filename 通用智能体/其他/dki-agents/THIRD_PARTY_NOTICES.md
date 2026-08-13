# Third-Party Notices

This project is released under the MIT License in `LICENSE`. The following
components are direct runtime, development, model, data, or vendored
dependencies. Their upstream licenses and terms continue to apply.

## Frameworks and runtime

| Component | Declared license | Use in this project |
|---|---|---|
| Nexent 0.1.2 | MIT | Agent framework and SDK |
| ModelEngine DataMate repository | MIT | Operator contracts and integration reference |
| DataMate `python-executor` package | Apache-2.0 | Optional operator runtime |
| PyYAML | MIT | YAML configuration |
| Jinja2 | BSD-3-Clause | HTML report templates |
| python-dotenv | BSD-3-Clause | Local environment configuration |
| pandas | BSD-3-Clause | Structured data processing |
| SQLGlot | MIT | SQL AST parsing and normalization |
| NumPy | BSD-3-Clause | Numeric arrays |
| OpenCV | Apache-2.0 | Optional DataMate dependency |
| Loguru | MIT | Optional DataMate logging |
| Unstructured | Apache-2.0 | Optional document parsing |
| SQLAlchemy | MIT | Optional DataMate persistence |
| Requests | Apache-2.0 | Optional HTTP client |
| Beautiful Soup | MIT | Optional HTML parsing |
| xmltodict | MIT | Optional XML parsing |
| datasketch | MIT | Optional similarity operations |

The sibling Nexent checkout contains its own `LICENSE` and `NOTICE`. The
DataMate repository root and its `python-executor` package declare different
licenses; consumers must retain the notice applicable to the exact component
they redistribute.

## Models and NPU toolchain

| Component | Declared license or terms | Use |
|---|---|---|
| PyTorch | BSD-3-Clause | Training and inference |
| Hugging Face Transformers | Apache-2.0 | Encoder and tokenizer loading |
| ONNX | Apache-2.0 | CMeEE export validation |
| Ascend CANN / pyACL / ATC | Huawei commercial/open-source distribution terms | Target NPU conversion and runtime |
| `bert-base-chinese` / `hfl/chinese-roberta-wwm-ext` | Upstream model-card terms; commonly Apache-2.0 | Base encoders |

Model weights, OM files, ONNX files, CANN binaries, and target-host libraries
are excluded from the submission bundle. Before redistributing a trained
checkpoint, verify the exact upstream model revision and its model-card
license.

## User interface

The vendored Apache ECharts JavaScript distribution is licensed under
Apache-2.0. Its local notice is retained at
`dki/analysis/static/ECHARTS_LICENSE.txt`.

## Data

CBLUE/CMeEE/CMeIE are third-party research datasets. Their records are not
redistributed in this repository or submission bundle. Access and use remain
subject to the dataset provider's terms. See `docs/cblue_datacard.md`.

## Development tools

pytest is MIT licensed. Jupyter and its core packages are BSD licensed.
Matplotlib uses the PSF-based Matplotlib license.

This notice is an inventory, not legal advice. Installed transitive
dependencies may add further notices; the lockfile or installed distribution
metadata for the final build remains authoritative.
