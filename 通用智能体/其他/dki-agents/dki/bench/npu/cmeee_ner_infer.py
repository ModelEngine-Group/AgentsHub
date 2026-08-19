"""CMeEE-V2 GlobalPointer NER 推理后端（供 MedicalEntityExtractor backend=cmeee）。

加载 scripts/train_ner_cmeee.py 的产物（RoBERTa-wwm-ext + GlobalPointer），对中文文本
抽取带类型实体（字符级 span）。9 类 CMeEE 标签映射到本项目 6 类 schema。

torch/transformers 惰性导入（仅在 NPU/GPU/CPU 有 torch 的机器上用）；本模块可被无 torch
的机器 import（供打包/契约测试），只有实例化 backend 时才拉起 torch。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CMEEE_DIR = _PROJECT_ROOT / "models" / "medical_ner_cmeee"

# CMeEE 9 类 → 本项目 6 类（bod/equ/mic 不在 6 类 schema，丢弃）
CMEEE_TO_PROJECT = {
    "dis": "Disease", "sym": "Symptom", "dru": "Drug",
    "ite": "Exam", "pro": "Treatment", "dep": "Department",
}
CMEEE_TYPES = ["dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"]


class CmeeeGlobalPointerBackend:
    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu",
                 max_len: Optional[int] = None, inner_dim: Optional[int] = None,
                 map_to_project: bool = True, threshold: Optional[float] = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_CMEEE_DIR
        self.device_str = device
        self.max_len = max_len
        self.inner_dim = inner_dim
        self.threshold = threshold
        self.map_to_project = map_to_project
        self._model = None
        self._tok = None
        self._device = None
        self._types = list(CMEEE_TYPES)
        self._softlex_dim = 0
        self._softlex_lexicon = None

    def _lazy_init(self):
        if self._model is not None:
            return
        import json as _json

        import torch
        if self.device_str == "npu":
            import torch_npu  # noqa: F401
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        from dki.nlp.globalpointer import EncoderGlobalPointer

        types_file = self.model_dir / "types.json"
        if types_file.is_file():
            self._types = _json.loads(types_file.read_text(encoding="utf-8"))
        config_file = self.model_dir / "globalpointer_config.json"
        deployment_config = (
            _json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
        )
        max_len = int(self.max_len or deployment_config.get("max_len", 256))
        inner_dim = int(self.inner_dim or deployment_config.get("inner_dim", 64))
        threshold = float(
            self.threshold if self.threshold is not None
            else deployment_config.get("selected_threshold", 0.0)
        )
        self.max_len, self.inner_dim, self.threshold = max_len, inner_dim, threshold
        self._device = torch.device(self.device_str if self.device_str != "npu" else "npu")
        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        encoder_config = self.model_dir / "config.json"
        if not encoder_config.is_file():
            raise RuntimeError(
                "legacy CMeEE artifact has no self-contained encoder config/weights; "
                "retrain with the current train_ner_cmeee.py or migrate it with the recorded base model"
            )
        encoder = AutoModel.from_pretrained(str(self.model_dir))
        pointer = str(deployment_config.get("pointer", "classic"))
        softlex_dim = int(deployment_config.get("softlex_dim", 0) or 0)
        softlex_on = bool(deployment_config.get("softlexicon", False)) or softlex_dim > 0
        use_bilstm = bool(deployment_config.get("use_bilstm", False))
        model = EncoderGlobalPointer(
            encoder, encoder.config.hidden_size, len(self._types), inner_dim,
            pointer=pointer,
            softlex_dim=softlex_dim if softlex_on else 0,
            softlex_mode=str(deployment_config.get("softlex_mode") or "gated"),
            use_bilstm=use_bilstm,
        )
        state = torch.load(self.model_dir / "globalpointer.pt", map_location="cpu")
        # schema v1 stores only GP head; legacy artifacts stored the whole wrapper.
        if state and all(key.startswith("gp.") or key.startswith("encoder.") for key in state):
            model.load_state_dict(state)
        else:
            model.gp.load_state_dict(state)
        if model.bilstm is not None and (self.model_dir / "bilstm.pt").is_file():
            model.bilstm.load_state_dict(torch.load(self.model_dir / "bilstm.pt", map_location="cpu"))
        if model.softlex_proj is not None and (self.model_dir / "softlexicon_proj.pt").is_file():
            model.softlex_proj.load_state_dict(
                torch.load(self.model_dir / "softlexicon_proj.pt", map_location="cpu")
            )
            if model.softlex_gate is not None and (self.model_dir / "softlexicon_gate.pt").is_file():
                gate = torch.load(self.model_dir / "softlexicon_gate.pt", map_location="cpu")
                model.softlex_gate.data.copy_(gate["gate"].reshape_as(model.softlex_gate.data))
            surfaces = _json.loads((self.model_dir / "softlexicon_surfaces.json").read_text(encoding="utf-8"))
            self._softlex_lexicon = {k: set(v) for k, v in surfaces.items()}
        else:
            self._softlex_lexicon = None
        self._model = model.to(self._device).eval()
        self._softlex_dim = softlex_dim if softlex_on else 0

    def extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        self._lazy_init()
        import numpy as np
        import torch

        from dki.nlp.globalpointer import decode_spans

        enc = self._tok(texts, truncation=True, max_length=self.max_len, padding="max_length",
                        return_offsets_mapping=True, return_tensors="np")
        offsets = enc["offset_mapping"]
        ids = torch.tensor(enc["input_ids"]).to(self._device)
        mask = torch.tensor(enc["attention_mask"]).to(self._device)
        tt = torch.tensor(enc.get("token_type_ids", np.zeros_like(enc["input_ids"]))).to(self._device)
        soft_kw = {}
        if getattr(self, "_softlex_dim", 0) > 0 and getattr(self, "_softlex_lexicon", None):
            from dki.nlp.softlexicon import encode_softlex_batch
            rows = [{"text": t, "entities": []} for t in texts]
            soft = encode_softlex_batch(
                rows, offsets, self._softlex_lexicon, len(self._types), self.max_len,
            )
            soft_kw["softlex"] = torch.tensor(soft).to(self._device)
        out: List[List[Dict[str, Any]]] = []
        with torch.no_grad():
            logits = self._model(ids, mask, tt, **soft_kw)
        logits_cpu = logits.float().cpu()
        pred = decode_spans(logits_cpu, threshold=self.threshold)
        for i, spans in enumerate(pred):
            off = offsets[i]
            ents = []
            for (h, ts, te) in spans:
                ctype = self._types[h]
                if self.map_to_project:
                    ptype = CMEEE_TO_PROJECT.get(ctype)
                    if ptype is None:
                        continue
                else:
                    ptype = ctype
                a = int(off[ts][0]); b = int(off[te][1])
                if b <= a:
                    continue
                conf = float(torch.sigmoid(logits_cpu[i, h, ts, te]).item())
                ents.append({"mention": texts[i][a:b], "type": ptype, "span": [a, b],
                             "cmeee_type": ctype, "confidence": conf,
                             "source": "cmeee_gp", "evidence": "model_span"})
            out.append(ents)
        return out


class CmeeeGlobalPointerOmBackend:
    """Schema-v3, text-to-entity backend for a static GlobalPointer OM."""

    def __init__(
        self,
        *,
        model_dir: str,
        om_path: str,
        manifest_path: str,
        export_receipt_path: str,
        atc_log_path: str,
        device: int = 0,
        map_to_project: bool = True,
        threshold: Optional[float] = None,
    ):
        self.model_dir = Path(model_dir)
        self.om_path = Path(om_path)
        self.manifest_path = Path(manifest_path)
        self.export_receipt_path = Path(export_receipt_path)
        self.atc_log_path = Path(atc_log_path)
        self.device = int(device)
        self.map_to_project = bool(map_to_project)
        self.threshold = threshold
        self._tok = None
        self._runner = None
        self._types = list(CMEEE_TYPES)
        self._config: Dict[str, Any] = {}
        self._softlex_lexicon = None

    def _lazy_init(self) -> None:
        if self._runner is not None:
            return
        import json as _json

        from transformers import AutoTokenizer

        from dki.bench.npu.om_manifest import enforce_preload_manifest
        from dki.bench.npu.static_om_runner import StaticOmRunner

        self._config = _json.loads(
            (self.model_dir / "globalpointer_config.json").read_text(encoding="utf-8")
        )
        self._types = _json.loads(
            (self.model_dir / "types.json").read_text(encoding="utf-8")
        )
        if self._types != CMEEE_TYPES:
            raise RuntimeError("CMeEE OM labels do not match the nine-type contract")
        self.threshold = float(
            self.threshold
            if self.threshold is not None
            else self._config.get("selected_threshold", 0.0)
        )
        manifest = enforce_preload_manifest({
            "manifestPath": str(self.manifest_path),
            "omPath": str(self.om_path),
            "tokenizerPath": str(self.model_dir / "tokenizer.json"),
            "labelsPath": str(self.model_dir / "types.json"),
            "checkpointPath": str(self.model_dir),
            "exportReceiptPath": str(self.export_receipt_path),
            "atcLogPath": str(self.atc_log_path),
        })
        if (
            manifest is None
            or manifest.schema_version != 3
            or manifest.model_family != "cmeee_globalpointer"
        ):
            raise RuntimeError("CMeEE OM requires a schema-v3 GlobalPointer manifest")

        self._tok = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        if self._config.get("softlexicon"):
            surfaces_path = self.model_dir / "softlexicon_surfaces.json"
            surfaces = _json.loads(surfaces_path.read_text(encoding="utf-8"))
            self._softlex_lexicon = {
                surface: set(type_ids) for surface, type_ids in surfaces.items()
            }
        runner = StaticOmRunner(self.om_path, self.device)
        try:
            manifest.verify_io(
                inputs=runner.input_specs,
                outputs=runner.output_specs,
            )
            if len(runner.output_specs) != 1:
                raise RuntimeError("CMeEE OM must expose exactly one logits output")
            shape = runner.output_specs[0].shape
            max_len = int(self._config["max_len"])
            if (
                len(shape) != 4
                or shape[1] != len(self._types)
                or shape[2:] != (max_len, max_len)
            ):
                raise RuntimeError(f"unexpected CMeEE GlobalPointer OM output: {shape}")
        except BaseException:
            runner.close()
            raise
        self._runner = runner

    @staticmethod
    def _sigmoid(value: float) -> float:
        import math

        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _arrays(self, texts: List[str]):
        import numpy as np

        max_len = int(self._config["max_len"])
        encoded = self._tok(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="np",
        )
        values = {
            "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
            "token_type_ids": np.asarray(
                encoded.get(
                    "token_type_ids",
                    np.zeros_like(encoded["input_ids"], dtype=np.int64),
                ),
                dtype=np.int64,
            ),
        }
        if any(spec.name == "softlex" for spec in self._runner.input_specs):
            if self._softlex_lexicon is None:
                raise RuntimeError("OM expects softlex but checkpoint surfaces are missing")
            from dki.nlp.softlexicon import encode_softlex_batch

            rows = [{"text": text, "entities": []} for text in texts]
            values["softlex"] = np.asarray(
                encode_softlex_batch(
                    rows,
                    encoded["offset_mapping"],
                    self._softlex_lexicon,
                    len(self._types),
                    max_len,
                ),
                dtype=np.float32,
            )
        unknown = [spec.name for spec in self._runner.input_specs if spec.name not in values]
        if unknown:
            raise RuntimeError(f"unsupported CMeEE OM input tensor(s): {unknown}")
        return (
            {spec.name: values[spec.name] for spec in self._runner.input_specs},
            encoded["offset_mapping"],
        )

    def extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        if not texts:
            return []
        self._lazy_init()
        import numpy as np

        static_batch = int(self._runner.input_specs[0].shape[0])
        output_shape = self._runner.output_specs[0].shape
        results: List[List[Dict[str, Any]]] = []
        for start in range(0, len(texts), static_batch):
            actual = texts[start:start + static_batch]
            padded = actual + [""] * (static_batch - len(actual))
            arrays, offsets = self._arrays(padded)
            logits = self._runner.execute(arrays)[0].reshape(output_shape)
            for index, text in enumerate(actual):
                entities = []
                coordinates = np.argwhere(logits[index] > float(self.threshold))
                for head, token_start, token_end in coordinates.tolist():
                    cmeee_type = self._types[int(head)]
                    project_type = (
                        CMEEE_TO_PROJECT.get(cmeee_type)
                        if self.map_to_project
                        else cmeee_type
                    )
                    if project_type is None:
                        continue
                    char_start = int(offsets[index][token_start][0])
                    char_end = int(offsets[index][token_end][1])
                    if char_end <= char_start:
                        continue
                    score = float(logits[index, head, token_start, token_end])
                    mention = text[char_start:char_end]
                    entities.append({
                        "name": mention,
                        "mention": mention,
                        "type": project_type,
                        "span": [char_start, char_end],
                        "cmeee_type": cmeee_type,
                        "confidence": self._sigmoid(score),
                        "source": "cmeee_gp_om",
                        "evidence": "model_span",
                    })
                results.append(entities)
        return results

    def execute(self, text: str) -> List[Dict[str, Any]]:
        return self.extract([text])[0]

    def execute_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        return self.extract(texts)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


class _OnnxRuntimeRunner:
    def __init__(self, path: Path, intra_op_threads: int = 0):
        import onnxruntime as ort

        from dki.bench.npu.om_manifest import TensorSpec

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = int(intra_op_threads)
        self._session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        def tensor_spec(value):
            shape = tuple(int(dimension) for dimension in value.shape)
            dtype = {
                "tensor(int64)": "INT64",
                "tensor(float)": "FP32",
                "tensor(float16)": "FP16",
            }.get(value.type)
            if dtype is None:
                raise RuntimeError(f"unsupported ONNX tensor type: {value.type}")
            return TensorSpec(value.name, dtype, shape)

        self.input_specs = tuple(
            tensor_spec(value) for value in self._session.get_inputs()
        )
        self.output_specs = tuple(
            tensor_spec(value) for value in self._session.get_outputs()
        )

    def execute(self, values):
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is closed")
        return self._session.run(
            None,
            {spec.name: values[spec.name] for spec in self.input_specs},
        )

    def close(self) -> None:
        self._session = None


class CmeeeGlobalPointerOnnxBackend(CmeeeGlobalPointerOmBackend):
    """CPU ONNX reference using the same preprocessing and decode as the OM."""

    def __init__(
        self,
        *,
        model_dir: str,
        onnx_path: str,
        map_to_project: bool = True,
        threshold: Optional[float] = None,
        intra_op_threads: int = 0,
    ):
        self.model_dir = Path(model_dir)
        self.onnx_path = Path(onnx_path)
        self.map_to_project = bool(map_to_project)
        self.threshold = threshold
        self.intra_op_threads = int(intra_op_threads)
        self._tok = None
        self._runner = None
        self._types = list(CMEEE_TYPES)
        self._config: Dict[str, Any] = {}
        self._softlex_lexicon = None

    def _lazy_init(self) -> None:
        if self._runner is not None:
            return
        import json as _json

        from transformers import AutoTokenizer

        self._config = _json.loads(
            (self.model_dir / "globalpointer_config.json").read_text(
                encoding="utf-8"
            )
        )
        self._types = _json.loads(
            (self.model_dir / "types.json").read_text(encoding="utf-8")
        )
        if self._types != CMEEE_TYPES:
            raise RuntimeError("CMeEE ONNX labels do not match the nine-type contract")
        self.threshold = float(
            self.threshold
            if self.threshold is not None
            else self._config.get("selected_threshold", 0.0)
        )
        self._tok = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        if self._config.get("softlexicon"):
            surfaces = _json.loads(
                (self.model_dir / "softlexicon_surfaces.json").read_text(
                    encoding="utf-8"
                )
            )
            self._softlex_lexicon = {
                surface: set(type_ids)
                for surface, type_ids in surfaces.items()
            }
        runner = _OnnxRuntimeRunner(
            self.onnx_path,
            intra_op_threads=self.intra_op_threads,
        )
        max_len = int(self._config["max_len"])
        if (
            len(runner.output_specs) != 1
            or runner.output_specs[0].shape[1:]
            != (len(self._types), max_len, max_len)
        ):
            runner.close()
            raise RuntimeError(
                f"unexpected CMeEE GlobalPointer ONNX output: "
                f"{runner.output_specs}"
            )
        self._runner = runner
