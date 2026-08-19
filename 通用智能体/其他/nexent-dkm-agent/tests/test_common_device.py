import sys
import types


def test_get_device_prefers_npu_when_torch_npu_is_available(monkeypatch):
    from src.common import device as device_utils

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = types.SimpleNamespace(npu=FakeNpu(), cuda=FakeCuda())

    def fake_import_module(name):
        if name == "torch_npu":
            return object()
        if name == "torch":
            return fake_torch
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(device_utils.importlib, "import_module", fake_import_module)

    spec = device_utils.get_device("auto")

    assert spec.kind == "npu"
    assert spec.device == "npu:0"
    assert spec.module == "torch_npu"


def test_get_device_falls_back_to_cuda_when_npu_is_unavailable(monkeypatch):
    from src.common import device as device_utils

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = types.SimpleNamespace(cuda=FakeCuda())

    def fake_import_module(name):
        if name == "torch_npu":
            raise ModuleNotFoundError(name)
        if name == "torch":
            return fake_torch
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(device_utils.importlib, "import_module", fake_import_module)

    spec = device_utils.get_device("auto")

    assert spec.kind == "cuda"
    assert spec.device == "cuda:0"
    assert spec.module == "torch"


def test_model_load_kwargs_and_move_for_npu():
    from src.common.device import DeviceSpec, model_load_kwargs, move_model_to_device

    spec = DeviceSpec(kind="npu", device="npu:0", module="torch_npu", reason="available")

    class FakeModel:
        moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    model = FakeModel()

    assert model_load_kwargs(spec) == {}
    assert move_model_to_device(model, spec) is model
    assert model.moved_to == "npu:0"


def test_kg_local_model_loader_uses_device_adapter_for_npu(monkeypatch, tmp_path):
    from src.common.device import DeviceSpec
    from src.operators.kg_ops import local_model_ner

    local_model_ner._MODEL_CACHE.clear()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class FakeModel:
        moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            return None

    class FakeAutoModel:
        last_kwargs = None
        last_model = None

        @staticmethod
        def from_pretrained(_model_path, **kwargs):
            FakeAutoModel.last_kwargs = kwargs
            FakeAutoModel.last_model = FakeModel()
            return FakeAutoModel.last_model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_path, **_kwargs):
            return object()

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        local_model_ner,
        "get_device",
        lambda: DeviceSpec(kind="npu", device="npu:0", module="torch_npu", reason="available"),
    )

    model, _tokenizer = local_model_ner._load_model(str(model_dir))

    assert model is FakeAutoModel.last_model
    assert FakeAutoModel.last_kwargs["trust_remote_code"] is True
    assert "device_map" not in FakeAutoModel.last_kwargs
    assert model.moved_to == "npu:0"
