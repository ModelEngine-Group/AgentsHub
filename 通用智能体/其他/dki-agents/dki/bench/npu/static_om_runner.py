"""Process-owned static-shape ACL OM runner used by production adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dki.bench.npu.om_manifest import TensorSpec

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

_ACL_DTYPES = {
    0: (np.float32, "float32"),
    1: (np.float16, "float16"),
    3: (np.int32, "int32"),
    9: (np.int64, "int64"),
    11: (np.float64, "float64"),
    12: (np.bool_, "bool"),
}


def _check(ret, action: str) -> None:
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError(f"{action} failed, ret={ret}")


def _enum(value) -> int:
    return int(value[0] if isinstance(value, tuple) else value)


def _dtype(value, label: str):
    enum = _enum(value)
    if enum not in _ACL_DTYPES:
        raise RuntimeError(f"unsupported ACL {label} dtype enum {enum}")
    return _ACL_DTYPES[enum]


def _dims(acl, desc, index: int, *, output: bool) -> Tuple[int, ...]:
    getter = acl.mdl.get_output_dims if output else acl.mdl.get_input_dims
    value, ret = getter(desc, index)
    _check(ret, f"acl.mdl.get_{'output' if output else 'input'}_dims")
    return tuple(int(item) for item in value["dims"])


def _input_name(raw_name: str) -> str:
    name = raw_name.split(":")[0]
    for key in ("input_ids", "attention_mask", "token_type_ids", "softlex"):
        if key in name:
            return key
    return name


class StaticOmRunner:
    """Own one ACL context/model and expose its exact loaded IO contract."""

    def __init__(self, model_path: Path, device_id: int = 0):
        import acl

        self.acl = acl
        self.device_id = device_id
        self.input_ptrs: Dict[str, int] = {}
        self.input_nbytes: Dict[str, int] = {}
        self.output_ptrs: List[int] = []
        self.output_nbytes: List[int] = []
        self.input_buffers = []
        self.output_buffers = []
        self.input_specs: tuple[TensorSpec, ...] = ()
        self.output_specs: tuple[TensorSpec, ...] = ()
        self.context = self.model_id = self.desc = None
        self.input_dataset = self.output_dataset = None
        self._device_set = False
        self._closed = False
        try:
            _check(acl.init(), "acl.init")
            _check(acl.rt.set_device(device_id), "acl.rt.set_device")
            self._device_set = True
            self.context, ret = acl.rt.create_context(device_id)
            _check(ret, "acl.rt.create_context")
            self.model_id, ret = acl.mdl.load_from_file(str(model_path))
            _check(ret, "acl.mdl.load_from_file")
            self.desc = acl.mdl.create_desc()
            _check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")
            self.input_dataset = acl.mdl.create_dataset()
            self.output_dataset = acl.mdl.create_dataset()

            inputs = []
            for index in range(acl.mdl.get_num_inputs(self.desc)):
                name = _input_name(acl.mdl.get_input_name_by_index(self.desc, index))
                if name in self.input_ptrs:
                    raise RuntimeError(f"duplicate normalized ACL input name: {name}")
                shape = _dims(acl, self.desc, index, output=False)
                _, dtype_name = _dtype(acl.mdl.get_input_data_type(self.desc, index), "input")
                inputs.append(TensorSpec(name, dtype_name, shape))
                size = acl.mdl.get_input_size_by_index(self.desc, index)
                ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, f"acl.rt.malloc(input:{name})")
                self.input_ptrs[name], self.input_nbytes[name] = ptr, size
                buffer = acl.create_data_buffer(ptr, size)
                self.input_buffers.append(buffer)
                _check(acl.mdl.add_dataset_buffer(self.input_dataset, buffer), "add input buffer")

            outputs = []
            for index in range(acl.mdl.get_num_outputs(self.desc)):
                name = str(acl.mdl.get_output_name_by_index(self.desc, index)).split(":")[0]
                shape = _dims(acl, self.desc, index, output=True)
                numpy_dtype, dtype_name = _dtype(
                    acl.mdl.get_output_data_type(self.desc, index), "output"
                )
                outputs.append(TensorSpec(name, dtype_name, shape))
                size = acl.mdl.get_output_size_by_index(self.desc, index)
                ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, f"acl.rt.malloc(output:{name})")
                self.output_ptrs.append(ptr)
                self.output_nbytes.append(size)
                buffer = acl.create_data_buffer(ptr, size)
                self.output_buffers.append(buffer)
                _check(acl.mdl.add_dataset_buffer(self.output_dataset, buffer), "add output buffer")
                if index == 0:
                    self.output_shape = shape
                    self.output_dtype = numpy_dtype
            self.input_specs, self.output_specs = tuple(inputs), tuple(outputs)
        except BaseException:
            self.close()
            raise

    def bind_thread(self) -> None:
        _check(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    def execute(self, arrays: Dict[str, np.ndarray], output_dtype=None) -> List[np.ndarray]:
        acl = self.acl
        output_dtype = self.output_dtype if output_dtype is None else output_dtype
        for name, ptr in self.input_ptrs.items():
            array = np.ascontiguousarray(arrays[name])
            if array.nbytes != self.input_nbytes[name]:
                raise ValueError(f"{name} has {array.nbytes} bytes, expected {self.input_nbytes[name]}")
            _check(acl.rt.memcpy(ptr, array.nbytes, acl.util.numpy_to_ptr(array),
                                 array.nbytes, ACL_MEMCPY_HOST_TO_DEVICE), f"memcpy H2D {name}")
        _check(acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "acl.mdl.execute")
        outputs = []
        for ptr, size in zip(self.output_ptrs, self.output_nbytes):
            host = np.empty(size // np.dtype(output_dtype).itemsize, dtype=output_dtype)
            _check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), size, ptr, size,
                                 ACL_MEMCPY_DEVICE_TO_HOST), "memcpy D2H output")
            outputs.append(host)
        return outputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acl = self.acl
        for buffer in reversed(self.input_buffers + self.output_buffers):
            try:
                acl.destroy_data_buffer(buffer)
            except BaseException:
                pass
        for dataset in (self.output_dataset, self.input_dataset):
            if dataset is not None:
                try:
                    acl.mdl.destroy_dataset(dataset)
                except BaseException:
                    pass
        for ptr in reversed(list(self.input_ptrs.values()) + self.output_ptrs):
            try:
                acl.rt.free(ptr)
            except BaseException:
                pass
        if self.desc is not None:
            try:
                acl.mdl.destroy_desc(self.desc)
            except BaseException:
                pass
        if self.model_id is not None:
            try:
                acl.mdl.unload(self.model_id)
            except BaseException:
                pass
        if self.context is not None:
            try:
                acl.rt.destroy_context(self.context)
            except BaseException:
                pass
        if self._device_set:
            try:
                acl.rt.reset_device(self.device_id)
            except BaseException:
                pass
