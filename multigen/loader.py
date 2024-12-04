from typing import Type, List, Union, Optional, Any
from dataclasses import dataclass
import random
import copy as cp
from contextlib import nullcontext
import torch
import logging
import threading
import psutil
import sys
import diffusers

from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline

from .util import get_model_size, awailable_ram, quantize, weightshare_copy


logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ModelDescriptor:
    """
    Descriptor class for model identification that includes quantization information
    """
    model_id: str
    quantize_dtype: Optional[Any] = None

    def __hash__(self):
        return hash((self.model_id, str(self.quantize_dtype)))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.model_id == other

        if not isinstance(other, ModelDescriptor):
            return False
        return (self.model_id == other.model_id and
                self.quantize_dtype == other.quantize_dtype)


class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cpu_pipes = dict()  # ModelDescriptor -> pipe
        self._gpu_pipes = dict()  # gpu idx -> list of (ModelDescriptor, pipe) pairs

    def get_gpu(self, model_id) -> List[int]:
        """
        return list of gpus with loaded model
        """
        with self._lock:
            result = list()
            for idx, items in self._gpu_pipes.items():
                for (model, _) in items:
                    if model == model_id:
                        result.append(idx)
            return result

    def load_pipeline(self, cls: Type[DiffusionPipeline], path, torch_dtype=torch.bfloat16,
                      device=None, offload_device=None, quantize_dtype=None, **additional_args):
        with self._lock:
            logger.debug(f'looking for pipeline {cls} from {path} on {device}')
            result = None
            descriptor = ModelDescriptor(path, quantize_dtype)
            found_quantized = False
            if device is None:
                device = torch.device('cpu', 0)
            if device.type == 'cuda':
                idx = device.index
                gpu_pipes = self._gpu_pipes.get(idx, [])
                for (key, value) in gpu_pipes:
                    if key == descriptor:
                        logger.debug(f'found pipe in gpu cache {key}')
                        result = self.from_pipe(cls, value, additional_args)
                        logger.debug(f'created pipe from gpu cache {key} on {device}')
                        return result
            for (key, pipe) in self._cpu_pipes.items():
                if key == descriptor:
                    found_quantized = True
                    logger.debug(f'found pipe in cpu cache {key} {pipe.device}')
                    if device.type == 'cuda':
                        pipe = cp.deepcopy(pipe)
                    result = self.from_pipe(cls, pipe, additional_args)
                    break
            if result is None:
                logger.info(f'not found {path} in cache, loading')
                if path.endswith('safetensors'):
                    result = cls.from_single_file(path, torch_dtype=torch_dtype, **additional_args)
                else:
                    result = cls.from_pretrained(path, torch_dtype=torch_dtype, **additional_args)
                logger.debug(f'loaded pipe {path} dtype {result.dtype}')
            if device.type == 'cuda':
                self.clear_cache(device)

            logger.debug("prepare pipe before returning from loader")
            logger.debug(f"{path} on {result.device} {result.dtype}")

            # Add quantization if specified
            if (not found_quantized) and quantize_dtype is not None:
                logger.debug(f'Quantizing pipeline to {quantize_dtype}')
                quantize(result, dtype=quantize_dtype)

            if result.device != device:
                logger.debug(f"move pipe to {device}")
                result = result.to(dtype=torch_dtype, device=device)
            if result.dtype != torch_dtype:
                result = result.to(dtype=torch_dtype)

            self.cache_pipeline(result, path)
            logger.debug(f'result device before weightshare_copy {result.device}')
            result = weightshare_copy(result)
            logger.debug(f'result device after weightshare_copy {result.device}')
            assert result.device.type == device.type
            if device.type == 'cuda':
                assert result.device.index == device.index
            logger.debug(f'returning {type(result)} from {path} \
                         on {result.device} scheduler {id(result.scheduler)}')
            return result

    def from_pipe(self, cls, pipe, additional_args):
        components = pipe.components
        if issubclass(cls, StableDiffusionXLControlNetPipeline) or issubclass(cls, StableDiffusionControlNetPipeline):
            # todo: keep controlnets in cache explicitly
            if 'controlnet' in additional_args:
                components.pop('controlnet')
            return cls(**components, **additional_args)
        # handling the case when the model in cache has controlnet in it
        # but we don't need it
        if 'controlnet' in components:
            components.pop('controlnet')
        return cls(**components, **additional_args)

    def cache_pipeline(self, pipe: DiffusionPipeline,  descriptor: ModelDescriptor):
        logger.debug(f'caching pipeline {descriptor} {pipe}')
        with self._lock:
            device = pipe.device
            if descriptor not in self._cpu_pipes:
                # deepcopy is needed since Module.to is an inplace operation
                size = get_model_size(pipe)
                ram = awailable_ram()
                logger.debug(f'{descriptor} has size {size}, ram {ram}')
                if ram < size * 2.5 and self._cpu_pipes:
                    key_to_delete = random.choice(list(self._cpu_pipes.keys()))
                    self._cpu_pipes.pop(key_to_delete)
                item = pipe
                if pipe.device.type == 'cuda':
                    device = pipe.device
                    logger.debug("deepcopy pipe from gpu to save it in cpu cache")
                    item = cp.deepcopy(pipe.to('cpu'))
                    pipe.to(device)
                self._cpu_pipes[descriptor] = item
                logger.debug(f'storing {descriptor} on cpu')
            assert pipe.device == device
            if pipe.device.type == 'cuda':
                self._store_gpu_pipe(pipe, descriptor)
                logger.debug(f'storing {descriptor} on {pipe.device}')

    def clear_cache(self, device):
        logger.debug(f'clear_cache pipelines from {device}')
        with self._lock:
            if device.type == 'cuda':
                self._gpu_pipes[device.index] = []

    def _store_gpu_pipe(self, pipe,  descriptor: ModelDescriptor):
        idx = pipe.device.index
        assert idx is not None
        # for now just clear all other pipelines
        self._gpu_pipes[idx] = [(descriptor, pipe)]

    def remove_pipeline(self, model_id):
        self._cpu_pipes.pop(model_id)

    def get_pipeline(self, descriptor: Union[ModelDescriptor, str], device=None):
        with self._lock:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)
            if device.type == 'cuda':
                idx = device.index
                gpu_pipes = self._gpu_pipes.get(idx, ())
                for (key, value) in gpu_pipes:
                    if key == descriptor:
                        return value
            for (key, pipe) in self._cpu_pipes.items():
                if key == descriptor:
                    return pipe

            return None
