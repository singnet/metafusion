from typing import Type, List
import random
import copy
from contextlib import nullcontext
import torch
import logging
import threading
import psutil
import sys
import diffusers

from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
else:
    init_empty_weights = nullcontext


logger = logging.getLogger(__file__)


def weightshare_copy(pipe):
    """
    Create a new pipe object then assign weights using load_state_dict from passed 'pipe'
    """
    copy = pipe.__class__(**pipe.components)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        for key, component in copy.components.items():
            if key in ('tokenizer', 'tokenizer_2', 'feature_extractor'):
                continue
            if getattr(copy, key) is None:
                continue
            cls = getattr(copy, key).__class__
            if hasattr(cls, 'from_config'):
                setattr(copy, key, cls.from_config(getattr(copy, key).config))
            else:
                setattr(copy, key, cls(getattr(copy, key).config))
    # assign=True is needed since our copy is on "meta" device, i.g. weights are empty
    for key, component in copy.components.items():
        if key == 'tokenizer' or key == 'tokenizer_2':
            continue
        obj = getattr(copy, key)
        if hasattr(obj, 'load_state_dict'):
            obj.load_state_dict(getattr(pipe, key).state_dict(), assign=True)
    return copy


class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cpu_pipes = dict()
        # idx -> list of (model_id, pipe) pairs
        self._gpu_pipes = dict()

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
                      device=None, offload_device=None, **additional_args):
        with self._lock:
            logger.debug(f'looking for pipeline {cls} from {path} on {device}')
            result = None
            if device is None:
                device = torch.device('cpu', 0)
            if device.type == 'cuda':
                idx = device.index
                gpu_pipes = self._gpu_pipes.get(idx, [])
                for (key, value) in gpu_pipes:
                    if key == path:
                        logger.debug(f'found pipe in gpu cache {key}')
                        result = self.from_pipe(cls, value, additional_args)
                        logger.debug(f'created pipe from gpu cache {key} on {device}')
                        return result
            for (key, pipe) in self._cpu_pipes.items():
                if key == path:
                    logger.debug(f'found pipe in cpu cache {key} {pipe.device}')
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

            if result.device != device:
                result = result.to(dtype=torch_dtype, device=device)
            if result.dtype != torch_dtype:
                result = result.to(dtype=torch_dtype)
            self.cache_pipeline(result, path)
            result = weightshare_copy(result)
            assert result.device.type == device.type
            if device.type == 'cuda':
                assert result.device.index == device.index
            logger.debug(f'returning {type(result)} from {path} on {result.device}')
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

    def cache_pipeline(self, pipe: DiffusionPipeline, model_id):
        with self._lock:
            device = pipe.device
            if model_id not in self._cpu_pipes:
                # deepcopy is needed since Module.to is an inplace operation
                size = get_model_size(pipe)
                ram = awailable_ram()
                logger.debug(f'{model_id} has size {size}, ram {ram}')
                if ram < size * 2.5 and self._cpu_pipes:
                    key_to_delete = random.choice(list(self._cpu_pipes.keys()))
                    self._cpu_pipes.pop(key_to_delete)
                item = pipe
                if pipe.device.type == 'cuda':
                    item = copy.deepcopy(pipe).to('cpu')
                self._cpu_pipes[model_id] = item
                logger.debug(f'storing {model_id} on cpu')
            assert pipe.device == device
            if pipe.device.type == 'cuda':
                self._store_gpu_pipe(pipe, model_id)
                logger.debug(f'storing {model_id} on {pipe.device}')

    def clear_cache(self, device):
        logger.debug(f'clear_cache pipelines from {device}')
        with self._lock:
            if device.type == 'cuda':
                self._gpu_pipes[device.index] = []

    def _store_gpu_pipe(self, pipe, model_id):
        idx = pipe.device.index
        assert idx is not None
        # for now just clear all other pipelines
        self._gpu_pipes[idx] = [(model_id, pipe)]

    def remove_pipeline(self, model_id):
        self._cpu_pipes.pop(model_id)

    def get_pipeline(self, model_id, device=None):
        with self._lock:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)
            if device.type == 'cuda':
                idx = device.index
                gpu_pipes = self._gpu_pipes.get(idx, ())
                for (key, value) in gpu_pipes:
                    if key == model_id:
                        return value
            for (key, pipe) in self._cpu_pipes.items():
                if key == model_id:
                    return pipe

            return None


def count_params(model):
    total_size = sum(param.numel() for param in model.parameters())
    mul = 2
    if model.dtype in (torch.float16, torch.bfloat16):
        mul = 2
    elif model.dtype == torch.float32:
        mul = 4
    return total_size * mul


def get_size(obj):
    return sys.getsizeof(obj)


def get_model_size(pipeline):
    total_size = 0
    for name, component in pipeline.components.items():
        if isinstance(component, torch.nn.Module):
            total_size += count_params(component)
        elif hasattr(component, 'tokenizer'):
            total_size += count_params(component.tokenizer)
        else:
            total_size += get_size(component)
    return total_size / (1024 * 1024)


def awailable_ram():
    mem = psutil.virtual_memory()
    available_ram = mem.available
    return available_ram / (1024 * 1024)
