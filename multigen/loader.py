from typing import Type
from contextlib import nullcontext
import torch
import logging
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from diffusers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import init_empty_weights
else:
    init_empty_weights = nullcontext


logger = logging.getLogger(__file__)

def copy_pipe(pipe):
    copy = pipe.__class__(**pipe.components)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        copy.unet = copy.unet.__class__.from_config(copy.unet.config)
        copy.text_encoder = copy.text_encoder.__class__(pipe.text_encoder.config)
        if hasattr(copy, 'text_encoder_2'):
            copy.text_encoder_2 = copy.text_encoder_2.__class__(pipe.text_encoder_2.config)
    # assign=True is needed since our copy is on "meta" device, i.g. weights are empty
    copy.unet.load_state_dict(pipe.unet.state_dict(), assign=True)
    copy.text_encoder.load_state_dict(pipe.text_encoder.state_dict(), assign=True)
    if hasattr(copy, 'text_encoder_2'):
        copy.text_encoder_2.load_state_dict(pipe.text_encoder_2.state_dict(), assign=True)
    return copy


class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._pipes = dict()

    def load_pipeline(self, cls: Type[DiffusionPipeline], path, torch_dtype=torch.float16, **additional_args):
        for key, pipe in self._pipes.items():
            if key == path:
                pipe = copy_pipe(pipe)
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
                return cls(**components, **additional_args).to(torch_dtype)

        if path.endswith('safetensors'):
            result = cls.from_single_file(path, **additional_args)
        else:
            result = cls.from_pretrained(path, **additional_args)
        self.register_pipeline(result, path)
        result = copy_pipe(result)
        return result

    def register_pipeline(self, pipe: DiffusionPipeline, model_id):
        self._pipes[model_id] = pipe

    def remove_pipeline(self, model_id):
        self._pipes.pop(model_id)

    def get_pipeline(self, model_id):
        return self._pipes.get(model_id, None)
