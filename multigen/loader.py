from typing import Type
import torch
import logging
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline


logger = logging.getLogger(__file__)

class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._pipes = dict()

    def load_pipeline(self, cls: Type[DiffusionPipeline], path, torch_dtype=torch.float16, **additional_args):
        for key, pipe in self._pipes.items():
            if key == path:
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
        return result

    def register_pipeline(self, pipe: DiffusionPipeline, model_id):
        self._pipes[model_id] = pipe

    def remove_pipeline(self, model_id):
        self._pipes.pop(model_id)

    def get_pipeline(self, model_id):
        return self._pipes.get(model_id, None)
