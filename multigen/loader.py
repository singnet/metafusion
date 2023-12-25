from typing import Type
from diffusers import DiffusionPipeline


class Loader:
    """
    class for loading diffusion pipelines from files.
    """
    def __init__(self):
        self._pipes = dict()

    def load_pipeline(self, cls: Type[DiffusionPipeline], path):
        for key, pipe in self._pipes.items():
            if key == path:
                return cls(**pipe.components)
        if path.endswith('safetensors'):
            result = cls.from_single_file(path)
        else:
            result = cls.from_pretrained(path)
        self.register_pipeline(result, path)
        return result

    def register_pipeline(self, pipe: DiffusionPipeline, model_id):
        self._pipes[model_id] = pipe

    def remove_pipeline(self, model_id):
        self._pipes.pop(model_id)
