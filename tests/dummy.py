from diffusers import StableDiffusionPipeline
import PIL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.configuration_utils import FrozenDict


class DummyVae:
    def enable_tiling(self):
        pass


class DummyScheduler:
    def __init__(self):
        self.config = FrozenDict({'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'trained_betas': None, 'solver_order': 2, 'prediction_type': 'epsilon', 'thresholding': False, 'dynamic_thresholding_ratio': 0.995, 'sample_max_value': 1.0, 'algorithm_type': 'dpmsolver++', 'solver_type': 'midpoint', 'lower_order_final': True,
                                  'use_karras_sigmas': False, 'lambda_min_clipped': float('-inf'),
                                  'variance_type': None, 'timestep_spacing': 'linspace', 'steps_offset': 1,
                                  '_use_default_values': ['prediction_type', 'use_karras_sigmas', 'solver_type',
                                                          'timestep_spacing', 'dynamic_thresholding_ratio', 'lambda_min_clipped',
                                                          'algorithm_type', 'sample_max_value', 'solver_order', 'thresholding',
                                                          'lower_order_final', 'variance_type'],
                                  'skip_prk_steps': True, 'set_alpha_to_one': False, '_class_name': 'DPMSolverMultistepScheduler',
                                  '_diffusers_version': '0.18.2', 'clip_sample': False})


class DummyDiffusionPipeline:
    """
    Dummy diffusion pipeline for testing
    """
    def __init__(self, *args, **kwargs):
        self._images = []
        self.vae = DummyVae()
        self.scheduler = DummyScheduler()

    def enable_xformers_memory_efficient_attention(self):
        pass

    def add_image(self, image: PIL.Image.Image):
        self._images.append(image)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyDiffusionPipeline(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> StableDiffusionPipelineOutput:
        return StableDiffusionPipelineOutput(self._images, False)

    def to(self, arg):
        return self
