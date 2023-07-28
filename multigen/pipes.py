import importlib
import torch

from PIL import Image
import cv2
import numpy as np
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# from diffusers import StableDiffusionKDiffusionPipeline


def get_diffusion_scheduler_names():
    """
    return list of schedulers that can be use in our pipelines
    """
    scheduler_names = []
    for scheduler in KarrasDiffusionSchedulers:
        scheduler_names.append(scheduler.name)
    return scheduler_names


def add_scheduler(pipe, scheduler):
    # setup scheduler
    if scheduler is None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return 'DPMSolverMultistepScheduler'
    else:
        # Import the scheduler class dynamically based on the provided name
        try:
            module_name = "diffusers.schedulers"
            class_name = scheduler
            module = importlib.import_module(module_name)
            scheduler_class = getattr(module, class_name)
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
            return class_name
        except (ImportError, AttributeError):
            raise ValueError("Invalid scheduler specified")
    return None


class BasePipe:
    def __init__(self):
        self.pipe = None
        self._scheduler = None


    def try_set_scheduler(self, inputs):
        # allow for scheduler overwrite
        scheduler = inputs.get('scheduler', None)
        if scheduler is not None and self.pipe is not None:
            sch_set = add_scheduler(self.pipe, scheduler=scheduler)
            if sch_set:
                self._scheduler = sch_set
            inputs.pop('scheduler')


class Prompt2ImPipe(BasePipe):

    def __init__(self, model_id, dtype=torch.float16, lpw=False, scheduler=None):
        super().__init__()
        self.model_id = model_id
        # TODO? Any other custom pipeline?
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype) if not lpw else \
            StableDiffusionPipeline.from_pretrained(model_id, #StableDiffusionKDiffusionPipeline
                                                    custom_pipeline="lpw_stable_diffusion",
                                                    torch_dtype=dtype)
        self.pipe.to("cuda")
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_vae_slicing()
        self.pipe.vae.enable_tiling()
        # --- the best one and seems to be enough ---
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention() # attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.pipe.vae.enable_xformers_memory_efficient_attention() # attention_op=None)
        self.try_set_scheduler(dict(scheduler=scheduler))

    def setup(self, width=768, height=768):
        self.pipe_params = {
            "width": width,
            "height": height,
        }

    def get_config(self):
        cfg = { "model_id": self.model_id }
        cfg.update(self.pipe_params)
        return cfg

    def gen(self, inputs):
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        # allow for scheduler overwrite
        self.try_set_scheduler(inputs)
        image = self.pipe(**inputs).images[0]
        return image


class Im2ImPipe(BasePipe):

    def __init__(self, model_id, dtype=torch.float16, scheduler=None):
        super().__init__()
        self.model_id = model_id
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to("cuda")
        self.pipe.vae.enable_tiling()
        # setup scheduler
        self.try_set_scheduler(dict(scheduler=scheduler))

    def setup(self, fimage, image=None, strength=0.75, gscale=7.5, scale=None):
        self.fname = fimage
        self.image = Image.open(fimage).convert("RGB") if image is None else image
        if scale is not None:
            if not isinstance(scale, list):
                scale = [8*(int(self.image.size[i]*scale)//8) for i in range(2)]
            self.image = self.image.resize((scale[0], scale[1]))
        self.pipe_params = {
            "strength": strength,
            "guidance_scale": gscale
        }

    def get_config(self):
        cfg = {
            "model_id": self.model_id,
            "source_image": self.fname,
        }
        cfg.update(self.pipe_params)
        return cfg

    def gen(self, inputs):
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({"image": self.image})
        self.try_set_scheduler(inputs)
        image = self.pipe(**inputs).images[0]
        return image


class CIm2ImPipe:

    # TODO: set path
    cpath = "./models-cn/"
    cmodels = {
        "canny": "sd-controlnet-canny",
        "pose": "control_v11p_sd15_openpose",
        "ip2p": "control_v11e_sd15_ip2p",
        "soft-sobel": "control_v11p_sd15_softedge",
        "soft": "control_v11p_sd15_softedge",
        "depth": "control_v11f1p_sd15_depth"
    }
    cscalem = {
        "canny": 0.75,
        "pose": 1.0,
        "ip2p": 0.5,
        "soft-sobel": 0.3,
        "soft": 0.95, #0.5
        "depth": 0.5
    }

    def __init__(self, model_id, dtype=torch.float16, ctypes=["soft"]):
        self.model_id = model_id
        if not isinstance(ctypes, list):
            ctypes = [ctypes]
        self.ctypes = ctypes
        cnets = [ControlNetModel.from_pretrained(CIm2ImPipe.cpath+CIm2ImPipe.cmodels[c], torch_dtype=dtype) for c in ctypes]
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=cnets, torch_dtype=dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
        self.pipe.vae.enable_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()

        if "soft" in ctypes:
            from controlnet_aux import PidiNetDetector, HEDdetector
            self.processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
            #processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        if "pose" in ctypes:
            from pytorch_openpose.src.body import Body
            from pytorch_openpose.src import util
            self.body_estimation = Body('pytorch_openpose/model/body_pose_model.pth')
            self.draw_bodypose = util.draw_bodypose
            #hand_estimation = Hand('model/hand_pose_model.pth')
        if "depth" in ctypes:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            self.dprocessor = DPTImageProcessor.from_pretrained("./models-other/dpt-large")
            self.dmodel = DPTForDepthEstimation.from_pretrained("./models-other/dpt-large")


    def setup(self, fimage, width=None, height=None, image=None, cscales=None, guess_mode=False):
        self.fname = fimage
        image = Image.open(fimage) if image is None else image
        self.image = self._proc_cimg(np.asarray(image))
        if cscales is None:
            cscales = [CIm2ImPipe.cscalem[c] for c in self.ctypes]
        self.pipe_params = {
            "width": image.size[0] if width is None else width,
            "height": image.size[1] if height is None else height,
            "controlnet_conditioning_scale": cscales,
            "guess_mode": guess_mode,
            "num_inference_steps": 20
        }

    def get_config(self):
        cfg = {
            "model_id": self.model_id,
            "source_image": self.fname,
            "control_type": self.ctypes
        }
        cfg.update(self.pipe_params)
        return cfg

    def _proc_cimg(self, oriImg):
        cimage = []
        for c in self.ctypes:
            if c == "canny":
                image = canny_processor(oriImg)
                cimage += [Image.fromarray(image)]
            elif c == "pose":
                candidate, subset = self.body_estimation(oriImg)
                canvas = np.zeros(oriImg.shape, dtype = np.uint8)
                canvas = self.draw_bodypose(canvas, candidate, subset)
                #canvas[:, :, [2, 1, 0]]
                cimage += [Image.fromarray(canvas)]
            elif c == "soft":
                cimage += [self.processor(oriImg)]
            elif c == "soft-sobel":
                edge = sobel_processor(oriImg)
                cimage += [Image.fromarray(edge)]
            elif c == "depth":
                image = Image.fromarray(oriImg)
                inputs = self.dprocessor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.dmodel(**inputs)
                    predicted_depth = outputs.predicted_depth
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                output = prediction.squeeze().cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype("uint8")
                cimage += [Image.fromarray(formatted)]
            else:
                cimage += [Image.fromarray(oriImg)]
        return cimage

    def gen(self, inputs):
        inputs = {**inputs}
        inputs.update(self.pipe_params)
        inputs.update({"image": self.image})
        image = self.pipe(**inputs).images[0]
        return image

def canny_processor(oriImg):
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(oriImg, low_threshold, high_threshold)
    image = image[:, :, None]
    return np.concatenate([image, image, image], axis=2)

def sobel_processor(oriImg, ksize=9):
    gray = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, 0, 1, 0, ksize=ksize, scale=1)
    y = cv2.Sobel(gray, 0, 0, 1, ksize=ksize, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    #edge = np.hypot(x, y)
    return edge
