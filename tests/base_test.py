import unittest
import os
import logging
import shutil
import PIL
import torch
import numpy

from PIL import Image
from multigen import Prompt2ImPipe, Im2ImPipe, Cfgen, GenSession, Loader, MaskedIm2ImPipe, CIm2ImPipe
from multigen.log import setup_logger
from multigen.pipes import ModelType
from dummy import DummyDiffusionPipeline


def found_models():
    if os.environ.get('METAFUSION_MODELS_DIR'):
        return True
    return False


class TestCase(unittest.TestCase):

    def compute_diff(self, im1: PIL.Image.Image, im2: PIL.Image.Image) -> float:
        # convert to numpy array
        im1 = numpy.asarray(im1)
        im2 = numpy.asarray(im2)
        # compute difference as float
        diff = numpy.sum(numpy.abs(im1.astype(numpy.float32) - im2.astype(numpy.float32)))
        return diff

    def setUp(self):
        self._pipeline = None
        self._img_count = 0
        self.schedulers = 'DPMSolverMultistepScheduler', 'DDIMScheduler', 'EulerAncestralDiscreteScheduler'
        self.device_args = dict()

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/icb_diffusers'
        return "hf-internal-testing/tiny-stable-diffusion-torch"

    def get_ref_image(self, dw, dh):
        img = Image.open("cube_planet_dms.png")
        img = img.resize((img.width + dw, img.height + dh))
        pth = './cube_planet_dms' + str(self._img_count) + '.png'
        self._img_count += 1
        img.save(pth)
        return pth

    def model_type(self):
        return ModelType.SDXL if 'TestSDXL' in str(self.__class__) else ModelType.SD

    def get_cls_by_type(self, pipe):
        classes = dict()
        classes[ModelType.SDXL] = pipe._classxl
        classes[ModelType.SD] = pipe._class
        classes[ModelType.FLUX] =  pipe._classflux
        return classes
