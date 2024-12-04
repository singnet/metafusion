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
from base_test import TestCase, found_models
from multigen.util import quantize, weightshare_copy


class LoaderTestCase(TestCase):

    def test_loader_same_weights(self):
        """
        Test that weights are shared for different pipelines loaded from the same
        checkpoint
        """
        loader = Loader()
        model_id = self.get_model()
        model_type = self.model_type()
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        if 'device' not in self.device_args:
            self.device_args['device'] = device
        if 'offload_device' in self.device_args:
            del self.device_args['offload_device']
        classes = self.get_cls_by_type(MaskedIm2ImPipe)
        # load inpainting pipe
        cls = classes[model_type]
        logging.info(f'loading {cls} from {model_id} {self.device_args}')
        pipeline = loader.load_pipeline(cls, model_id, **self.device_args)
        inpaint = MaskedIm2ImPipe(model_id, pipe=pipeline,  **self.device_args)

        prompt_classes = self.get_cls_by_type(Prompt2ImPipe)
        # create prompt2im pipe
        cls = prompt_classes[model_type]
        device_args = dict(**self.device_args)
        device = device_args.get('device', None)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda', 0)
            else:
                device = torch.device('cpu')
            device_args['device'] = device

        pipeline = loader.load_pipeline(cls, model_id, **device_args)
        prompt2image = Prompt2ImPipe(model_id, pipe=pipeline, **device_args)
        prompt2image.setup(width=512, height=512, scheduler=self.schedulers[0], clip_skip=2, steps=5)
        self.assertEqual(inpaint.pipe.vae.decoder.conv_out.weight.data_ptr(),
                        prompt2image.pipe.vae.decoder.conv_out.weight.data_ptr(),
                        "unets are different")

    def test_different_id(self):
        """
        Check that loader returns new pipeline with new components when loading the same checkpoint
        """
        model_id = self.get_model()
        model_type = self.model_type()
        classes = self.get_cls_by_type(MaskedIm2ImPipe)
        cls = classes[model_type]
        loader = Loader()
        load_device = torch.device('cpu')
        pipe11 = loader.load_pipeline(cls, model_id,
                torch_dtype=torch.bfloat16,
                device=load_device)
        for value in loader._cpu_pipes.values():
            assert id(value) != id(pipe11)
        pipe1 = Prompt2ImPipe('model_id', pipe=pipe11, device=load_device, offload_device=0)
        pipe22 = loader.load_pipeline(cls, model_id,
                torch_dtype=torch.bfloat16,
                device=load_device)

        pipe2 = Prompt2ImPipe(model_id, pipe=pipe22, device=load_device, offload_device=1)

        for comp_name in pipe2.pipe.components.keys():
            comp1 = pipe1.pipe.components[comp_name]
            comp2 = pipe2.pipe.components[comp_name]
            if comp_name not in ['tokenizer', 'tokenizer_2'] and comp1 is not None:
                assert id(comp1) != id(comp2)

    def test_weightshare(self):
        """
        Check that pipes after weightshare copy share weights, but otherwise
        independant. Check that enable_sequential_cpu_offload doesn't modify copy's device
        """
        model_id = self.get_model()
        model_type = self.model_type()
        cuda0 = torch.device('cuda', 0)
        cuda1 = torch.device('cuda', 1)
        prompt_classes = self.get_cls_by_type(Prompt2ImPipe)
        # create prompt2im pipe
        cls = prompt_classes[model_type]
        offload_device = 0
        cpu = torch.device('cpu')
        pipe0 = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(cpu)
        pipe1 = weightshare_copy(pipe0)
        self.assertNotEqual(id(pipe1.scheduler), id(pipe0.scheduler))
        self.assertNotEqual(id(pipe1.vae), id(pipe0.vae))
        self.assertEqual(pipe1.vae.decoder.conv_in.weight.data_ptr(),
                         pipe0.vae.decoder.conv_in.weight.data_ptr())
        pipe0.enable_sequential_cpu_offload(offload_device)
        self.assertNotEqual(pipe1.device.type, 'meta', "Check that enable_sequential_cpu_offload doesn't modify copy's device")

    def test_quantized(self):
        model_id = self.get_model()
        model_type = self.model_type()
        cuda0 = torch.device('cuda', 0)
        cuda1 = torch.device('cuda', 1)
        prompt_classes = self.get_cls_by_type(Prompt2ImPipe)
        # create prompt2im pipe
        cls = prompt_classes[model_type]
        offload_device = 1
        cpu = torch.device('cpu')
        pipe0 = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(cpu)
        quantize(pipe0)
        pipe0.to(cuda0)
        pipe1 = weightshare_copy(pipe0)
        self.assertNotEqual(id(pipe1.scheduler), id(pipe0.scheduler))
        self.assertNotEqual(id(pipe1.vae), id(pipe0.vae))
        self.assertEqual(pipe1.vae.decoder.conv_in.weight.data_ptr(),
                         pipe0.vae.decoder.conv_in.weight.data_ptr())


class TestFlux(LoaderTestCase):

    def setUp(self):
        super().setUp()
        self._pipeline = None
        self.schedulers = ['FlowMatchEulerDiscreteScheduler']
        self.device_args = dict()
        self.device_args['device'] = torch.device('cpu')
        if torch.cuda.is_available():
            self.device_args['offload_device'] = 0
            self.device_args['torch_dtype'] = torch.bfloat16

    def model_type(self):
        return ModelType.FLUX

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/flux-1-dev'
        return './models-sd/' + "flux/tiny-flux-pipe"

    @unittest.skip('flux does not need test')
    def test_lpw_turned_off(self):
        pass


class TestSDXL(LoaderTestCase):

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/SDXL/stable-diffusion-xl-base-1.0'
        return "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


def get_test_cases():
    suites = []
    # Manually add or discover test case classes that are subclasses of LoaderTestCase
    for subclass in LoaderTestCase.__subclasses__():
        suite = unittest.TestLoader().loadTestsFromTestCase(subclass)
        suites.append(suite)
    return unittest.TestSuite(suites)


if __name__ == '__main__':
    setup_logger('test_loader.log')
    runner = unittest.TextTestRunner()
    result = runner.run(get_test_cases())
