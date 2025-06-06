import unittest
import os
import logging
import shutil
import PIL
import torch
import numpy

from PIL import Image
from multigen import Prompt2ImPipe, Im2ImPipe, Cond2ImPipe, Cfgen, GenSession, Loader, MaskedIm2ImPipe, CIm2ImPipe
from multigen.log import setup_logger
from multigen.pipes import ModelType
from dummy import DummyDiffusionPipeline
from base_test import TestCase, found_models


class MyTestCase(TestCase):

    def setUp(self):
        TestCase.setUp(self)

    def test_basic_txt2im(self):
        model = self.get_model()
        # create pipe
        pipe = Prompt2ImPipe(model, pipe=self._pipeline, model_type=self.model_type(), **self.device_args)
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler=self.schedulers[0], steps=5)
        seed = 49045438434843
        params = dict(prompt="a cube  planet, cube-shaped, space photo, masterpiece",
                      negative_prompt="spherical",
                      generator=torch.Generator().manual_seed(seed))
        image = pipe.gen(params)[0]
        image.save("cube_test.png")

        # generate with different scheduler
        if self.model_type() == ModelType.FLUX:
            params.update(generator=torch.Generator().manual_seed(seed + 1))
        else:
            params.update(scheduler=self.schedulers[1])
        image_ddim = pipe.gen(params)[0]
        image_ddim.save("cube_test2_dimm.png")
        diff = self.compute_diff(image_ddim, image)
        # check that difference is large
        self.assertGreater(diff, 1000)

    def test_with_session(self):
        model = self.get_model()

        nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

        prompt = [["+", ["bioinformatics", "bio-tech"], "lab", "with",
                   ["computers", "flasks"], {"w": "and exotic flowers", "p": 0.5}],
                  "happy vibrant",
                  ["green colors", "dream colors", "neon glowing"],
                  ["8k RAW photo, masterpiece, super quality", "artwork", "unity 3D"],
                  ["surrealism", "impressionism", "high tech", "cyberpunk"]]
        pipe = Prompt2ImPipe(model, pipe=self._pipeline, model_type=self.model_type(), **self.device_args)
        pipe.setup(width=512, height=512, scheduler=self.schedulers[0], steps=5)
        # remove directory if it exists
        dirname = "./gen_batch"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        # create session
        gs = GenSession(dirname, pipe, Cfgen(prompt, nprompt))
        gs.gen_sess(add_count=2)
        # count number of generated files
        # each images goes with a txt file
        self.assertEqual(len(os.listdir(dirname)), 4)

    def test_img2img_basic(self):
        pipe = Im2ImPipe(self.get_model(), model_type=self.model_type(), **self.device_args)
        dw, dh = -1, 1
        im = self.get_ref_image(dw, dh)
        seed = 49045438434843
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=3.3)
        self.assertEqual(3.3, pipe.pipe_params['guidance_scale'])
        image = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator().manual_seed(seed)))[0]
        image.save('test_img2img_basic.png')
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=7.6)
        image1 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator().manual_seed(seed)))[0]
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertGreater(diff, 1000)
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=3.3)
        image2 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator().manual_seed(seed)))[0]
        diff = self.compute_diff(image2, image)
        # check that difference is small
        self.assertLess(diff, 1)

    def test_maskedimg2img_basic(self):
        pipe = MaskedIm2ImPipe(self.get_model(), model_type=self.model_type(), **self.device_args)
        img = PIL.Image.open("./mech_beard_sigm.png")
        dw, dh = -1, -1
        img = img.crop((0, 0, img.width + dw, img.height + dh))
        logging.info(f'testing on image {img.size}')

        # read image with mask painted over
        img_paint = PIL.Image.open("./mech_beard_sigm_mask.png")
        img_paint = img_paint.crop((0, 0, img_paint.width + dw, img_paint.height + dh))
        img_paint = numpy.asarray(img_paint)

        scheduler = self.schedulers[-1]
        seed = 49045438434843
        blur = 48
        param_3_3 = dict(image=img, image_painted=img_paint, strength=0.96,
               scheduler=scheduler, clip_skip=0, blur=blur, blur_compose=3, steps=5, guidance_scale=3.3)
        param_7_6 = dict(image=img, image_painted=img_paint, strength=0.96,
               scheduler=scheduler, clip_skip=0, blur=blur, blur_compose=3, steps=5, guidance_scale=7.6)
        pipe.setup(**param_3_3)
        self.assertEqual(3.3, pipe.pipe_params['guidance_scale'])
        image = pipe.gen(dict(prompt="cube planet cartoon style", 
                              generator=torch.Generator().manual_seed(seed)))[0]
        self.assertEqual(image.width, img.width)
        self.assertEqual(image.height, img.height)
        image.save('test_img2img_basic.png')
        pipe.setup(**param_7_6)
        image1 = pipe.gen(dict(prompt="cube planet cartoon style", 
                               generator=torch.Generator().manual_seed(seed)))[0]
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertGreater(diff, 1000)
        pipe.setup(**param_3_3)
        image2 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator().manual_seed(seed)))[0]
        diff = self.compute_diff(image2, image)
        # check that difference is small
        self.assertLess(diff, 1)
        self.assertEqual(image.width, img.width)
        self.assertEqual(image.height, img.height)

    @unittest.skipIf(not found_models(), "can't run on tiny version of SD")
    def test_lpw(self):
        """
        Check that last part of long prompt affect the generation
        """
        pipe = Prompt2ImPipe(self.get_model(), model_type=self.model_type(), lpw=True, **self.device_args)
        prompt = ' a cubic planet with atmoshere as seen from low orbit, each side of the cubic planet is ocuppied by an ocean, oceans have islands, but no continents, atmoshere of the planet has usual sperical shape, corners of the cube are above the atmoshere, but edges largely are covered by the atomosphere, there are cyclones in the atmoshere, the photo is made from low-orbit, famous sci-fi illustration'
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler=self.schedulers[0], steps=5)
        seed = 49045438434843
        params = dict(prompt=prompt,
                      negative_prompt="spherical",
                      generator=torch.Generator().manual_seed(seed))
        image = pipe.gen(params)[0]
        image.save("cube_test_lpw.png")
        params = dict(prompt=prompt + " , best quality, famous photo",
                negative_prompt="spherical",
                generator=torch.Generator().manual_seed(seed))
        image1 = pipe.gen(params)[0]
        image.save("cube_test_lpw1.png")
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertGreater(diff, 1000)

    @unittest.skipIf(not found_models(), "can't run on tiny version of SD")
    def test_lpw_turned_off(self):
        """
        Check that last part of long prompt don't affect the generation with lpw turned off
        """
        pipe = Prompt2ImPipe(self.get_model(), model_type=self.model_type(), lpw=False)
        prompt = ' a cubic planet with atmoshere as seen from low orbit, each side of the cubic planet is ocuppied by an ocean, oceans have islands, but no continents, atmoshere of the planet has usual sperical shape, corners of the cube are above the atmoshere, but edges largely are covered by the atomosphere, there are cyclones in the atmoshere, the photo is made from low-orbit, famous sci-fi illustration'
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler=self.schedulers[0], steps=5)
        seed = 49045438434843
        params = dict(prompt=prompt,
                      negative_prompt="spherical",
                      generator=torch.Generator().manual_seed(seed))
        image = pipe.gen(params)[0]
        image.save("cube_test_no_lpw.png")
        params = dict(prompt=prompt + " , best quality, famous photo",
                negative_prompt="spherical",
                generator=torch.Generator().manual_seed(seed))
        image1 = pipe.gen(params)[0]
        image.save("cube_test_no_lpw1.png")
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertLess(diff, 1)

    @unittest.skipIf(not found_models(), "can't run on tiny version of SD")
    def test_controlnet(self):
        model = self.get_model()
        model_type = self.model_type()
        # create pipe
        if model_type == ModelType.FLUX:
            # pass 
            canny_path = os.path.join(os.environ.get('METAFUSION_MODELS_DIR'), "ControlNetFlux/FLUX.1-dev-Controlnet-Canny-alpha/")
            canny_path = "InstantX/FLUX.1-dev-Controlnet-Canny"
            pipe = CIm2ImPipe(model, model_type=self.model_type(), cnet_ids=[canny_path], ctypes=['soft'], **self.device_args)
        else:
            pipe = CIm2ImPipe(model, model_type=self.model_type(), ctypes=['soft'], **self.device_args)
        
        logging.info(f"pipe's device {pipe.pipe.device}")
        dw, dh = 1, -1
        imgpth = self.get_ref_image(dw, dh)
        pipe.setup(imgpth, cscales=[0.3], guidance_scale=7, scheduler=self.schedulers[0], steps=5)
        seed = 49045438434843
        params = dict(prompt="cube planet minecraft style",
                      negative_prompt="spherical",
                      generator=torch.Generator().manual_seed(seed))
        image = pipe.gen(params)[0]
        image.save("mech_test.png")
        img_ref = PIL.Image.open(imgpth)
        self.assertEqual(image.width, img_ref.width)
        self.assertEqual(image.height, img_ref.height)

        if self.model_type() == ModelType.FLUX:
            # generate with different generator
            params.update(generator=torch.Generator().manual_seed(seed + 1))
        else:
            # generate with different scheduler
            params.update(scheduler=self.schedulers[1])
        image_ddim = pipe.gen(params)[0]
        image_ddim.save("cube_test2_dimm.png")
        diff = self.compute_diff(image_ddim, image)
        # check that difference is large
        self.assertGreater(diff, 1000)
    
    def test_cond2im(self):
        model = self.get_model()
        model_type = self.model_type()
        pipe = Cond2ImPipe(model, ctypes=["pose"], model_type=model_type)
        pipe.setup("./pose6.jpeg", width=768, height=768)
        seed = 49045438434843
        params = dict(prompt="child in the coat playing in sandbox",
                      negative_prompt="spherical",
                      generator=torch.Generator().manual_seed(seed))
        img = pipe.gen(params)[0]
        self.assertEqual(img.size, (768, 768))
        pipe.setup("./pose6.jpeg")
        img1 = pipe.gen(params)[0]
        self.assertEqual(img1.size, (450, 450))


class TestSDXL(MyTestCase):

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/SDXL/stable-diffusion-xl-base-1.0'
        return "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


class TestFlux(MyTestCase):

    def setUp(self):
        super().setUp()
        self._pipeline = None
        self.schedulers = ['FlowMatchEulerDiscreteScheduler']
        self.device_args = dict()
        self.device_args['device'] = torch.device('cpu')
        if torch.cuda.is_available():
            self.device_args['offload_device'] = 0

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

    @unittest.skip('not implemented yet')
    def test_cond2im(self):
        pass


if __name__ == '__main__':
    setup_logger('test_pipe.log')
    unittest.main()
