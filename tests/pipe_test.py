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


class TestCase(unittest.TestCase):

    def compute_diff(self, im1: PIL.Image.Image, im2: PIL.Image.Image) -> float:
        # convert to numpy array
        im1 = numpy.asarray(im1)
        im2 = numpy.asarray(im2)
        # compute difference as float
        diff = numpy.sum(numpy.abs(im1.astype(numpy.float32) - im2.astype(numpy.float32)))
        return diff



def found_models():
    if os.environ.get('METAFUSION_MODELS_DIR'):
        return True
    return False


class MyTestCase(TestCase):

    def setUp(self):
        self._pipeline = None
        self._img_count = 0

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

    def test_basic_txt2im(self):
        model = self.get_model()
        # create pipe
        pipe = Prompt2ImPipe(model, pipe=self._pipeline, model_type=self.model_type())
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="DPMSolverMultistepScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt="a cube  planet, cube-shaped, space photo, masterpiece",
                      negative_prompt="spherical",
                      generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image = pipe.gen(params)
        image.save("cube_test.png")

        # generate with different scheduler
        params.update(scheduler="DDIMScheduler")
        image_ddim = pipe.gen(params)
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
        pipe = Prompt2ImPipe(model, pipe=self._pipeline, model_type=self.model_type())
        pipe.setup(width=512, height=512, scheduler="DPMSolverMultistepScheduler", steps=5)
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

    def test_loader(self):
        loader = Loader()
        model_id = self.get_model()

        # load inpainting pipe
        is_xl = 'TestSDXL' in str(self.__class__)
        if is_xl:
            cls = MaskedIm2ImPipe._classxl
        else:
            cls = MaskedIm2ImPipe._class
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        pipeline = loader.load_pipeline(cls, model_id, device=device)
        inpaint = MaskedIm2ImPipe(model_id, pipe=pipeline)

        # create prompt2im pipe
        if is_xl:
            cls = Prompt2ImPipe._classxl
        else:
            cls = Prompt2ImPipe._class
        pipeline = loader.load_pipeline(cls, model_id, device=device)
        prompt2image = Prompt2ImPipe(model_id, pipe=pipeline)
        prompt2image.setup(width=512, height=512, scheduler="DPMSolverMultistepScheduler", clip_skip=2, steps=5)
        if device.type == 'cuda':
            self.assertEqual(inpaint.pipe.unet.conv_out.weight.data_ptr(),
                         prompt2image.pipe.unet.conv_out.weight.data_ptr(),
                         "unets are different")

    def test_img2img_basic(self):
        pipe = Im2ImPipe(self.get_model(), model_type=self.model_type())
        dw, dh = -1, 1
        im = self.get_ref_image(dw, dh)
        seed = 49045438434843
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=3.3)
        self.assertEqual(3.3, pipe.pipe_params['guidance_scale'])
        image = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
        image.save('test_img2img_basic.png')
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=7.6)
        image1 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertGreater(diff, 1000)
        pipe.setup(im, strength=0.7, steps=5, guidance_scale=3.3)
        image2 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
        diff = self.compute_diff(image2, image)
        # check that difference is small
        self.assertLess(diff, 1)

    def test_maskedimg2img_basic(self):
        pipe = MaskedIm2ImPipe(self.get_model(), model_type=self.model_type())
        img = PIL.Image.open("./mech_beard_sigm.png")
        dw, dh = -1, -1
        img = img.crop((0, 0, img.width + dw, img.height + dh))
        logging.info(f'testing on image {img.size}')

        # read image with mask painted over
        img_paint = PIL.Image.open("./mech_beard_sigm_mask.png")
        img_paint = img_paint.crop((0, 0, img_paint.width + dw, img_paint.height + dh))
        img_paint = numpy.asarray(img_paint)

        scheduler = "EulerAncestralDiscreteScheduler"
        seed = 49045438434843
        blur = 48
        param_3_3 = dict(original_image=img, image_painted=img_paint, strength=0.96,
               scheduler=scheduler, clip_skip=0, blur=blur, blur_compose=3, steps=5, guidance_scale=3.3)
        param_7_6 = dict(original_image=img, image_painted=img_paint, strength=0.96,
               scheduler=scheduler, clip_skip=0, blur=blur, blur_compose=3, steps=5, guidance_scale=7.6)
        pipe.setup(**param_3_3)
        self.assertEqual(3.3, pipe.pipe_params['guidance_scale'])
        image = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
        self.assertEquals(image.width, img.width)
        self.assertEquals(image.height, img.height)
        image.save('test_img2img_basic.png')
        pipe.setup(**param_7_6)
        image1 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertGreater(diff, 1000)
        pipe.setup(**param_3_3)
        image2 = pipe.gen(dict(prompt="cube planet cartoon style", generator=torch.Generator(pipe.pipe.device).manual_seed(seed)))
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
        pipe = Prompt2ImPipe(self.get_model(), model_type=self.model_type(), lpw=True)
        prompt = ' a cubic planet with atmoshere as seen from low orbit, each side of the cubic planet is ocuppied by an ocean, oceans have islands, but no continents, atmoshere of the planet has usual sperical shape, corners of the cube are above the atmoshere, but edges largely are covered by the atomosphere, there are cyclones in the atmoshere, the photo is made from low-orbit, famous sci-fi illustration'
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="DPMSolverMultistepScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt=prompt,
                      negative_prompt="spherical",
                      generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image = pipe.gen(params)
        image.save("cube_test_lpw.png")
        params = dict(prompt=prompt + " , best quality, famous photo",
                negative_prompt="spherical",
                generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image1 = pipe.gen(params)
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
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="DPMSolverMultistepScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt=prompt,
                      negative_prompt="spherical",
                      generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image = pipe.gen(params)
        image.save("cube_test_no_lpw.png")
        params = dict(prompt=prompt + " , best quality, famous photo",
                negative_prompt="spherical",
                generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image1 = pipe.gen(params)
        image.save("cube_test_no_lpw1.png")
        diff = self.compute_diff(image1, image)
        # check that difference is large
        self.assertLess(diff, 1)

    @unittest.skipIf(not found_models(), "can't run on tiny version of SD")
    def test_controlnet(self):
        model = self.get_model()
        # create pipe
        pipe = CIm2ImPipe(model, model_type=self.model_type(), ctypes=['soft'])
        dw, dh = 1, -1
        imgpth = self.get_ref_image(dw, dh)
        pipe.setup(imgpth, cscales=[0.3], guidance_scale=7, scheduler="DPMSolverMultistepScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt="cube planet minecraft style",
                      negative_prompt="spherical",
                      generator=torch.Generator(pipe.pipe.device).manual_seed(seed))
        image = pipe.gen(params)
        image.save("mech_test.png")
        img_ref = PIL.Image.open(imgpth)
        self.assertEqual(image.width, img_ref.width)
        self.assertEqual(image.height, img_ref.height)

        # generate with different scheduler
        params.update(scheduler="DDIMScheduler")
        image_ddim = pipe.gen(params)
        image_ddim.save("cube_test2_dimm.png")
        diff = self.compute_diff(image_ddim, image)
        # check that difference is large
        self.assertGreater(diff, 1000)

class TestSDXL(MyTestCase):

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/SDXL/stable-diffusion-xl-base-1.0'
        return "hf-internal-testing/tiny-stable-diffusion-xl-pipe"



class TestFlux(TestCase):

    def setUp(self):
        self._pipeline = None

    def model_type(self):
        return ModelType.FLUX

    def get_model(self):
        models_dir = os.environ.get('METAFUSION_MODELS_DIR', None)
        if models_dir is not None:
            return models_dir + '/flux.1-schnell'
        return './models-sd/' + "/tiny-flux-pipe"


    def test_basic_txt2im(self):
        model = self.get_model()
        device = torch.device('cpu', 0)
        # create pipe
        offload = 0 if torch.cuda.is_available() else None
        pipe = Prompt2ImPipe(model, pipe=self._pipeline,
                             model_type=self.model_type(),
                             device=device, offload_device=offload)
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="FlowMatchEulerDiscreteScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt="a cube  planet, cube-shaped, space photo, masterpiece",
                      negative_prompt="spherical",
                      generator=torch.Generator(device).manual_seed(seed))
        image = pipe.gen(params)
        image.save("cube_test.png")

        # generate with different seed
        params['generator'] = torch.Generator(device).manual_seed(seed + 1)
        image_ddim = pipe.gen(params)
        image_ddim.save("cube_test2_dimm.png")
        diff = self.compute_diff(image_ddim, image)
        # check that difference is large
        self.assertGreater(diff, 1000)


if __name__ == '__main__':
    setup_logger('test_pipe.log')
    unittest.main()
