import unittest
import os
import shutil
import PIL
import torch
import numpy

from multigen import Prompt2ImPipe, Cfgen, GenSession, Loader, MaskedIm2ImPipe
from dummy import DummyDiffusionPipeline


class TestCase(unittest.TestCase):

    def compute_diff(self, im1: PIL.Image.Image, im2: PIL.Image.Image) -> float:
        # convert to numpy array
        im1 = numpy.asarray(im1)
        im2 = numpy.asarray(im2)
        # compute difference as float
        diff = numpy.sum(numpy.abs(im1.astype(numpy.float32) - im2.astype(numpy.float32)))
        return diff



class MyTestCase(TestCase):

    def setUp(self):
        self._pipeline = None

    def get_model(self):
        return "hf-internal-testing/tiny-stable-diffusion-torch"

    def get_ref_image(self):
        return PIL.Image.open("cube_planet_dms.png")

    def test_basic_txt2im(self):
        model = self.get_model()
        # create pipe
        pipe = Prompt2ImPipe(model, pipe=self._pipeline)
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="DPMSolverMultistepScheduler", steps=5)
        seed = 49045438434843
        params = dict(prompt="a cube  planet, cube-shaped, space photo, masterpiece",
                      negative_prompt="spherical",
                      generator=torch.cuda.manual_seed(seed))
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

        pipe = Prompt2ImPipe(model, pipe=self._pipeline)
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
        pipeline = loader.load_pipeline(MaskedIm2ImPipe._class, model_id)
        inpaint = MaskedIm2ImPipe(model_id, pipe=pipeline)

        # create prompt2im pipe
        pipeline = loader.load_pipeline(Prompt2ImPipe._class, model_id)
        prompt2image = Prompt2ImPipe(model_id, pipe=pipeline)
        prompt2image.setup(width=512, height=512, scheduler="DPMSolverMultistepScheduler", clip_skip=2, steps=5)

        self.assertEqual(inpaint.pipe.unet.conv_out.weight.data_ptr(),
                         prompt2image.pipe.unet.conv_out.weight.data_ptr(),
                         "unets are different")

    
class TestSDXL(MyTestCase):

    def get_model(self):
        return "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


if __name__ == '__main__':
    unittest.main()
