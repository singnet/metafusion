import unittest
import os
import shutil
import PIL
import torch
import numpy

from multigen import Prompt2ImPipe, Cfgen, GenSession, Loader
from dummy import DummyDiffusionPipeline
from pipes import MaskedIm2ImPipe


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self._pipeline = None
        # read environment variables
        self.use_dummy = os.environ.get("USE_DUMMY")
        if self.use_dummy:
            self._pipeline = DummyDiffusionPipeline()
            self._pipeline.add_image(PIL.Image.open("cube_planet_dms.png"))

    def test_basic_txt2im(self):
        model = "runwayml/stable-diffusion-v1-5"
        # create pipe
        pipe = Prompt2ImPipe(model, pipe=self._pipeline)
        pipe.setup(width=512, height=512, guidance_scale=7, scheduler="DPMSolverMultistepScheduler")
        seed = 49045438434843
        params = dict(prompt="a cube  planet, cube-shaped, space photo, masterpiece",
                      negative_prompt="spherical",
                      generator=torch.cuda.manual_seed(seed))
        image = pipe.gen(params)
        image.save("cube_test.png")
        # load reference image
        ref_image = PIL.Image.open("cube_planet_dms.png")

        # compute difference as float
        diff = self.compute_diff(ref_image, image)
        # check that difference is small
        self.assertLess(diff, 0.0001)
        # generate with different scheduler
        params.update(scheduler="DDIMScheduler")
        image = pipe.gen(params)
        image.save("cube_test2_dimm.png")
        diff = self.compute_diff(ref_image, image)
        # check that difference is large
        if not self.use_dummy:
            self.assertGreater(diff, 1000)

    def test_with_session(self):
        model = "runwayml/stable-diffusion-v1-5"

        nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

        prompt = [["+", ["bioinformatics", "bio-tech"], "lab", "with",
                   ["computers", "flasks"], {"w": "and exotic flowers", "p": 0.5}],
                  "happy vibrant",
                  ["green colors", "dream colors", "neon glowing"],
                  ["8k RAW photo, masterpiece, super quality", "artwork", "unity 3D"],
                  ["surrealism", "impressionism", "high tech", "cyberpunk"]]

        pipe = Prompt2ImPipe(model, pipe=self._pipeline)
        pipe.setup(width=512, height=512, scheduler="DPMSolverMultistepScheduler")
        # remove directory if it exists
        dirname = "./gen_batch"
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        # create session
        gs = GenSession(dirname, pipe, Cfgen(prompt, nprompt))
        gs.gen_sess(add_count=10)
        # count number of generated files
        self.assertEqual(len(os.listdir(dirname)), 20)

    def compute_diff(self, im1: PIL.Image.Image, im2: PIL.Image.Image) -> float:
        # convert to numpy array
        im1 = numpy.asarray(im1)
        im2 = numpy.asarray(im2)
        # compute difference as float
        diff = numpy.sum(numpy.abs(im1.astype(numpy.float32) - im2.astype(numpy.float32)))
        return diff

    def test_loader(self):
        loader = Loader()

        # create prompt2im pipe
        pipeline = loader.load_pipeline(Prompt2ImPipe._class, 'models-sd/icbinp')
        prompt2image = Prompt2ImPipe('models-sd/icbinp', pipe=pipeline)

        # load inpainting pipe
        pipeline = loader.load_pipeline(MaskedIm2ImPipe._class, 'models-sd/icbinp')
        inpaint = MaskedIm2ImPipe('models-sd/icbinp', pipe=pipeline)
        self.assertEqual(inpaint.pipe.unet.conv_out.weight.data_ptr(),
                         prompt2image.pipe.unet.conv_out.weight.data_ptr(),
                         "unets are different")


if __name__ == '__main__':
    unittest.main()
