import unittest
import os
import shutil
import PIL
import torch
import numpy

from multigen import Prompt2ImPipe, Cfgen, GenSession


class MyTestCase(unittest.TestCase):

    def test_basic_txt2im(self):
        model = "runwayml/stable-diffusion-v1-5"
        # create pipe
        pipe = Prompt2ImPipe(model)
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

        pipe = Prompt2ImPipe(model)
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


if __name__ == '__main__':
    unittest.main()
