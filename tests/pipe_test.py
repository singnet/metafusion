import unittest
import os
import PIL
import torch
import numpy

from multigen.pipes import Prompt2ImPipe


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model_dir = os.getenv('MODEL_DIR')

    def test_basic_txt2im(self):
        model = os.path.join(self.model_dir, 'icb_diffusers_final')
        # create pipe
        pipe = Prompt2ImPipe(model, scheduler="DPMSolverMultistepScheduler")
        pipe.setup(width=512, height=512, guidance_scale=7)
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

    def compute_diff(self, im1: PIL.Image.Image, im2: PIL.Image.Image) -> float:
        # convert to numpy array
        im1 = numpy.asarray(im1)
        im2 = numpy.asarray(im2)
        # compute difference as float
        diff = numpy.sum(numpy.abs(im1.astype(numpy.float32) - im2.astype(numpy.float32)))
        return diff


if __name__ == '__main__':
    unittest.main()
