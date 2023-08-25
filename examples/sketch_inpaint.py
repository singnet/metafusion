from PIL import ImageFilter

from multigen.pipes import MaskedIm2ImPipe
import PIL.Image
import numpy
import torch
import random


def main():
    blur = 8
    model_dir = "./models-sd/"
    model_id = "icb_diffusers_final"
    img = PIL.Image.open("./mech_beard_sigm.png")
    # read image with mask painted over
    img_paint = numpy.array(PIL.Image.open("./mech_beard_sigm_mask.png"))

    scheduler = 'DPMSolverMultistepScheduler'
    scheduler = "EulerAncestralDiscreteScheduler"

    pipe = MaskedIm2ImPipe(model_dir+model_id)
    pipe.setup(original_image=img, image_painted=img_paint, strength=0.75,
               scheduler=scheduler, guidance_scale=7, clip_skip=3)
    for seed in range(10):
        img_gen = pipe.gen(dict(prompt="a man wearing a mask",
                            generator=torch.cuda.manual_seed(seed)))
        img_gen.save("./result{0}.png".format(seed))


if __name__ == "__main__":
    main()
