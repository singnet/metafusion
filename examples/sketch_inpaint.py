from multigen import GenSession, Cfgen
from multigen.pipes import MaskedIm2ImPipe
import PIL.Image
import numpy


def main():
    blur = 8
    model_dir = "./models-sd/"
    model_id = "icbinp"  # i can't belive it's not a photography
    img = PIL.Image.open("./mech_beard_sigm.png")
    # read image with mask painted over
    img_paint = numpy.array(PIL.Image.open("./mech_beard_sigm_mask.png"))

    scheduler = 'DPMSolverMultistepScheduler'
    scheduler = "EulerAncestralDiscreteScheduler" # gives good results

    pipe = MaskedIm2ImPipe(model_dir+model_id)
    pipe.setup(original_image=img, image_painted=img_paint, strength=0.75,
               scheduler=scheduler, guidance_scale=7, clip_skip=3, blur=blur)

    prompt = "a man wearing a mask"
    gs = GenSession("./masked_im2im", pipe, Cfgen(prompt, "", seeds=range(10)))
    gs.gen_sess(add_count=10)


if __name__ == "__main__":
    main()