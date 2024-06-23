from multigen import GenSession, Cfgen
from multigen.pipes import MaskedIm2ImPipe, ModelType
import PIL.Image
import numpy


def main():
    model_id = 'frankjoshua/juggernautXL_v8Rundiffusion'
    model_id = '/home/imgen/models/SDXL/juggernautXL_v8Rundiffusion.safetensors'
    img = PIL.Image.open("./mech_beard_sigm.png")
    # read image with mask painted over
    img_paint = numpy.array(PIL.Image.open("./mech_beard_sigm_mask.png"))

    scheduler = "EulerAncestralDiscreteScheduler" # gives good results

    pipe = MaskedIm2ImPipe(model_id, model_type=ModelType.SDXL)
    blur = 48
    pipe.setup(original_image=img, image_painted=img_paint, strength=0.96,
               scheduler=scheduler, guidance_scale=7, clip_skip=0, blur=blur, blur_compose=3, steps=50, sample_mode='random')

    prompt = "a man wearing a mask"
    gs = GenSession("./masked_im2im_xl", pipe, Cfgen(prompt, "", seeds=range(0,10)))
    gs.gen_sess(add_count=10)


if __name__ == "__main__":
    main()
