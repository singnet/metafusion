from multigen.pipes import Im2ImPipe, CIm2ImPipe, get_diffusion_scheduler_names
import PIL.Image
import numpy
import cv2
import torch


def main():
    blur = 8
    model_dir = "./models-sd/"
    model_id = "icb_diffusers_final"


    # read image
    img = PIL.Image.open("./mech_beard_sigm.png")
    img = numpy.array(img)
    # read image with mask painted over
    img_paint = numpy.array(PIL.Image.open("./mech_beard_sigm_mask.png"))

    neq = numpy.any(numpy.array(img) != numpy.array(img_paint), axis=-1)
    mask = neq.astype(numpy.uint8) * 255
    # apply blur to mask
    mask_blur = cv2.blur(mask, (blur, blur))
    img_paint_blur = cv2.blur(img_paint, (blur, blur))
    # mask is height,width but image is height,width,channels, so broadcast the mask
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    img_compose = numpy.where(mask == 0, img, img_paint_blur)
    PIL.Image.fromarray(img_compose).save("./mech_beard_sigm_sketch_compose.png")
    #cv2.imshow("img_compose", img_compose)
    #cv2.imshow("img_paint_blur", img_paint_blur)
    #cv2.imshow("mask_blur", mask_blur)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    seed = 124505997
    pipe = Im2ImPipe(model_dir+model_id)
    pipe.setup(fimage=None, image=img_compose, strength=0.75, gscale=7, clip_skip=3, scheduler='DPMSolverMultistepScheduler')
    img_gen = pipe.gen(dict(prompt="a man wearing a mask", generator=torch.cuda.manual_seed(seed)))
    img_gen.save("./mech_beard_sigm_sketch_gen.png")
    # compose with original using mask
    mask1 = numpy.tile(mask_blur / mask_blur.max(), (3, 1, 1)).transpose(1,2,0)
    img_compose = mask1 * img_gen + (1 - mask1) * img
    # convert to PIL image
    img_compose = PIL.Image.fromarray(img_compose.astype(numpy.uint8))
    img_compose.save("./mech_beard_sigm_sketch.png")

if __name__ == "__main__":
    main()