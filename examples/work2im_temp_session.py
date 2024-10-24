"""
Example of using Hyper-SD with metafusion


# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
requires this model to be in models-sd/SDXL/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors

and https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-4steps-lora.safetensors to be in

models-sd/Lora/Hyper-SDXL-4steps-lora.safetensors
"""

import time
import random
import PIL.Image
import torch
from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe

from multigen.worker import ServiceThread

from multigen.log import setup_logger
setup_logger()

nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

prompt = "Close-up portrait of a woman wearing suit posing with black background, rim lighting, octane, unreal"
seed = 383947828373273


cfg_file = 'config.yaml'


def random_session(pipe, model):
    id = str(random.randint(0, 1024*1024*1024*4-1))
    session = dict()
    session["images"] = []
    session["user"] = id
    session["pipe"] = pipe
    session["model"] = model
    return { "session_id": id, 'session': session }


worker = ServiceThread(cfg_file)
worker.start()

pipe = "prompt2image" 
pipe = "image2image"
model = list(worker.models['base'].keys())[-1]

count = 5
c = 0
def on_new_image(*args, **kwargs):
    print(args, kwargs)
    print('on new image')
    global c
    c += 1

def on_finish(*args, **kwargs):
    print('finish')
    print(args, kwargs)


#	worker.queue_gen(session_id=sess_id, 
#					images=None,
#					prompt=prompt, pipe='Prompt2ImPipe',
#                    nprompt=nprompt,
#                    image_callback=on_new_image,
#		#		    lpw=False,
#					width=1024, height=1024, steps=4, 
#					guidance_scale=0, 
#                    count=count,
#                    seeds=[seed + i for i in range(count)],
#                    )

generator = torch.Generator().manual_seed(92)
init_image = PIL.Image.open('cr.png')
random_sess = random_session(pipe, model)
#worker.queue_gen(
#        gen_dir='/tmp/img1',
#        image_callback=on_new_image,
#        prompt=prompt, count=count,
#        images=[init_image], generator=generator,
#        num_inference_steps=50, strength=0.82, guidance_scale=3.5,
#        **random_sess)


pipe = "inpaint"
prompt = "a football player holding a gun, pointing it towards viewer"

mask = PIL.Image.open('select.png')
random_sess = random_session(pipe, model)
worker.queue_gen(
        gen_dir='/tmp/img1_inp',
        finish_callback=on_finish,
        image_callback=on_new_image,
        prompt=prompt, count=count,
        image=init_image, mask=mask, generator=generator,
        guidance_scale=7,
        strength=0.9, steps=30, 
        **random_sess)



while count != c:
    time.sleep(1)
worker.stop()
