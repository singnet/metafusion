"""
Example of using Hyper-SD with metafusion


# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
requires this model to be in models-sd/SDXL/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors

and https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-4steps-lora.safetensors to be in

models-sd/Lora/Hyper-SDXL-4steps-lora.safetensors
"""

import time
from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe

from multigen.worker import ServiceThread


nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

prompt = "Close-up portrait of a woman wearing suit posing with black background, rim lighting, octane, unreal"
seed = 383947828373273


cfg_file = 'config.yaml'

worker = ServiceThread(cfg_file)
worker.start()

pipe = "prompt2image" 
model = 'sdxl1'
result = worker.open_session(
    user='test',
    project="results",
    model=model,
    pipe=pipe,
    loras=['hypersd4steps'],
)

count = 5 
c = 0
def on_new_image(*args, **kwargs):
    print(args, kwargs)
    global c
    c += 1

if 'error' not in result:
	sess_id = result['session_id']
	worker.queue_gen(session_id=sess_id, 
					images=None,
					prompt=prompt, pipe='Prompt2ImPipe',
                    image_callback=on_new_image,
				    lpw=True,
					width=1024, height=1024, steps=4, 
					timestep_spacing='trailing', 
					guidance_scale=0, 
					scheduler='EulerAncestralDiscreteScheduler',
                    count=count,
                    seeds=[seed + i for i in range(count)],
                    )

while count != c:
    time.sleep(1)
worker.stop()
