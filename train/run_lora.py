from diffusers import AutoPipelineForText2Image
from diffusers.schedulers import DPMSolverMultistepScheduler

import torch


model_name = '/home/imgen/models/SDXL/juggernautXL_v8Rundiffusion/'
pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16,  custom_pipeline="lpw_stable_diffusion_xl").to("cuda")


pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

lora_dir = './digitalben-lora/checkpoint-20000/'
pipeline.load_lora_weights(lora_dir)
pipeline.vae.enable_tiling()
# pipeline.enable_xformers_memory_efficient_attention()

prompt = "digitalben is hugging his lamb, farm in background, animal photography, big sheep, full-height photo, best quality, camera low, camera close, best quality, amazing,ultra high res, masterpiece, round glasses, long hair"
negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation" 

generator = torch.Generator('cuda').manual_seed(4394893432342354523) 
images = [pipeline(prompt, num_inference_steps=50, generator=generator, negative_prompt=negative_prompt, guidance_scale=4, cross_attention_kwargs={"scale": 0.9}).images[0] for _ in range(10)]

i = 0
for im in images:
    im.save(f'bens/ben{i}.jpeg')
    i += 1

