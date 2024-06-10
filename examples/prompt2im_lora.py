from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe
from diffusers import DPMSolverMultistepScheduler


prompt = "digitalben is hugging his lamb, farm in background, animal photography, big sheep, full-height photo, best quality, camera low, camera close, best quality, amazing,ultra high res, masterpiece, round glasses, long hair"


negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.2), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation" 


model_name = '~/models/SDXL/juggernautXL_v8Rundiffusion/'

path_lora = 'checkpoint-20500'
pipe = Prompt2ImPipe(model_name, lpw=True)
pipe.setup(width=1024, height=1024, guidance_scale=4, clip_skip=1)
pipe.load_lora(path_lora, 0.9)

gs = GenSession("./benben", pipe, Cfgen(prompt, negative_prompt, seeds=[1877029948 + i for i in range(10)]))
gs.gen_sess(add_count=10)
