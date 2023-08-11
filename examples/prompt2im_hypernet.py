from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe


prompt = "photo of a (digitalben:1.1) farmer, man is on a farm next to his horse, 24mm, 4k textures, soft cinematic light, RAW photo, photorealism, photorealistic, highly detailed, sharp focus, soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded" 

negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation" 


path = "./models/icb_diffusers_final/"
path_hypernet = './models/hypernetworks/digitalben.pt'
pipe = Prompt2ImPipe(path, lpw=True)
pipe.setup(width=512, height=512, guidance_scale=5.5, clip_skip=1, scheduler='DPMSolverMultistepScheduler')
pipe.add_hypernet(path_hypernet, 0.65)
# pipe.clear_hypernets()

gs = GenSession("./benben", pipe, Cfgen(prompt, negative_prompt, seeds=[1877029948]))
gs.gen_sess(add_count=1)
