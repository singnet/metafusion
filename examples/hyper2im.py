"""
Example of using Hyper-SD with metafusion, see https://huggingface.co/ByteDance/Hyper-SD
"""

from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe

model_dir = "./models-sd/"

# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
model_id = "SDXL/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
# https://civitai.com/models/133005?modelVersionId=471120
model_id = "SDXL/juggernautXL_v8Rundiffusion.safetensors"
# https://civitai.com/models/410737/neta-art-xl
model_id = "SDXL/netaArtXL_v10.safetensors"


nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

prompt = "Close-up portrait of a woman wearing suit posing with black background, rim lighting, octane, unreal"
seed = 383947828373273
pipe = Prompt2ImPipe(model_dir+model_id, lpw=False)

# base SDXL and juggernautXL works with steps=5, guidance_scale=0
pipe.setup(width=1024, height=1024, steps=10, timestep_spacing='trailing', guidance_scale=4, scheduler='EulerAncestralDiscreteScheduler')
pipe.load_lora("./models-sd/Lora/Hyper-SDXL-4steps-lora.safetensors")
gs = GenSession("./_projects/timeste", pipe, Cfgen(prompt, nprompt, seeds=list(range(seed, seed+10))))
gs.gen_sess(add_count=10)
