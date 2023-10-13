from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Cond2ImPipe, ControlnetType


nprompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
prompt = "a fire whirlpool"

model_dir = "./models-sd/"

# diffusers format or safetensors
model_id = "icbinp"
model_id = "icbinpICantBelieveIts_v8.safetensors"

# diffusers format or safetensors
model_id = "./stabilityai/stable-diffusion-xl-base-1.0"
model_id = "./Stable-diffusion/sdXL_v10VAEFix.safetensors"

pipe = Cond2ImPipe(model_dir + model_id, ctypes=["qr"], model_type=ControlnetType.SDXL)
pipe.setup("./spiral.png", width=768, height=512, cscales=[0.3])
gs = GenSession("./_projects/cnet", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=5)

