import torch
from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Cond2ImPipe, ModelType, CIm2ImPipe


cnet_ids = ["InstantX/FLUX.1-dev-Controlnet-Canny-alpha"]

# comment this line if FLUX.1-dev-Controlnet-Canny-alpha is not saved locally
cnet_ids = ["/home/imgen/models/ControlNetFlux/FLUX.1-dev-Controlnet-Canny-alpha/"]
ctypes = ["canny"]
device = torch.device('cpu')
offload_device = 0
model_id = '/home/imgen/models/flux-1-dev'
pipe = CIm2ImPipe(model_id, ctypes=ctypes, cnet_ids=cnet_ids, device=device, offload_device=offload_device)


prompt = ["bioinformatics lab with flasks and exotic flowers",
           "happy vibrant", "green colors", "artwork", "high tech"]

nprompt = "jpeg artifacts, blur, distortion, watermark, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy"

pipe.setup("./_projects/biolab/00000.png", strength=1, steps=25, cscales=0.8)

gs = GenSession("./_projects/biolab/controlnet-flux/", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=10)
