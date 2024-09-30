import torch
from diffusers import FluxControlNetModel, FluxControlNetInpaintPipeline 
from diffusers import FluxInpaintPipeline 
from multigen.pipes import InpaintingPipe
from diffusers.utils import load_image, make_image_grid

# requires https://huggingface.co/black-forest-labs/FLUX.1-dev
model_id = "/home/imgen/models/flux-1-dev"
pipe = InpaintingPipe(model_id, 
        device=torch.device('cpu', 0), offload_device=0, torch_dtype=torch.bfloat16)

# load base and mask image
init_image = load_image("cr.png")
mask_image = load_image("select.png")

generator = torch.Generator("cuda").manual_seed(92)
prompt = "a football player holding a gun, pointing it towards viewer"
pipe.setup(image=init_image, mask=mask_image, guidance_scale=7,
        strength=0.9, steps=30)
image = pipe.gen(dict(prompt=prompt, generator=generator))
image.save('result0.png')
