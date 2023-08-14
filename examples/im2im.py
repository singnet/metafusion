from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Im2Im, ControlNet2Im, get_diffusion_scheduler_names


model_dir = "./models-sd/"
model_id = "icbinp"

nprompt = "jpeg artifacts, blur, distortion, watermark, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy"
prompt = ["bioinformatics lab with flasks and exotic flowers",
          "happy vibrant", "green colors", "artwork", "high tech"]

#pipe = ControlNet2Im(model_dir+model_id)

pipe = Im2Im(model_dir+model_id)
pipe.setup("./_projects/biolab/00000.png", strength=0.95, scheduler=get_diffusion_scheduler_names()[0])
gs = GenSession("./_projects/biolab/modified/", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=10)

