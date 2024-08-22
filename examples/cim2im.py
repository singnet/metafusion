"""
first run promp2im.py

this script expects https://huggingface.co/lllyasviel/control_v11p_sd15_softedge
to be placed in ./models-cn/control_v11p_sd15_softedge


modified versions of the input image will be placed in ./_projects/biolab/controlnet/
"""

from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Im2ImPipe, CIm2ImPipe, ModelType


model_dir = "./models-sd/"
model_id = "icbinp"

prompt = ["bioinformatics lab with flasks and exotic flowers",
           "happy vibrant", "green colors", "artwork", "high tech"]

nprompt = "jpeg artifacts, blur, distortion, watermark, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy"


pipe = CIm2ImPipe(model_dir + model_id, model_type=ModelType.SD, ctypes=['soft'])
pipe.setup("./_projects/biolab/00000.png", strength=0.6, steps=25)
gs = GenSession("./_projects/biolab/controlnet/", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=10)
