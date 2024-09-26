from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import CInpaintingPipe

from PIL import Image

model_dir = "./models-sd/"
model_id = "icbinp"

nprompt = "jpeg artifacts, blur, distortion, watermark, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy"
prompt = ["spider on the biolab ceiling",
          "happy vibrant", "green colors", "artwork", "high tech"]

mask_image = Image.open("mask_up.png").resize((768, 768))
pipe = CInpaintingPipe(model_dir+model_id)
pipe.setup("./_projects/biolab/00000.png", mask_image)
gs = GenSession("./_projects/biolab/inpaint/", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=5)
