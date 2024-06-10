from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Prompt2ImPipe

model_dir = "./models-sd/"
model_id = "icbinp"

nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, bad hands, duplicate heads, bad anatomy, bad crop"

prompt = [["+", ["bioinformatics", "bio-tech"], "lab", "with",
                ["computers", "flasks"], {"w": "and exotic flowers", "p": 0.5}],
          "happy vibrant",
          ["green colors", "dream colors", "neon glowing"],
          ["8k RAW photo, masterpiece, super quality", "artwork", "unity 3D"],
          ["surrealism", "impressionism", "high tech", "cyberpunk"]]


pipe = Prompt2ImPipe(model_dir+model_id, lpw=True)
pipe.setup(width=768, height=768)
gs = GenSession("./_projects/biolab", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=10)

