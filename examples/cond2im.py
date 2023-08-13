from multigen.prompting import Cfgen
from multigen.sessions import GenSession
from multigen.pipes import Cond2Im


nprompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
prompt = "child in the coat playing in sandbox"

model_dir = "./models-sd/"
model_id = "icbinp"

pipe = Cond2Im(model_dir + model_id, ctypes=["pose"])
pipe.setup("./pose6.jpeg", width=768, height=768)
gs = GenSession("./_projects/cnet", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=5)

