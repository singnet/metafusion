
# metafusion library

# Pipes classes

Pipes classes implement different ways to generate 
or process images using diffusion models.


All Pipe classes have two methods: *setup* and *gen*.

*setup*'s purpuse is define parameters of the pipeline for image generation. Pipelines that take an image as an 
input perform image preprocessing in *setup*. Setup's arguments are stored in the pipeline and used whenever *gen* method is used.

*gen* takes as an input parameters are not stored inside the pipeline. All the pipelines take a dictionary as an input.
Expected parameters are prompt(str), generator (torch.Generator), negative_prompt (str).

**Prompt2ImPipe** is a pipe that generates an image from text prompt.


```example
pipe = Prompt2ImPipe(model_id='runwayml/stable-diffusion-v1-5')
pipe.setup(width=768, height=768, clip_skip=2)
image = pipe.gen({'prompt': 'bio-tech lab with computers and exotic flowers, artwork'})
image.save('bio.png')
```

*setup* parameters

width - width of image to generate  
heigth - height of image to generate  
guidance_scale - strength of the prompt influence on generation process.  
steps - The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. Default value is 50.  
clip_skip - Number of layers to be skipped from CLIP while computing the prompt embeddings. Skipping some layers gives less precise representation of the prompt. Default value is 0.  

Optimal values of guidance_scale and steps vary a lot between different checkpoints.


*gen* parameters


***Im2ImPipe** is a pipe that generates an image from another image.

```
pipe = Im2ImPipe(model_id='runwayml/stable-diffusion-v1-5')
pipe.setup("./_projects/biolab/00000.png", strength=0.5, steps=25)
img = pipe.gen({'prompt': 'biolab, anime style drawing, green colors'})
img.save('bio1.png')
```

*setup* parameters

fimage - File path to the input image.  
image - Input image. Can be used instead of fimage.  
strength - Strength of image modification. Defaults to 0.75. A lower strength values keep result close to the input image. value of 1 means input image more or less ignored.  
scale -  The scale factor for resizing of the input image. The output image will have dimentions (height * scale, width * scale) Defaults to None.  
guidance_scale, steps, clip_skip - same as in Prompt2ImPipe  


**Cond2ImPipe** is a pipe that generates 
an image from a special conditioning 
image e.g.image after canny edge detection, or etc.  
Conditioning image is processed internally with controlnet and  
uses StableDiffusion(XL)ControlNetPipeline

Models are expected to be in ./models-cn/ for StableDiffusion 
and in ./models-cn-xl/ for StableDiffusionXL


**CIm2ImPipe** is a subclass of Cond2ImPipe.  

The difference is that the conditional image is not
taken as input but is obtained from the input image, which
is processed internally by the image processor. The image processor   
depends on the conditioning type specified in the constructor.



*setup* parameters

fimage, image - same as in Im2ImPipe
cscales - strength of control image influence  
width, height, steps, clip_skip, guidance_scale - same as in Prompt2ImPipe

```
model_id = 'runwayml/stable-diffusion-v1-5'
pipe = CIm2ImPipe(model_id, model_type=ModelType.SD)
pipe.setup("./bio1.png", strength=0.5, steps=25, ctypes=['soft'])
img = pipe.gen({'prompt': 'biolab, anime style drawing, green colors'})
img.save('bio2.png')
```

possible values for ctypes:
* 'canny' - canny edge detection
* 'soft-sobel' - sobel edge detection
* 'soft' - same as soft-sobel with different edge detector.	 
* 'depth' - depth map
* 'pose' - A OpenPose bone image.
* 'ip2p' - original image will be used as control
* 'inpaint' - original image will be thresholded and inpainted(use InpaintingPipe for this option)
* 'qr' - original image will be used as control.


**MaskedIm2ImPipe** is image to image pipeline that uses mask to redraw only certain parts of the input image.
It can be used as an inpainting pipeline with any non-inpaint models.
The pipeline computes mask from the difference between  
original image and image with a mask on it. Color of the mask affects the result.

```
blur = 8
model_id = 'runwayml/stable-diffusion-v1-5'
img = PIL.Image.open("./mech_beard_sigm.png")
# read image with mask painted over
img_paint = numpy.array(PIL.Image.open("./mech_beard_sigm_mask.png"))
scheduler = "EulerAncestralDiscreteScheduler" # gives good results

pipe = MaskedIm2ImPipe(model_dir+model_id)
pipe.setup(original_image=img, image_painted=img_paint, strength=0.85,
           scheduler=scheduler, guidance_scale=7, clip_skip=3, blur=blur)

prompt = "a man wearing a mask"
img = pipe.gen({'prompt': prompt, 'seed':84958344})
img.save('inpaint1.png')
```

*setup* parameters

original_image - image without mask
image_painted - modified version of original_image, this parameter should be skipped if mask is passed.  
mask - The mask. Defaults to None. If None it will be computed from the difference 
between original_image and image_painted, should be skipped if image_painted is passed  
blur - The blur radius for the mask to apply for generation process.  
blur_compose - The blur radius for composing the original and generated images.  
scale - The scale factor for resizing of the input image. The output image will have dimentions (height * scale, width * scale)

## metafusion service

**ServiceThread** from multigen.worker module implements  
generation queue that can be used to implement e.g. web service.

This class needs two config files, one(config.yaml in our example) that specifies directories to use in ServiceThread

*config.yaml*
```config.yaml
model_list: models.yaml   # 
model_dir: ./models-sd/
logging_folder: ./logs/
```
Another file *models.yaml* specifies models and pipelines.

```
base:
  sdxl1:
    # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors
    id: SDXL/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors
    xl: True
lora:
  # https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SDXL-4steps-lora.safetensors
  hypersd4steps:
    id: Hyper-SDXL-4steps-lora.safetensors
    xl: True

pipes:
    prompt2image:
        name: "Prompt to Image"
        classname: Prompt2ImPipe
        need_imgs: False
        needs_cnet: False
```


Before scheduling image generation one needs to create a session. Session determines where
generation results are stored both inside the ServiceThread and on filesystem.
open_session returns a dictionary containing "session_id" which can  
be used later to query generation results.

This argument are expected by **open_session**:  
*user* - user name  
*project* - project name, determines where generated results are stored.  
*pipe* - The name of the pipe to use for the session, the one specified in models config
file under "pipes" field.  
*model* - The name of the model to use for the session, the one specified in config
file under "base" field.  
*loras* - list of lora models to load, loras must be present in models config under "lora" field.

**queue_gen** is used to schedule generation of images. 
These keyword arguments are expected:
*session_id* - one of the session ids returned by open_session  
*count* - number of images to generate  
*image_callback* - callback function that accepts one argument - path to generated image.  
Other arguments are passed as is to setup() method of the pipeline.
and *prompt* is passed to gen().

full example:
```
cfg_file = 'config.yaml'

worker = ServiceThread(cfg_file)
worker.start()

pipe = "prompt2image"
model = 'sdxl1'
result = worker.open_session(
    user='test',
    project="results",
    model=model,
    pipe=pipe,
    loras=['hypersd4steps'],
)

count = 5
c = 0
def on_new_image(*args, **kwargs):
    print(args, kwargs)
    global c
    c += 1

if 'error' not in result:
    sess_id = result['session_id']
    worker.queue_gen(session_id=sess_id,
                    images=None,
                    prompt=prompt,
                    image_callback=on_new_image,
                    lpw=True,
                    width=1024, height=1024, steps=4,
                    timestep_spacing='trailing',
                    guidance_scale=0,
                    scheduler='EulerAncestralDiscreteScheduler',
                    count=count,
                    seeds=[seed + i for i in range(count)],
                    )

```

**GenSession**

This class together with *Cfgen* is used by *ServiceThread* 
to generate images.


```
nprompt = "jpeg artifacts, blur, distortion, lowres, bad anatomy, bad crop"
prompt = "biological lab with computers and exotic flowers, green colors"
pipe = Prompt2ImPipe(model_id)
pipe.setup(width=768, height=768)
gs = GenSession("./_projects/biolab", pipe, Cfgen(prompt, nprompt))
gs.gen_sess(add_count=10)
```

