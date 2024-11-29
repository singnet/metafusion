import sys
import torch
import psutil
from PIL import Image
import copy as cp
import optimum.quanto
from optimum.quanto import freeze, qfloat8, quantize as _quantize
from diffusers.utils import is_accelerate_available

import logging


if is_accelerate_available():
    from accelerate import init_empty_weights
else:
    init_empty_weights = nullcontext

def create_exif_metadata(im: Image, custom_metadata):
    exif = im.getexif()
    # Encode the custom metadata as bytes (convert to utf-8 if needed)
    custom_metadata_bytes = custom_metadata.encode('utf-8')

    # Add the custom metadata to the EXIF dictionary under the "UserComment" tag
    # https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif/usercomment.html
    exif[0x9286]  = custom_metadata_bytes
    # there is no api to set exif in loaded image it seems
    return exif


def pad_image_to_multiple_of_8(image: Image) -> Image:
    """
    Pads the input image by repeating the bottom or right-most column of pixels
    so that the height and width of the image is divisible by 8.

    Args:
        image (Image): The input PIL image.

    Returns:
        Image: The padded PIL image.
    """

    # Calculate the new dimensions
    new_width = (image.width + 7) // 8 * 8
    new_height = (image.height + 7) // 8 * 8

    # Create a new image with the new dimensions and paste the original image onto it
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (0, 0))

    # Repeat the right-most column of pixels to fill the horizontal padding
    for x in range(new_width - image.width):
        box = (image.width + x, 0, image.width + x + 1, image.height)
        region = image.crop((image.width - 1, 0, image.width, image.height))
        padded_image.paste(region, box)

    # Repeat the bottom-most row of pixels to fill the vertical padding
    for y in range(new_height - image.height):
        box = (0, image.height + y, image.width, image.height + y + 1)
        region = image.crop((0, image.height - 1, image.width, image.height))
        padded_image.paste(region, box)

    return padded_image


def count_params(model):
    total_size = sum(param.numel() for param in model.parameters())
    mul = 2
    if model.dtype in (torch.float16, torch.bfloat16):
        mul = 2
    elif model.dtype == torch.float32:
        mul = 4
    return total_size * mul


def get_size(obj):
    return sys.getsizeof(obj)


def get_model_size(pipeline):
    total_size = 0
    for name, component in pipeline.components.items():
        if isinstance(component, torch.nn.Module):
            total_size += count_params(component)
        elif hasattr(component, 'tokenizer'):
            total_size += count_params(component.tokenizer)
        else:
            total_size += get_size(component)
    return total_size / (1024 * 1024)


def awailable_ram():
    mem = psutil.virtual_memory()
    available_ram = mem.available
    return available_ram / (1024 * 1024)


def quantize(pipe, dtype=qfloat8):
    components = ['unet', 'transformer', 'text_encoder', 'text_encoder_2', 'vae']
    
    for component in components:
        if hasattr(pipe, component):
            component_obj = getattr(pipe, component)
            _quantize(component_obj, weights=dtype)
            freeze(component_obj)
            # Add attributes to indicate quantization
            component_obj._is_quantized = True
            component_obj._quantization_dtype = dtype


def weightshare_copy(pipe):
    """
    Create a new pipe object then assign weights using load_state_dict from passed 'pipe'
    """
    copy = pipe.__class__(**pipe.components)
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        for key, component in copy.components.items():
            if getattr(copy, key) is None:
                continue
            if key in ('tokenizer', 'tokenizer_2', 'feature_extractor'):
                setattr(copy, key, cp.deepcopy(getattr(copy, key)))
                continue
            cls = getattr(copy, key).__class__
            if hasattr(cls, 'from_config'):
                setattr(copy, key, cls.from_config(getattr(copy, key).config))
            else:
                setattr(copy, key, cls(getattr(copy, key).config))

            pipe_component = getattr(pipe, key)
            if getattr(pipe_component, '_is_quantized', False):
                # Quantize the component in the copy using the same dtype
                component_obj = getattr(copy, key)
                _quantize(component_obj, weights=pipe_component._quantization_dtype)
    # assign=True is needed since our copy is on "meta" device, i.g. weights are empty
    for key, component in copy.components.items():
        if key == 'tokenizer' or key == 'tokenizer_2':
            continue
        obj = getattr(copy, key)
        if hasattr(obj, 'load_state_dict'):
            obj.load_state_dict(getattr(pipe, key).state_dict(), assign=True)
    # some buffers might not be transfered from pipe to copy
    copy.to(pipe.device)
    return copy
