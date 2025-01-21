import sys
import torch
import psutil
from PIL import Image
import copy as cp
from inspect import signature
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


def pad_image_to_multiple(image: Image, padding_size: int = 8) -> Image:
    """
    Pads the input image by repeating the bottom and right-most rows and columns of pixels
    so that its dimensions are divisible by 'padding_size'.

    Args:
        image (Image): The input PIL Image.
        padding_size (int): The multiple to which dimensions are padded.

    Returns:
        Image: The padded PIL Image.
    """
    # Calculate the new dimensions
    new_width = ((image.width + padding_size - 1) // padding_size) * padding_size
    new_height = ((image.height + padding_size - 1) // padding_size) * padding_size

    # Calculate padding amounts
    pad_right = new_width - image.width
    pad_bottom = new_height - image.height

    # Create a new image with the new dimensions
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (0, 0))

    # Check if padding is needed
    if pad_right > 0 or pad_bottom > 0:
        # Get the last column and row
        if pad_right > 0:
            last_column = image.crop((image.width - 1, 0, image.width, image.height))
            # Resize the last column to fill the right padding area
            right_padding = last_column.resize((pad_right, image.height), Image.NEAREST)
            padded_image.paste(right_padding, (image.width, 0))

        if pad_bottom > 0:
            last_row = image.crop((0, image.height - 1, image.width, image.height))
            # Resize the last row to fill the bottom padding area
            bottom_padding = last_row.resize((image.width, pad_bottom), Image.NEAREST)
            padded_image.paste(bottom_padding, (0, image.height))

        if pad_right > 0 and pad_bottom > 0:
            # Fill the bottom-right corner
            last_pixel = image.getpixel((image.width - 1, image.height - 1))
            corner = Image.new(image.mode, (pad_right, pad_bottom), last_pixel)
            padded_image.paste(corner, (image.width, image.height))

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


IS_QUANTIZED = '_is_quantized'


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
            if getattr(pipe_component, IS_QUANTIZED, False):
                # Quantize the component in the copy using the same dtype
                component_obj = getattr(copy, key)
                _quantize(component_obj, weights=pipe_component._quantization_dtype)
                setattr(component_obj, IS_QUANTIZED, True)
                component_obj._quantization_dtype = pipe_component._quantization_dtype
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


def get_allowed_components(cls: type) -> dict:
    params = signature(cls.__init__).parameters
    components = [name for name in params.keys() if name != 'self']
    return components
