from PIL import Image


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

