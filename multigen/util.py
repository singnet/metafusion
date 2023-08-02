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
