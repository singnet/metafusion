from PIL import Image

def save_metadata(image_path, custom_metadata):
    im = Image.open(image_path)
    exif = im.getexif()

    # Encode the custom metadata as bytes (convert to utf-8 if needed)
    custom_metadata_bytes = custom_metadata.encode('utf-8')

    # Add the custom metadata to the EXIF dictionary under the "UserComment" tag
    exif[0x9286]  = custom_metadata_bytes

    # Convert the EXIF dictionary back to bytes and save it to the image
    im.save(image_path, exif=exif)
