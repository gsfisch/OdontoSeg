from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

old_path = "new_dataset/Images/N-260-01_old.jpg"
new_path = "new_dataset/Images/N-260-01.jpg"

with Image.open(old_path) as im:
    im = im.convert("RGB")
    im.save(new_path, quality=95)