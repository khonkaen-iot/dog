from PIL import Image

def process_image(filename):
    #load image by PIL
    img = Image.open(filename)

    #Image processing
    processed_img = img.rotate(90) # rotate image

    processed_img.save(filename)

    return filename
