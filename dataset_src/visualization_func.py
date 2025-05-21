import matplotlib.pyplot as plt
from PIL import Image
def show_image_with_caption(image_path, caption):
    """Display images and generated descriptions"""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption, wrap=True)
    plt.show()
