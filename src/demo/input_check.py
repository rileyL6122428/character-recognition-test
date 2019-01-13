from PIL import Image
from data_management.train_set import images_28_by_28, character_classes
from data_visualization.render_character import render_in_terminal
from demo_adapters import as_28_by_28, as_grayscale_list, flatten

test_g_image = None
for image, label in zip(images_28_by_28, character_classes):
    if label == 'b':
        test_g_image = image
        break

render_in_terminal(test_g_image)

my_g_image = as_28_by_28(as_grayscale_list(Image.open(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/src/demo/riley-B-cap.png'
).getdata()))

render_in_terminal(my_g_image)
