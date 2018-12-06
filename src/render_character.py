from load_character_data import characters_data_set, characters_images_28_by_28, character_classes
from matplotlib import pyplot, cm
import numpy

# pyplot

def render_image(image):
    xs, ys, colors = [], [], []

    for row_index in range(len(image)):
        for col_index in range(len(image[row_index])):
            xs.append(row_index)
            ys.append(28 - col_index)

            color = image[row_index][col_index]
            colors.append(color)
    
    pyplot.scatter(xs, ys, c=colors, cmap=cm.Greys)
    pyplot.show()

def render_in_terminal(image):
    reversed_rows = numpy.transpose(image)
    for row in reversed_rows:
        next_row = []
        for pixel in row:
            next_row.append('x') if pixel > 5 else next_row.append(' ')
        print(''.join(next_row))


for image, image_class in zip(characters_images_28_by_28[:25], character_classes[:25]):
    print('class: ', image_class)
    # render_image(image)
    render_in_terminal(image)
