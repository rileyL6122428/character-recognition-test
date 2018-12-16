from data_management.formatted_data import images_28_by_28, character_classes
from matplotlib import pyplot, cm
import numpy

def render_with_pyplot(image_28_by_28):
    xs, ys, colors = [], [], []

    for row_index in range(len(image_28_by_28)):
        for col_index in range(len(image_28_by_28[row_index])):
            xs.append(row_index)
            ys.append(28 - col_index)

            color = image_28_by_28[row_index][col_index]
            colors.append(color)
    
    pyplot.scatter(xs, ys, c=colors, cmap=cm.Greys)
    pyplot.show()

def render_in_terminal(image_28_by_28):
    reversed_rows = numpy.transpose(image_28_by_28)
    for row in reversed_rows:
        next_row = []
        for pixel in row:
            next_row.append('x') if pixel > 25 else next_row.append(' ')
        print(''.join(next_row))

