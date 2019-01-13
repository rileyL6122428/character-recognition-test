import numpy

def as_grayscale_list(rgb_list):
    return [
        255 - ((red + green + blue) / 3)
        for (red, green, blue, alpha)
        in rgb_list
    ]


def as_28_by_28(grayscale_list):
    return numpy.transpose(
        numpy.reshape(grayscale_list, (28, 28))
    )


def flatten(image):
    flattened_image = []

    for row in image:
        for gray_scale_val in row:
            flattened_image.append(gray_scale_val)

    return flattened_image
