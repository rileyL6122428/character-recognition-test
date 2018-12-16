import numpy
from data_management.load_serialized import  character_flattened_images as unwrapped_flat_images, character_number_classes as unwrapped_number_classes

flat_images = numpy.array(unwrapped_flat_images)

images_28_by_28 = numpy.array([
    numpy.reshape(image, (28, 28))
    for image
    in unwrapped_flat_images
])

character_classes = numpy.array([
    chr(character_number + 96)
    for character_number
    in unwrapped_number_classes
])

number_classes = numpy.array(unwrapped_number_classes)

images_classes_zip = numpy.array(list(zip(images_28_by_28, character_classes)))
