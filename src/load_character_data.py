from scipy.io import loadmat
import numpy

characters_mat_dict = loadmat('/Users/rileylittlefield/Desktop/classify_chars_ml/data/matlab/emnist-letters.mat')
# print('characters_mat_dict', characters_mat_dict)
# print('dir(characters_mat_dict)', dir(characters_mat_dict))

# count = 0
# for key, value in characters_mat_dict.items():
#     print('KEY = ', key)
#     print('VALUE = ', value)

characters_data_set = characters_mat_dict.get('dataset')
# print(len(characters_data_set)) # = 1
# print(len(characters_data_set[0])) # = 1
# print(len(characters_data_set[0][0])) # = 3
# print(characters_data_set[0][0]) # = 3
# print(characters_data_set[0][0][0])
# print(characters_data_set[0][0][0][0])
# print(characters_data_set[0][0][0][0][0])
# print(characters_data_set[0][0][0][0][0][0])
# print(len(characters_data_set[0][0][0][0][0][0])) # => 124800

# SAMPLES
def as_28_by_28_image(sample):
    return numpy.reshape(sample, (28, 28))

character_images = characters_data_set[0][0][0][0][0][0]

characters_images_28_by_28 = [
    as_28_by_28_image(image)
    for image
    in character_images
]



# flattend_char = characters_data_set[0][0][0][0][0][0][0]
# flattend_char_2 = characters_data_set[0][0][0][0][0][0][1]
# print(flattend_char)
# print(as_28_by_28_image(flattend_char))

# some_flattened_chars = characters_data_set[0][0][0][0][0][0][0]
# print(some_flattened_chars) 
# print(len(some_flattened_chars)) # => 28 x 28 


# CLASSES

unmapped_char_classes = characters_data_set[0][0][0][0][0][1]

def classes_as_numbers(unmapped_classes):
    return [
        unmapped_class[0]
        for unmapped_class
        in unmapped_classes
    ]

def classes_as_characters(unmapped_classes):
    classes = classes_as_numbers(unmapped_classes)
    return [
        chr(character_number + 96)
        for character_number
        in classes
    ]

number_classes = classes_as_numbers(unmapped_char_classes)
character_classes = classes_as_characters(unmapped_char_classes)

