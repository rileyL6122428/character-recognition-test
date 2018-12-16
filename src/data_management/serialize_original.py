from scipy.io import loadmat
import numpy
import _pickle as pickle

# ORIGINAL DATA SET
characters_mat_dict = loadmat('/Users/rileylittlefield/Desktop/classify_chars_ml/data/matlab/emnist-letters.mat')
characters_data_set = characters_mat_dict.get('dataset')

# IMAGES
character_images = characters_data_set[0][0][0][0][0][0]
pickle.dump(character_images, open(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/character_flattened_images.p',
    'wb'
))

# IMAGE CLASSES
character_number_classes = [
    unmapped_class[0]
    for unmapped_class
    in characters_data_set[0][0][0][0][0][1]
]
pickle.dump(character_number_classes, open(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/character_number_classes.p',
    'wb'
))
