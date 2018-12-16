import _pickle as pickle

character_number_classes = pickle.load(open(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/character_number_classes.p',
    'rb'
))

character_flattened_images = pickle.load(open(
    '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/character_flattened_images.p',
    'rb'
))
