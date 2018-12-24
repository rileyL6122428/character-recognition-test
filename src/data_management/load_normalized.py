import _pickle as pickle

# character_flattened_normalized_images = pickle.load(open(
#     '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/character_flattened_images_normalized.p',
#     'rb'
# ))

normalized_images = []

for index in range(3200):
    print('progress: ', (index / 3200) * 100 ,'%')
    next_batch = pickle.load(open(
        '/Users/rileylittlefield/Desktop/classify_chars_ml/data/python/normalized/character_flattened_images_%s_.p' % index,
        'rb'
    ))
    for normalized_image in next_batch:
        normalized_images.append(normalized_image)
        
