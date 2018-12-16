from data_stats.character_counts import character_counts, total_characters, total_image_bytes, unique_character_class_count
from data_management.formatted_data import images_classes_zip
from data_visualization.render_character import render_in_terminal
# import numpy
import pdb
from random import randint

print('total characters = ', total_characters)
print('total image bytes = ', total_image_bytes)

print('number of unique character classes = ', unique_character_class_count)

for character, count in character_counts.most_common():
    print(character, ' has ', count, ' occurrences')

print('*five random images with classes*')
five_random_images = list(map(
    lambda index: images_classes_zip[index],
    [ randint(0, total_characters) for _ in range(2) ]
))

for image_28_by_28, character_class in five_random_images:
    print('character', character_class, '\n')
    render_in_terminal(image_28_by_28)

