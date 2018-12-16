from data_management.formatted_data import character_classes, flat_images
from collections import Counter

total_characters = len(flat_images)
total_image_bytes = flat_images.size

character_counts = Counter(character_classes)
unique_character_class_count = len(character_counts.most_common())
