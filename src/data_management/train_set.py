from data_management.test_train_indices import train_indices
from data_management.formatted_data import character_classes, flat_images, images_28_by_28, number_classes

character_classes = character_classes[train_indices]
number_classes = number_classes[train_indices]
flat_images = flat_images[train_indices]
images_28_by_28 = images_28_by_28[train_indices]
