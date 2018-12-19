from data_management.test_train_indices import test_indices
from data_management.formatted_data import character_classes, flat_images, images_28_by_28, number_classes

character_classes = character_classes[test_indices]
number_classes = number_classes[test_indices]
flat_images = flat_images[test_indices]
images_28_by_28 = images_28_by_28[test_indices]
