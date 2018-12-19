from sklearn.model_selection import StratifiedShuffleSplit
from data_management.formatted_data import character_classes, number_classes, flat_images, images_28_by_28

splitter = StratifiedShuffleSplit(train_size=0.85, random_state=42, n_splits=1)
 
train_indices_placeholder, test_indices_placeholder = [], []

for train_split, test_split in splitter.split(flat_images, number_classes):
    train_indices_placeholder = train_split
    test_indices_placeholder = test_split

train_indices, test_indices = train_indices_placeholder, test_indices_placeholder
