from scipy.io import loadmat

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

# some_flattened_chars = characters_data_set[0][0][0][0][0][0][0]
# print(some_flattened_chars) 
# print(len(some_flattened_chars)) # => 28 x 28 

some_character_classes = characters_data_set[0][0][0][0][0][1]
print(some_character_classes)
print('len', len(some_character_classes))  # => 124800

print({
    wrapped_character_class[0]
    for wrapped_character_class
    in some_character_classes
})
