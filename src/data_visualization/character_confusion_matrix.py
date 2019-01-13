from demo.character_generator import character_generator

def render_confusion_matrix(confusion_matrix):
    characters = list(character_generator())

    print('COLS -> LABELS, ROWS -> PREDICTIONS')

    print('    ', end='')
    for character in characters:
        print(character, '  ,', end='')
    print('')

    for character_index, row in zip(range(26), confusion_matrix):
        print(characters[character_index], '  ', end='')
        for number in row:
            number_str = str(number) + '  '
            print(number_str[:3], ',', end='')
        print('')
