def to_character(number_class):
    return chr(number_class + 96)

def normalize_image_0_to_1(flat_images):
    return [
        [
            grayscale_value / 256
            for grayscale_value
            in flat_image
        ]

        for flat_image
        in flat_images
    ]