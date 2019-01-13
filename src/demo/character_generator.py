def character_generator():
    return map(
        lambda ascii_number: chr(ascii_number),
        range(97, 123)
    )
