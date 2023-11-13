BLOCK_SIZE = 16

alphabet_lookup_table = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 
    'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15,
    'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
    'y': 24, 'z': 25
}

def blockify(list: int):
    for i in range(0, len(list), BLOCK_SIZE):
        yield list[i:i + BLOCK_SIZE]

def tetrahash(input: str) -> str:

    # Remove anything that isn't a letter

    # Convert all letters to their lookup value
    convert_char = lambda char: alphabet_lookup_table.get(char, 0)
    converted_numbers = list(map(convert_char, input))

    numeric_blocks = list(blockify(converted_numbers))

    if len(numeric_blocks[-1]) < BLOCK_SIZE:
        numeric_blocks[-1] += [0] * (BLOCK_SIZE - len(numeric_blocks[-1]))

    for block in numeric_blocks:
        print(block)
    
    return str(0)

tetrahash("securityissomuchfun")