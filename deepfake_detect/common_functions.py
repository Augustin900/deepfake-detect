from requirements import fg, attr, system, name

def insert_at_position(text, insert_chars, position):
    """
    Inserts a set of characters at a specific position in the text, adjusting the text to ensure
    the characters are always at that position.

    Parameters:
    - text (str): The original text.
    - insert_chars (str): The characters to insert.
    - position (int): The position at which to insert the characters (0-indexed).

    Returns:
    - str: The modified text with the inserted characters.
    """
    # Ensure the position is within bounds
    if position < 0:
        position = 0

    # Create padding if needed
    if len(text) < position:
        text = text.ljust(position)

    # Insert the characters
    modified_text = text[:position] + insert_chars + text[position:]
    
    return modified_text

def insert_at_position_and_print(text, insert_chars, position):
    print(insert_at_position(text, insert_chars, position))

def print_error_header():
    print(f"{fg('red')}", end="")
    insert_at_position_and_print(f"|- Error:", " -|", 75)
    print(f"{attr('reset')}", end="")

def print_error_end():
    print(f"{fg('red')}", end="")
    insert_at_position_and_print(f"|- Aborting!", " -|", 75)
    print(f"{attr('reset')}", end="")

    print("|- ======================================================================== -|")
    exit()

def print_error_ext(message):
    print(f"{fg('red')}", end="")
    insert_at_position_and_print(f"|- {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

def print_info(message):
    print(f"{fg('blue')}", end="")
    insert_at_position_and_print(f"|- I: {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

def print_info_ext(message):
    print(f"{fg('blue')}", end="")
    insert_at_position_and_print(f"|- {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

system("cls" if name == "nt" else "clear")

print("|- ======================================================================== -|")