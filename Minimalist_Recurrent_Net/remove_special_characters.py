# Manually remove special characters without using the unidecode library
def remove_special_symbols(text):
    replacements = {
        'ç': 'c', 'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'é': 'e', 'ê': 'e', 'í': 'i', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ú': 'u', 'ü': 'u',
        'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A', 'É': 'E', 'Ê': 'E', 'Í': 'I', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ú': 'U', 'Ü': 'U'
    }
    for special_char, replacement in replacements.items():
        text = text.replace(special_char, replacement)
    return text

# Convert the original text buffer to plain ASCII
ascii_text_manual = remove_special_symbols(sentence_buffer)

# Write the ASCII version to a new .txt file
ascii_file_path_manual = '/mnt/data/portuguese_text_1MB_ascii_manual.txt'
with open(ascii_file_path_manual, 'w', encoding='utf-8') as f:
    f.write(ascii_text_manual)

ascii_file_path_manual  # Return the file path of the manually processed ASCII version file
