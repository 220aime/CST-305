# buggy_code.py
# Programmer: Aime Serge Tuyishime
#4/16/2025
# Purpose: Simulate butterfly effect by adding a small typo

# Correct function definition
def calculate_area(length, width):
    return length * width

# Function that contains a bug in the function call
def print_area():
    l = 10
    w = 5
    area = calculat_area(l, w)  # Typo here: "calculat_area" instead of "calculate_area"
    print('Area:', area)

# Call the function which will trigger an error
print_area()
