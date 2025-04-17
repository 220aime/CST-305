# clean_code.py
#4/16/2025
# Programmer: Aime Serge Tuyishime
# Purpose: Calculate area correctly and display the result

# Function to calculate the area of a rectangle
def calculate_area(length, width):
    return length * width

# Function that uses calculate_area and prints result
def print_area():
    l = 10       # Length
    w = 5        # Width
    area = calculate_area(l, w)  # Correct function call
    print('Area:', area)         # Output the result

# Call the main function
print_area()
