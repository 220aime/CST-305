
# Aime Serge Tuyishime
# Professor Ricardo Citro
# Feb 28. 2025
#Degradation of Data Integrity
# Principles of Modeling and Simulation Lecture & Lab | CST-305
# This project models and visualizes image degradation in digital 
# storage due to bit flips and charge leakage using Python, 
# OpenCV, and Matplotlib. It quantitatively analyzes image quality 
# deterioration over time with Mean Squared Error, offering 
# insights into data integrity challenges in digital archiving and 
# storage solutions.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def simulate_degradation(image, bit_flip_rate, charge_leakage_rate, time_steps):
    original_image = image.copy()  # Save the original image for comparison
    mse_values = []  # Store MSE values over time

    for t in range(time_steps):
        # Simulate bit flips
        bit_flips = np.random.random(image.shape) < bit_flip_rate
        image[bit_flips] = 255 - image[bit_flips]  # Flip pixel values (0 -> 255, 255 -> 0)

        # Simulate charge leakage (gradual loss of intensity)
        image = image * (1 - charge_leakage_rate)
        image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure pixel values stay within valid range

        # Compute Mean Squared Error (MSE)
        mse = np.mean((original_image - image) ** 2)
        mse_values.append(mse)

        # Display the degraded image and original for comparison
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Degraded Image at Time Step {t+1}')
        plt.axis('off')

        plt.show()
        print(f'Time Step {t+1}: MSE = {mse:.2f}')

    return mse_values

if __name__ == "__main__":
    image_path = 'image.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Image not found. Please check the file path.")
    else:
        bit_flip_rate = 0.01  # Probability of a bit flip per pixel
        charge_leakage_rate = 0.02  # Rate of charge leakage (2% per time step)
        time_steps = 10  # Number of time steps to simulate

        # Simulate degradation
        mse_values = simulate_degradation(image, bit_flip_rate, charge_leakage_rate, time_steps)

        # Plot MSE over time
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, time_steps + 1), mse_values, marker='o', linestyle='-', color='b')
        plt.title('Mean Squared Error (MSE) Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.savefig('mse_trend.png')  # Save the plot as an image file
        plt.show()
        print("MSE trend plot saved as 'mse_trend.png'.")
