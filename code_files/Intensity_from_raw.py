import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import center_of_mass
import pandas as pd

path = 'C:/Users/'
name1 = '50X_1_i'
min_area = 1000
max_area = 100000 

def extract_color_channels(raw_image):
    # Extract red, green, and blue pixels from Bayer pattern BGGR
    red_pixels = raw_image[1::2, 1::2]
    green_pixels_1 = raw_image[::2, 1::2]
    green_pixels_2 = raw_image[1::2, ::2]
    blue_pixels = raw_image[::2, ::2]
    
    return red_pixels, green_pixels_1, green_pixels_2, blue_pixels

def create_color_image(raw_image, red_pixels, green_pixels_a, green_pixels_b, blue_pixels):
    # Create blank arrays for red, green, and blue images
    red_img = np.zeros_like(raw_image)
    green_img = np.zeros_like(raw_image)
    blue_img = np.zeros_like(raw_image)
    
    # Assign pixels to new individual color images
    red_img[1::2, 1::2] = red_pixels
    green_img[::2, 1::2] = green_pixels_a
    green_img[1::2, ::2] = green_pixels_b
    blue_img[::2, ::2] = blue_pixels
    
    return red_img, green_img, blue_img

def crop(img):
    
    # Remove 5%(0.05) of strip from all sides
    top_crop_percentage = 0.05  
    bottom_crop_percentage = 0.05
    left_crop_percentage = 0.05  
    right_crop_percentage = 0.05  

    # Calculate crop heights and widths
    height, width = img.shape[:2]
    top_crop_height = int(height * top_crop_percentage)
    bottom_crop_height = int(height * bottom_crop_percentage)
    left_crop_width = int(width * left_crop_percentage)
    right_crop_width = int(width * right_crop_percentage)

    # Crop the image
    cropped_image = img[top_crop_height:height-bottom_crop_height, left_crop_width:width-right_crop_width]
    # Print the size of the cropped image
    print(f"Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]} (width x height)")
    
    return cropped_image    

def process_image(image_path):
    # Load the raw image
    with rawpy.imread(image_path) as raw:
        raw_img = raw.raw_image.copy()
        # Apply cropping to the RAW Bayer image
        cropped_raw_img = crop(raw_img)
        # Postprocessing to get the RGB image (for visualization and watershed)
        rgb1 = raw.postprocess()
        rgb = crop(rgb1)
        
    # Extract color channels from the cropped raw image
    red_pixels, green_pixels_a, green_pixels_b, blue_pixels = extract_color_channels(cropped_raw_img)

    # Create individual color images
    red_img, green_img, blue_img = create_color_image(cropped_raw_img, red_pixels, green_pixels_a, green_pixels_b, blue_pixels)

    # Visualize the color channels
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(red_img, cmap='Reds')
    plt.title('Red Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_img, cmap='Greens')
    plt.title('Green Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_img, cmap='Blues')
    plt.title('Blue Image')
    plt.axis('off')

    plt.suptitle("Color Channels Visualization", y=0.95)
    plt.tight_layout()
    plt.show()

    # Convert the RGB image to grayscale for watershed segmentation
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Watershed
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(rgb, markers.astype(np.int32))

    # Filter regions based on pre-defined area thresholds
    min_area_threshold = min_area
    max_area_threshold = max_area
    print('Total segments:', len(np.unique(markers)))
    print('Area filter window:', min_area_threshold, 'to', max_area_threshold)

    filtered_markers = np.zeros_like(markers)
    for label in np.unique(markers):
        if label == -1:
            continue
        area = np.sum(markers == label)
        if min_area_threshold < area < max_area_threshold:
            filtered_markers[markers == label] = label

    # Calculate average intensities and other properties
    average_intensity_r = []
    average_intensity_g = []
    average_intensity_b = []
    area_all = []
    centroid_x = []
    centroid_y = []

    for label in tqdm(np.unique(filtered_markers), desc="Processing labels"):
        if label == -1:
            continue
        mask = np.where(filtered_markers == label, 1, 0).astype(np.uint8)
        mean_intensity_r = np.mean(red_img, where=np.logical_and(mask, red_img != 0))
        mean_intensity_g = np.mean(green_img, where=np.logical_and(mask, green_img != 0))
        mean_intensity_b = np.mean(blue_img, where=np.logical_and(mask, blue_img != 0))
        centroid = center_of_mass(mask)
        centroid_x.append(centroid[1])  # X coordinate
        centroid_y.append(centroid[0])  # Y coordinate
        area_all.append(np.sum(mask))
        average_intensity_r.append(mean_intensity_r)
        average_intensity_g.append(mean_intensity_g)
        average_intensity_b.append(mean_intensity_b)

    print('Filtered Segments:', len(np.unique(filtered_markers)))

    # Visualize the segmentation results
    num_labels = np.max(filtered_markers)
    random_colors = np.random.rand(num_labels + 1, 3)
    segmented_image = np.zeros((filtered_markers.shape[0], filtered_markers.shape[1], 3))
    for label in range(1, num_labels + 1):
        segmented_image[filtered_markers == label] = random_colors[label]
    segmented_image[filtered_markers == 0] = [1, 1, 1]  # White background

    plt.imshow(segmented_image)
    plt.title('Filtered Segmented Image with Random Particle Colors')
    plt.axis('off')
    plt.show()

    return centroid_x, centroid_y, average_intensity_r, average_intensity_g, average_intensity_b

centroid_x1, centroid_y1, avg_r1, avg_g1, avg_b1 = process_image(path+name1+'.dng')

print(f"Mean Red Intensity: {np.mean(avg_r1):.2f}")
print(f"Mean Green Intensity: {np.mean(avg_g1):.2f}")
print(f"Mean Blue Intensity: {np.mean(avg_b1):.2f}")

# Create a DataFrame with all data in separate columns
df = pd.DataFrame({
    "Centroid_X": centroid_x1,
    "Centroid_Y": centroid_y1,
    "Avg_Red": avg_r1,
    "Avg_Green": avg_g1,
    "Avg_Blue": avg_b1
})

# Define output filename
output_filename = f"{name1}_RGB_intensities.xlsx"

# Save to a single Excel sheet
df.to_excel(path+output_filename, index=False, engine='openpyxl')

# Plot histogram for avg_r1, avg_g1, avg_b1 in the same plot
plt.figure(figsize=(12, 5))
plt.suptitle(name1, y=1.01)
x_lim = 255

plt.hist(avg_r1, bins='auto', color='r', alpha=0.6, label='Red')  # Red channel
plt.hist(avg_g1, bins='auto', color='g', alpha=0.6, label='Green')  # Green channel
plt.hist(avg_b1, bins='auto', color='b', alpha=0.6, label='Blue')  # Blue channel

plt.title('Histogram of RGB Intensities')
plt.xlabel('Mean Pixel Intensity')
plt.xlim([1, x_lim])
plt.ylabel('Frequency')
plt.legend()

plt.show()