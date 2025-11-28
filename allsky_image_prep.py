import os
import cv2
import numpy as np

def preprocess_images(input_dir, target_size=(224, 224), initial_crop_size=(1300, 1300)):
    """
    Reads images from a directory and its subdirectories, performs a center crop
    to a specific size, resizes the result to a final target size, and saves the 
    converted images back into their original folders with a '_prepped' suffix.

    Args:
        input_dir (str): The root directory containing your labeled image folders.
                         e.g., 'training_data'
        target_size (tuple): The desired final output size (width, height) for AI training. (e.g., 224x224)
        initial_crop_size (tuple): The specific dimension (width, height) to center crop
                                   the image to before the final resize. Use None or 
                                   (0, 0) to skip the intermediate crop step.
                                   (e.g., 1300x1300)
    """
    # Extract crop dimensions and set flag to skip if dimensions are invalid
    crop_w, crop_h = initial_crop_size
    skip_crop = crop_w <= 0 or crop_h <= 0 or initial_crop_size is None

    # Walk through the input directory to find all image files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is a common image format AND is not already a prepped file
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and '_prepped.' not in file.lower():
                input_path = os.path.join(root, file)

                # --- NEW OUTPUT PATH LOGIC ---
                # Create the new filename with a '_prepped' suffix
                base_name, ext = os.path.splitext(file)
                new_file_name = f"{base_name}_prepped{ext}"
                output_path = os.path.join(root, new_file_name)
                # -----------------------------
                
                try:
                    # Read the image
                    image = cv2.imread(input_path)
                    if image is None:
                        print(f"Warning: Could not read image {input_path}. Skipping.")
                        continue
                    
                    # --- Step 1: Center Crop to specific dimensions (e.g., 1300x1300) ---
                    if not skip_crop:
                        h, w = image.shape[:2]
                        
                        # Check if the original image is large enough for the requested crop
                        if h < crop_h or w < crop_w:
                            print(f"Warning: Image {input_path} ({w}x{h}) is smaller than requested crop size ({crop_w}x{crop_h}). Skipping intermediate crop, resizing directly.")
                        else:
                            # Find the coordinates for the center crop
                            start_x = (w - crop_w) // 2
                            start_y = (h - crop_h) // 2
                            
                            # Perform the crop
                            image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
                            # Simplified print message
                            # print(f"Cropped {input_path} to {crop_w}x{crop_h}.")


                    # --- Step 2: Resize the cropped image to the final target size (e.g., 224x224) ---
                    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                    # Simplified print message
                    # print(f"Resized to {target_size[0]}x{target_size[1]}.")

                    # Save the processed image
                    cv2.imwrite(output_path, resized_image)
                    print(f"Processed and saved: {output_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# --- Example Usage ---
# The script will save the new, converted files in the same folders as the originals
# with a '_prepped' suffix (e.g., 'image_001_prepped.jpg').

# Set your input directory using a raw string for safety with Windows backslashes
input_dir = r'E:\observatory design\Allsky_AI_Training'

# The call below will first center-crop to 1300x1300, then resize to 224x224,
# and save the result as 'original_name_prepped.jpg' in the original folder.
preprocess_images(
    input_dir, 
    target_size=(224, 224), 
    initial_crop_size=(1300, 1300)
)