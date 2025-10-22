import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

def lighten_color(hex_color, factor=0.3):
    """Lighten a hex color by mixing with white"""
    rgb = np.array(mcolors.hex2color(hex_color))
    # Mix with white: new_color = color + factor * (white - color)
    lightened = rgb + factor * (1 - rgb)
    return mcolors.rgb2hex(lightened)

def darken_color(hex_color, factor=0.3):
    """Darken a hex color by mixing with black"""
    rgb = np.array(mcolors.hex2color(hex_color))
    # Mix with black: new_color = color * (1 - factor)
    darkened = rgb * (1 - factor)
    return mcolors.rgb2hex(darkened)

def get_color_shades(hex_color, num_shades=5):
    """
    Generate a list of color shades from dark to light with original color in center.
    
    Parameters:
    -----------
    hex_color : str
        Base hex color (e.g., '#009639')
    num_shades : int
        Number of shades on each side of the original color
        Total colors returned = 2 * num_shades + 1
        
    Returns:
    --------
    list
        List of hex colors ordered from darkest to lightest
        [darkest, ..., darker, ORIGINAL, lighter, ..., lightest]
    """
    shades = []
    
    # Generate darker shades (from darkest to less dark)
    for i in range(num_shades, 0, -1):
        factor = i / (num_shades + 3)  # Avoid pure black by using num_shades+3
        dark_shade = darken_color(hex_color, factor)
        shades.append(dark_shade)
    
    # Add original color in the center
    shades.append(hex_color)
    
    # Generate lighter shades (from less light to lightest)
    for i in range(1, num_shades + 1):
        factor = i / (num_shades + 1)  # Avoid pure white by using num_shades+1
        light_shade = lighten_color(hex_color, factor)
        shades.append(light_shade)
    
    return shades

def crop_image_by_fraction(image_path, left=0, right=0, top=0, bottom=0, rotate_90=0):
    """
    Crop an image by specified fractions from each edge and optionally rotate.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    left : float
        Fraction to crop from left edge (0-1)
    right : float  
        Fraction to crop from right edge (0-1)
    top : float
        Fraction to crop from top edge (0-1)
    bottom : float
        Fraction to crop from bottom edge (0-1)
    rotate_90 : int
        Number of 90-degree rotations (0, 1, 2, 3 or negative values)
        0: no rotation, 1: 90° clockwise, 2: 180°, 3: 270° clockwise, etc.
        
    Returns:
    --------
    numpy.ndarray
        Cropped and rotated image array
    """
    # Read the image
    img = mpimg.imread(image_path)
    
    # Get dimensions
    height, width = img.shape[:2]  # Works for both grayscale and color images
    print(f"Original image size: {width}x{height}")
    
    # Calculate crop boundaries
    left_px = int(width * left)
    right_px = int(width * (1 - right))
    top_px = int(height * top)
    bottom_px = int(height * (1 - bottom))
    
    # Crop the image using array slicing
    # Format: img[top:bottom, left:right] for 2D
    # Format: img[top:bottom, left:right, :] for 3D (color)
    if len(img.shape) == 3:  # Color image
        cropped_img = img[top_px:bottom_px, left_px:right_px, :]
    else:  # Grayscale image
        cropped_img = img[top_px:bottom_px, left_px:right_px]
    
    new_height, new_width = cropped_img.shape[:2]
    print(f"Cropped image size: {new_width}x{new_height}")
    
    # Apply rotation if specified
    if rotate_90 != 0:
        # Normalize rotation to 0-3 range
        rotation_steps = rotate_90 % 4
        if rotation_steps != 0:
            cropped_img = np.rot90(cropped_img, k=rotation_steps)
            final_height, final_width = cropped_img.shape[:2]
            print(f"After {rotate_90 * 90}° rotation: {final_width}x{final_height}")
    
    return cropped_img
