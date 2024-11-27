import os
import mimetypes
from pathlib import Path

def is_image(file_path):
    """Check if a file is an image based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('image')

def get_top_50_images(directory):
    """Return the raw file paths of the top 50 most recent images in a directory."""
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"The path {directory} is not a valid directory.")

    # Get all files in the directory
    all_files = [f for f in Path(directory).iterdir() if f.is_file()]

    # Filter the image files
    image_files = [f for f in all_files if is_image(f)]

    # Sort the image files by modification time, in descending order
    image_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Get the top 50 most recently modified images
    top_images = image_files[:1000]

    # Return the raw string paths
    return [str(image) for image in top_images]

# Example usage:
directory = r'C:\Users\Shashwat\Downloads\dataset\train\FAKE'  # Change this to your target directory
top_images = get_top_50_images(directory)
