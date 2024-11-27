import os
import mimetypes
from pathlib import Path

def is_image(file_path):
    """Check if a file is an image based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('image')

def get_top_50_images(directory):
    """Return the raw file paths of the top 50 most recent images in a directory."""
    
    if not os.path.isdir(directory):
        raise ValueError(f"The path {directory} is not a valid directory.")

    
    all_files = [f for f in Path(directory).iterdir() if f.is_file()]

   
    image_files = [f for f in all_files if is_image(f)]

   
    image_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    
    top_images = image_files[:1000]

    
    return [str(image) for image in top_images]


directory = r'C:\Users\Shashwat\Downloads\dataset\train\FAKE'  
top_images = get_top_50_images(directory)
