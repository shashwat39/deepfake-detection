import cv2
import dlib
import numpy as np
import os
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img
from matplotlib import pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/drive/MyDrive/CS542/shape_predictor_68_face_landmarks.dat")

TARGET_SIZE = (256,256)


def resize_image(image):
    """Resizes the image to the target size (256x256)."""
    image_resize = cv2.resize(image, TARGET_SIZE)
    return (image_resize)


def normalise_image(image):
    """Standardizes the image by subtracting mean and dividing by std for each channel."""
    return (image) / 255


def align_face(image):
    """Aligns the face using dlib landmarks and affine transformation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return image 

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        
        dY = right_eye[1] - left_eye  [1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
        aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return aligned_face

drive_folder_path = '/content/drive/MyDrive/CS542'

# Step 4: Train-Time Augmentation
def augment_image_train(image):
    """Augment image for training using albumentations library."""
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.RandomBrightnessContrast(p=0.75)
    ])

    augmented = augmentation(image=image)
    return augmented['image']

# Step 5: Test-Time Augmentation (TTA)
def test_time_augmentation(image):
    """Apply multiple test-time augmentations and return the average prediction."""
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.5),
    ])

    augmented_images = [image]
    for _ in range(4):  # Create 4 augmented versions
        augmented = augmentation(image=image)
        augmented_images.append(augmented['image'])

    return augmented_images

#preprocessing
def preprocess_image(image_path):
    ''' resizing, face aligning, normalising input image and returning modified image'''
    img = cv2.imread(image_path)

    # 1. Resize the image
    img_resized = resize_image(img)

    # 2. Align the face
    img_aligned = align_face(img_resized)

    # 3. Standardize the image
    img_normalised = normalise_image(img_aligned)

    #return plt.imshow(img_normalised)
    return (img_normalised)

# Data Augmentation for Training
def train_time_augmentation_pipeline(image_folder):
    """Applies preprocessing and augmentation for training."""
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        img = preprocess_image(img_path)

        # Apply augmentation
        augmented_img = augment_image_train(img)

        # Save or use the augmented image
        cv2.imwrite('augmented_' + img_file, augmented_img)

# Test-Time Augmentation
def test_time_augmentation_pipeline(image_path, drive_folder_path):
    """Applies test-time augmentations to a single image and displays them in a grid."""
    img = preprocess_image(image_path)

    # Apply test-time augmentations
    augmented_images = test_time_augmentation(img)

    # Convert augmented images to uint8 for visualization and then BGR to RGB
    augmented_images = [cv2.cvtColor(np.clip(aug_img * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        for aug_img in augmented_images]

    # Set up the figure and axes for a grid display (e.g., 1 row, 5 columns)
    fig, axes = plt.subplots(1, len(augmented_images), figsize=(15, 5))

    # Display each augmented image in the grid
    for i, aug_img in enumerate(augmented_images):
        axes[i].imshow(aug_img)
        axes[i].set_title(f'Augmented Image {i+1}')
        axes[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()
    for i, aug_img in enumerate(augmented_images):
        output_path = os.path.join(drive_folder_path, f'augmented_{i+1}.png')
        cv2.imwrite(output_path, aug_img)

    print(f"Augmented images saved to {drive_folder_path}")