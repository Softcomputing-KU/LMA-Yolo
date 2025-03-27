import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def stratified_split_dataset(image_files, label_files, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    # Split files into train, val, test
    train_files, test_files, train_labels, test_labels = train_test_split(image_files, label_files, test_size=test_ratio, stratify=labels, random_state=42)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=val_ratio/(train_ratio + val_ratio), stratify=[labels[image_files.index(f)] for f in train_files], random_state=42)

    return train_files, val_files, test_files, train_labels, val_labels, test_labels

def copy_files(files, src_dirs, dest_dir):
    for file in files:
        # Skip directories
        if os.path.isdir(file):
            continue
        # Determine the source directory
        src_dir = src_dirs[0] if file in os.listdir(src_dirs[0]) else src_dirs[1]
        src_file_path = os.path.join(src_dir, file)
        if os.path.exists(src_file_path):  # Check if the source file exists
            shutil.copy(src_file_path, os.path.join(dest_dir, file))
        else:
            print(f"Warning: File {src_file_path} not found. Skipping.")

def main():
    random.seed(42)

    # Paths
    base_dir = '/root/autodl-tmp/dataset_remote_sensing/NWPU VHR-10 dataset/NWPU-VHR-10'
    positive_dir = os.path.join(base_dir, 'positive image set')
    negative_dir = os.path.join(base_dir, 'negative image set')
    labels_dir = os.path.join(base_dir, 'ground_coco')
    
    # Output directories
    output_dir = '/root/autodl-tmp/dataset_remote_sensing/NWPU VHR-10 dataset/NWPU-VHR-10-SP'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    # Create output directories
    for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']:
        create_dir(os.path.join(output_dir, subdir))
    
    # Print directory paths for debugging
    print(f"Positive directory: {positive_dir}")
    print(f"Negative directory: {negative_dir}")
    
    # Ensure positive_dir and negative_dir paths are correct
    if not os.path.exists(positive_dir):
        print(f"Error: Directory {positive_dir} does not exist.")
        return
    
    if not os.path.exists(negative_dir):
        print(f"Error: Directory {negative_dir} does not exist.")
        return

    # Collect all image and label files
    positive_images = [f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))]
    negative_images = [f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))]
    
    image_files = positive_images + negative_images
    label_files = [f.replace('.jpg', '.txt') for f in image_files]  # Changed to .txt
    
    # Label distribution: 1 for positive, 0 for negative
    labels = [1] * len(positive_images) + [0] * len(negative_images)
    
    # Ensure all images have corresponding labels
    image_files = [f for f in image_files if f.replace('.jpg', '.txt') in label_files]  # Changed to .txt
    label_files = [f.replace('.jpg', '.txt') for f in image_files]  # Changed to .txt

    # Split dataset with stratification
    train_images, val_images, test_images, train_labels, val_labels, test_labels = stratified_split_dataset(image_files, label_files, labels)

    # Copy files to respective directories
    copy_files(train_images, [positive_dir, negative_dir], os.path.join(train_dir, 'images'))
    copy_files(val_images, [positive_dir, negative_dir], os.path.join(val_dir, 'images'))
    copy_files(test_images, [positive_dir, negative_dir], os.path.join(test_dir, 'images'))

    copy_files(train_labels, [labels_dir, labels_dir], os.path.join(train_dir, 'labels'))
    copy_files(val_labels, [labels_dir, labels_dir], os.path.join(val_dir, 'labels'))
    copy_files(test_labels, [labels_dir, labels_dir], os.path.join(test_dir, 'labels'))
    
    print(f"Dataset split completed. Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

if __name__ == "__main__":
    main()
