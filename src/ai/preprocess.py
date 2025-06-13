import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

def load_and_preprocess_data(data_dir, img_size=(96, 96), test_size=0.2, val_size=0.1, seed=42, batch_size=16):
    """
    Load and preprocess image data from classified directories

    Args:
        data_dir (str): Path to root data directory
        img_size (tuple): Output image dimensions
        test_size (float): Proportion for test split
        val_size (float): Proportion for validation from training data
        seed (int): Random seed for reproducibility
        batch_size (int): Batch size for dataset iteration
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names)
    """
    
    classes = [
        'battery',
        'biological',
        'brown-glass',
        'cardboard',
        'clothes',
        'green-glass',
        'metal',
        'paper',
        'plastic',
        'shoes',
        'trash',
        'white-glass'
    ]
    
    class_lables = {
        'battery': 0,
        'biological': 1,
        'brown-glass': 2,
        'cardboard': 3,
        'clothes': 4,
        'green-glass': 5,
        'metal': 6,
        'paper': 7,
        'plastic': 8,
        'shoes': 9,
        'trash': 10,
        'white-glass': 11
    }
    
    file_paths = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name, 'images')
        print(f"Checking Directory: {class_dir}")
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(class_lables[class_name])
        
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=seed
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    def preprocess_image(image_path, label):
        """Convet file path to preprocessed image"""
        
        image = cv2.imread(image_path.decode('utf-8'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image / 250.0
        image = image.astype(np.float32)
        
        return image, np.int32(label)
    
    def create_dataset(paths, current_labels, current_batch_size=16, shuffle=False):
        """Create TensorFlow dataset from file paths"""
        dataset = tf.data.Dataset.from_tensor_slices((paths, current_labels))
        
        def wrapped_preprocess(path_tensor, label_tensor):
            
            img, lbl = tf.numpy_function(
                preprocess_image,
                [path_tensor, label_tensor],
                [tf.float32, tf.int32]
            )
            
            img.set_shape((*img_size, 3))
            lbl.set_shape([])
            return img, lbl
        
        dataset = dataset.map(wrapped_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(paths) if paths else 1024)
        dataset = dataset.batch(current_batch_size)
        return dataset
    
    train_dataset = create_dataset(X_train, y_train, current_batch_size=16, shuffle=True)
    val_dataset = create_dataset(X_val, y_val, current_batch_size=16)
    test_dataset = create_dataset(X_test, y_test, current_batch_size=16)
    
    return train_dataset, val_dataset, test_dataset, classes