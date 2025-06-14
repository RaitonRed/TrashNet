import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models, optimizers, callbacks, Sequential
from src.ai.preprocess import load_and_preprocess_data
import matplotlib.pyplot as plt
import os
import src.options as options

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.config.threading.set_inter_op_parallelism_threads(2)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        
def build_model(input_shape=(96, 96, 3), num_classes=12):
    """
    Build model based on MobileNetV2 with transfer learning

    Args:
        input_shape (tuple): Input image dimensions
        num_classes (int): Number of output classes

    Returns:
        tf.keras.Model: Compiled model
    """
    
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir, model_save_path=os.path.join(options.MODELS_DIR, 'model.h5'), epochs=10):
    """
    Train model with input data

    Args:
        data_dir (str): Path to data directory
        model_save_path (str): Path to save trained model
        epochs (int): Number of training epochs

    Returns:
        tuple: (trained model, training history)
    """
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    train_dataset, val_dataset, test_dataset, class_names = load_and_preprocess_data(
        data_dir,
        img_size=(96, 96),
        batch_size=16
    )
    
    model = build_model(num_classes=len(class_names))
    model.summary()
    
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=2
    )
    
    model.save(model_save_path)
    
    return model, history, test_dataset, class_names

def plot_training_history(history, save_dir=options.RESULTS_DIR+'training_plots'):
    """
    Plot training accuracy and loss metrics

    Args:
        history: History object returned from model.fit()
        save_dir: Directory to save plots
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()
    
if __name__ == '__main__':
    DATA_DIR = options.RAW_DATA_DIR
    model, history, test_dataset, class_names = train_model(DATA_DIR)
    plot_training_history(history)
    print('Model training completed successfully!')