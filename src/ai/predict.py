import tensorflow as tf
import keras
import numpy as np
import cv2
import os
import argparse
import src.options as options

class Predictor:
    def __init__(self, model_path=None, img_size=(96, 96)):
        """
        Initialize predictor

        Args:
            model_path (str): Path to saved model
            img_size (tuple): Input image dimensions
        """
        if model_path is None:
            model_path = os.path.join(options.MODELS_DIR, 'model.h5')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        self.class_names = [
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
        
    def preprocess_image(self, image_path):
        """
        Preprocess input image

        Args:
            image_path (str): path to input image
        
        Returns:
            np.array: Preprocessed image
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.resize(image, self.img_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path):
        """
        Predict class for input image

        Args:
            image_path (str): path to input image
            
        Returns:
            dict: Prediction results
        """
        
        image = self.preprocess_image(image_path)
        
        probs = self.model.predict(image)[0]
        pred_class = np.argmax(probs)
        
        result = {
            'class': self.class_names[pred_class],
            'confidence': float(probs[pred_class]),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.class_names, probs)
            }
        }
        
        return result
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    args = parser.parse_args()
    
    predictor = Predictor()
    
    if os.path.exists(args.image): 
        prediction = predictor.predict(args.image)
        print("Prediction Results:")
        print(f"Class: {prediction['class']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print("Probabilities:")
        for cls, prob in prediction['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
        else:
            print(f"image {args.image} not found")