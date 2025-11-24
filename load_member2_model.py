import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")  # Latest Update 2.20.0
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda
import tensorflow as tf

def load_member2_model():
    print("Loading Member 2's 92% accuracy model...")
    original_model = load_model("models/best_model.keras")
    print("Member 2's model loaded successfully!")
    
    # Create new input for your 224x224 images
    inputs = Input(shape=(224, 224, 3))
    
    # Resize 224 -> 128 using Lambda (100% compatible)
    x = Lambda(lambda img: tf.image.resize(img, (128, 128)))(inputs)
    
    # Run Member 2's model on resized image
    outputs = original_model(x)
    
    # Build final model
    compatible_model = Model(inputs=inputs, outputs=outputs)
    
    # Plain ASCII only — no more Unicode errors!
    print("Model now accepts 224x224 images -> auto-resized to 128x128")
    print("Ready! You are using Member 2's 92%+ accuracy model")
    return compatible_model