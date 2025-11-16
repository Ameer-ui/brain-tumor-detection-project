import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data

def predict_and_visualize(model_path='models/multi_class_model.h5', num_samples=4):
    try:
        # Load data
        print("Loading data...")
        _, test_generator, class_names = load_data()
        print(f"Class names: {class_names}")

        # Load model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Collect one sample per class
        class_indices = {c: None for c in range(len(class_names))}
        images_list, labels_list, preds_list = [], [], []
        
        # Iterate through test_generator until we get at least one sample per class
        print("Fetching test batches...")
        for images, labels in test_generator:
            # Predict on current batch
            preds = model.predict(images)
            images_list.append(images)
            labels_list.append(labels)
            preds_list.append(preds)
            
            # Check for samples in this batch
            for i in range(len(labels)):
                c = np.argmax(labels[i])
                if class_indices[c] is None:
                    class_indices[c] = (len(images_list) - 1, i)  # Store (batch_idx, sample_idx)
            
            # Stop if we have one sample per class
            if all(idx is not None for idx in class_indices.values()):
                break
        
        # Check if all classes were found
        for c, idx in class_indices.items():
            if idx is None:
                print(f"Warning: No sample found for class {class_names[c]} in test set")
        
        # Visualize and save one image per class
        for c, idx in class_indices.items():
            if idx is None:
                continue
            batch_idx, sample_idx = idx
            img = images_list[batch_idx][sample_idx]
            # Ensure image is in [0,1] for display
            if img.max() > 1:
                img = img / 255.0
            true_label = class_names[np.argmax(labels_list[batch_idx][sample_idx])]
            pred_label = class_names[np.argmax(preds_list[batch_idx][sample_idx])]
            print(f"Saving prediction for class {class_names[c]}: Pred={pred_label}, True={true_label}")
            plt.figure()
            plt.imshow(img)
            plt.title(f'Pred: {pred_label}, True: {true_label}')
            plt.axis('off')
            save_name = f'prediction_result_Tr-{class_names[c][:2]}Tr_0000.jpg'
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_name}")

    except Exception as e:
        print(f"Error during prediction/visualization: {str(e)}")

if __name__ == '__main__':
    predict_and_visualize()