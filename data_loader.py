import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir='data', batch_size=32, img_size=(224, 224)):
    # Define data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test

    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Training'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Testing'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())  # ['glioma', 'meningioma', 'notumor', 'pituitary']
    return train_generator, test_generator, class_names

def visualize_samples(generator, class_names, save_path='multi_class_samples.png'):
    # Get a batch of images
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    train_generator, _, class_names = load_data()
    visualize_samples(train_generator, class_names)