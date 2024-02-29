# CIFAR-10 Image Classification with Convolutional Neural Network (CNN)

This repository contains a Python script using TensorFlow and Keras to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification. The trained model is saved and can be reused for making predictions on new images.

## Prerequisites

Make sure you have the following libraries installed:

- TensorFlow
- Matplotlib
- NumPy

You can install them using the following command:

```bash
pip install tensorflow matplotlib numpy
```

# Dataset
The CIFAR-10 dataset is used for training and testing the model. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

# Model Architecture
The CNN model is defined using TensorFlow's Keras API. It consists of two convolutional layers with max-pooling, followed by a flatten layer, dropout for regularization, and two dense layers for classification.

```bash
# Model Architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Dropout layer for regularization
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
```

# Training
The model is trained using the Adam optimizer and Sparse Categorical Crossentropy as the loss function. The training history is stored in the history variable.

```bash
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))
```

# Predictions and Visualization
The model's predictions on the test dataset are visualized using a custom plotting function.

```bash
predictions = model.predict(test_images)

def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(f"{class_names[labels[i]]}\n{class_names[np.argmax(predictions[i])]}")
    plt.show()
```

# Saving the Model
The trained model is saved in the Hierarchical Data Format (HDF5) using the .save() method.

```bash
model.save('cnn_model.h5')
```

Feel free to clone and use this repository for your image classification tasks. If you have any questions or suggestions, please open an issue.
