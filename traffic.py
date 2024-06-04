import tensorflow as tf
import cv2
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs = EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose = 2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))

    images = []
    labels = []

    #   Tip : listdir goes through all subdirections of data_dir
    for i in range(NUM_CATEGORIES):

        folder = os.path.join(data_dir, str(i))

        label = i

        for img in os.listdir(folder):

            image = cv2.imread(os.path.join(data_dir, str(i), img))
            image30x30 = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            
            images.append(image30x30)
            labels.append(label)

    
    return images, labels

def get_model():
    
    #   Create a convolutional neural network
    model = tf.keras.models.Sequential([

        #   Convolutional Layer, learn 32 filters using a 3x3 kernel, why 3 in input shape because there is 3 channels RGB
        tf.keras.layers.Conv2D(
            32, (3, 3), activation = 'relu', input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
            ),

        #   Max-pooling layet, using 2x2 pool size (to reduce dimentionality)
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation = 'relu', input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
            ),

        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

        #   Flatten neurons (units) cause it should be a vector before feeding to fully connected layers
        tf.keras.layers.Flatten(),

        #   Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),

        #   Add another hidden layer with dropout
        # tf.keras.Dense(128, activation = 'relu')
        # tf.keras.Dropout(0.5)

        #   Output Layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = 'softmax')
        
        ])

    #   Train (compile) neural network

    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'recall']
    )

    return model

if __name__ == "__main__":
    main()
