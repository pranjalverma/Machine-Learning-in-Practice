import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# Manage input
def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="./tmp/mnist_model"
)

# test input
img = cv2.imread('images/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(255-gray, (28, 28))
gray = gray.flatten() / 255.0

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #x={"x": input(mnist.train)[0]},
    #y=input(mnist.train)[1],
    x={"x": np.array([gray])},
    y=np.array([1]),
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

# train net
classifier.train(input_fn=train_input_fn, steps=100) #steps=100000

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy of net
#accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
#print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
