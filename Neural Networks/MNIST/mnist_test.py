import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# test input
img = cv2.imread('images/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(255-gray, (28, 28))
gray = gray.flatten() / 255.0

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array([gray])},
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False
)

classifier_1 = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="./tmp/mnist_model"
)

# predict on input
preds = classifier_1.predict(
    input_fn=test_input_fn
    )

for pred_dict in preds:
    class_id = pred_dict['class_ids'][0]
    print(class_id, pred_dict['probabilities'][class_id])

