import tensorflow as tf # Tensorflow is a backend module that include keras module
mnist = tf.keras.datasets.mnist # Importing build-in MNIST(Digit Recognition) datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Splitting data for training and testing
x_train, x_test = x_train / 255.0, x_test / 255.0 #Normalize data by divided with range of RGB color

# Build a model with Sequential and provide layers for deep learning algorithm
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# compiling the model before training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=5)

# Serving the accuracy and other
model.evaluate(x_test,  y_test, verbose=2)

# print summary about the model
print(model.summary())
