import tensorflow as tf
from tensorflow import keras
import os

# Ensure TensorFlow 2.15 is used
assert tf.__version__.startswith('2.15'), "This script requires TensorFlow 2.15"

# Path to the original model.h5 file
h5_model_path = 'best_model.h5'

# Verify the model.h5 file exists
if not os.path.exists(h5_model_path):
    raise FileNotFoundError(f"Model file {h5_model_path} not found. Please provide the correct path.")

# Define the Sequential model based on the provided model.json
model = keras.Sequential([
    # InputLayer with batch_input_shape for Keras 2.15 compatibility
    keras.layers.InputLayer(batch_input_shape=(None, 30, 126), name='input_layer'),
    
    # Conv1D layer (filters=64, kernel_size=3, activation=relu)
    keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='conv1d'
    ),
    
    # Dropout layer (rate=0.4)
    keras.layers.Dropout(rate=0.4, name='dropout'),
    
    # Conv1D layer (filters=32, kernel_size=3, activation=relu)
    keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='conv1d_1'
    ),
    
    # Dropout layer (rate=0.4)
    keras.layers.Dropout(rate=0.4, name='dropout_1'),
    
    # GlobalAveragePooling1D layer
    keras.layers.GlobalAveragePooling1D(data_format='channels_last', name='global_average_pooling1d'),
    
    # Dense layer (units=32, activation=relu)
    keras.layers.Dense(
        units=32,
        activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='dense'
    ),
    
    # Dense output layer (units=26, activation=softmax)
    keras.layers.Dense(
        units=40,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='dense_1'
    )
])

# Compile the model with the same configuration as in model.json
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to verify architecture
model.summary()

# Load weights from the model.h5 file
try:
    model.load_weights(h5_model_path)
    print("Weights loaded successfully from", h5_model_path)
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Save the model in HDF5 and SavedModel formats
model.save('my_model_tf215.h5')  # HDF5 format
model.save('my_model_tf215')     # SavedModel format
print("Model saved as my_model_tf215.h5 and my_model_tf215")

# Optional: Test the model with a dummy input to verify functionality
dummy_input = tf.random.normal([1, 30, 126])
prediction = model.predict(dummy_input)
print("Prediction shape:", prediction.shape)  # Expected: (1, 26)

# Optional: Convert to TensorFlow.js format programmatically
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'tfjs_model_converted')
print("Model converted to TensorFlow.js format in tfjs_model_converted directory")

# After this, run the command: tensorflowjs_converter --input_format keras sign_language_model sign_language_model_tfjs