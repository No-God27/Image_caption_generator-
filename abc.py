import tensorflow as tf

# Check if TensorFlow detects any GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow GPU is available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available.")
