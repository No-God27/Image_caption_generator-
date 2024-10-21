import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
import tensorflow as tf

# Define the NotEqual custom layer
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.backend.not_equal(inputs[0], inputs[1])

    def get_config(self):
        config = super(NotEqual, self).get_config()
        return config

# Load the tokenizer
tokenizer = load(open("tokenizer.p", "rb"))

# Define vocab_size from tokenizer
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding

# Set max_length (this should match your dataset's maximum caption length)
max_length = 32  

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        print(f"ERROR: Couldn't open image! {e}")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    # For images with 4 channels, convert to 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Load the model with the custom object
with tf.keras.utils.custom_object_scope({'NotEqual': NotEqual}):
    model = load_model('gen_models/model_9.h5')

# Load the Xception model for feature extraction
xception_model = Xception(include_top=False, pooling="avg")

# Extract features from the image
photo = extract_features(img_path, xception_model)
if photo is not None:  # Proceed only if the photo was successfully processed
    img = Image.open(img_path)

    # Generate the description
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\nGenerated Description:")
    print(description)

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()  # Ensure the plot is displayed
