# Image_caption_generator-

the entire code for the project lies within the image_trainer.ipynb file.


The task of creating an image caption entails using principles from computer vision and natural language processing to identify the context of a picture and provide a description in a natural language, such as English.

# Image Caption Generator with CNN – About the Python based Project
The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.

In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

# Dataset 

flickr8k dataset was used. 
Alternatively flickr32k dataset can be used

# What is CNN? 
Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.

# What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

## This model has a BLEU score of only 0.1 because it is not trained enough due to limited resources. If this model was trained with more no of epochs, more accurate results would have been obtained
