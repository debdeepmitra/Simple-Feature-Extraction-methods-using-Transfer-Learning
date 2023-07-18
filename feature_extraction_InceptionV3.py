import numpy as np
from tensorflow.keras.applications import InceptionV3
from PIL import Image

def extract_feature_InceptionV3(img):

	#Pre-processing the imput image
	img = img.resize((224, 224))	#Resizing the image to mantain uniformity
	img = np.array(img)	#Converting the image to a numpy array
	img_batch = img.reshape(1, 224, 224, 3)	#Expanding the dimension witch batch size 1

	#Generate the model with pretrained weights
	model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))

	for layer in model.layers:	#Don't train existing weights
  		layer.trainable = False

  	#Extracting features for the given image
  	features = model.predict(img_batch)

  	return features

def main():
	path = '/path/to/the/image'
	img = Image.open(path)
	extracted_features = extract_feature_InceptionV3(img)

if __name__ == '__main__':
	main()