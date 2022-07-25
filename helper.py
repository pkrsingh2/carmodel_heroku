# import cv2
import os
import numpy as np
import pickle

import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalMaxPool2D
from tensorflow.keras.models import Model

current_path = os.getcwd()

#let's create the number of unique labels we have available in our dataset


car_category_path = os.path.join(current_path, 'static/car_category.pickle')

def VGGModel(input_size):
    #layers
    vgg = VGG16(input_shape=input_size[1:],include_top=False,weights='imagenet')
    pool = GlobalMaxPool2D()(vgg.output)
    cls = Dense(196,activation='softmax',name="names")(pool)
    box = Dense(4,activation='relu',name="boxes")(pool)    
    # model assembly
    model = Model(inputs=vgg.inputs,outputs=[cls,box])
    return model

#predictor_model = load_model(r'static/car_brand_clf_resnet50.h5')
#predictor_model = load_model(r'static/car_brand_clf_resnet50.h5')
predictor_model = VGGModel(input_size=(None,128,128,3))
predictor_model.load_weights(r'static/VGGModel_2.h5')

with open(car_category_path, 'rb') as handle:
    car_types = pickle.load(handle)

from keras.applications.resnet_v2 import ResNet50V2 , preprocess_input as resnet_preprocess
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from keras.layers import concatenate
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.models import Model

#input_shape = (224,224,3)
#input_shape = (128,128,3)
#input_layer = Input(shape=input_shape)


#first extractor inception_resnet
#preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
#inception_resnet = ResNet50V2(weights = 'imagenet',
#                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_resnet)

#preprocessor_densenet = Lambda(densenet_preprocess)(input_layer)
#densenet = DenseNet121(weights = 'imagenet',
#                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_densenet)


#merge = concatenate([inception_resnet,densenet])
#feature_extractor = Model(inputs = input_layer, outputs = predictor_model)
#model = Model(inputs = input_layer, outputs = merge)
#model.save('feature_extractor.h5')

#predictor_model = load_model(r'static/feature_extractor.h5')

#def decode

print('\nmodel loaded')
def predictor(img_path): # here image is file name 
    img = load_img(img_path, target_size=(128,128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    print("here, I am\n")
    print(img.shape)
    #features = feature_extractor.predict(img)
    #prediction = predictor_model.predict(features)*100
    prediction = predictor_model.predict(img)
    print("Here again\n")
    top_5 = sorted(prediction[0][0])[:5]
    print(top_5)
    print('\n')

    prediction = pd.DataFrame(data=top_5, columns=['car_types'])

    #prediction.columns = ['values']
    #prediction  = prediction.nlargest(5, 'values')
    #prediction = prediction.reset_index()
    #prediction.columns = ['name', 'values']
    return(prediction)
