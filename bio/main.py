import numpy as np
import base64
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing import image
import os
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools
import h5py
import io
import pickle
from keras.models import load_model
from keras.models import Model
# Deep learning libraries
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu
##code startes
with st.sidebar:
    selected = option_menu(None,
                          ['Home',
                          'Pneumonia',
                          'Malaria',
                           'Heart Disease',
                           ],
                          icons=['house-fill','bi bi-meta','thermometer-half','bi bi-heart'],
                          default_index=0)

if selected=="Home":
     
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
        }}
    </style>
    """,
        unsafe_allow_html=True
    )
    add_bg_from_local('images/home.png')
   
if selected=="Malaria":
    img1=Image.open("images/image-removebg-preview (32).png")
    st.image(img1)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    def load_cnn1():
        model_ = load_model('weights1.h5')
        return model_

    def preprocessed_image(file):
        image = file.resize((44,44), Image.ANTIALIAS)
        image = np.array(image)
        image = np.expand_dims(image, axis=0) 
        return image

    def main():
        st.title('Prediction of Malaria')
        model_1 = load_cnn1()
        images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
        if images is not None:
            images = Image.open(images)
            st.text("Image Uploaded!")
            st.image(images,width=300)
            used_images = preprocessed_image(images)
            predictions = np.argmax(model_1.predict(used_images), axis=-1)
            if predictions == 1:
                st.error("Cells get parasitized")
            elif predictions == 0:
                st.success("Cells is healty Uninfected")
                
    if __name__ == "__main__":
        main()

if selected=="Pneumonia":
    loaded_model=tf.keras.models.load_model('my_model.h5')
    img2=Image.open('images/image-removebg-preview (31).png')
    st.image(img2)
    st.title("Prediction of Pneumonia")
    file=st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
    def predict(image_path):
        image1 = image.load_img(image_path, target_size=(150, 150))
        image1 = image.img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        #st.write(image1.shape)
        img_array= image1/255
        prediction = loaded_model.predict(img_array)
        if prediction[0][0]>.6:
            st.error("You have a high chance of having Pneoumonia, Please consult a doctor")
        else :
            st.success("You have a low chance of having Pneoumonia, Nothing to panic!")
        
    if file is not None:
        img=Image.open(file).convert('RGB')
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        predict(file)


if selected=="Heart Disease":
    loaded_model=tf.keras.models.load_model('my_model.h5')
    img2=Image.open('images/ss.png')
    st.image(img2)
    pickle_in = open('heart.pkl','rb')
    heart = pickle.load(pickle_in)
    def prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        prediction = heart.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        print(prediction)
        return prediction
        # this is the main function in which we define our webpage
    def main():
            # giving the webpage a title
        st.title("Prediction of Heart Disease")
            # the following lines create text boxes in which the user can enter
            # the data required to make the prediction
        age = st.slider("select your age",15,100)
        sex = st.selectbox("Sex",["Male","Female"])
        if sex=="Male":
            sex=1
        if sex=="Female" :
            sex=0
        link_url_1='https://en.wikipedia.org/wiki/Constrictive_pericarditis'
        link_text_1='CLick to learn more'
        label_1=f"Constrictive pericarditis : [{link_text_1}]({link_url_1})"
        cp=st.radio(label_1,["Yes","No"])
        if cp=="Yes":
            cp=1
        if cp=="No":
            cp=0
        trestbps = st.slider("Heart Beat",45,180)
        link_url='https://en.wikipedia.org/wiki/Cholesterol'
        link_text='CLick to learn more'
        label1=f"Cholestrol : [{link_text}]({link_url})"
        chol = st.slider(label1,100,300)
        fbs = st.slider("Fasting blood sugar",80,300)
        if fbs>=120:
            fbs =1
        if fbs<120:
            fbs=0
        restecg = st.slider("Rest ecg",0,2)
        thalach = st.slider("Maximun Heart Rate",100,200)
        exang = st.radio("Exercise induced angina",["Yes","No"])
        if exang =="Yes":
            exang=1
        if exang=="No":
            exang=0
        oldpeak = st.slider("Oldpeak",0.0,7.0)
        slope = st.slider("Slope",0,3)
        ca = st.slider("Ca",0,3)
        thal = st.slider("Thal",0,3)
            #target= st.text_input("target", "0 or 1")
        result =""
        if st.button("Predict"):
            result = prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
            print(result)
            if result == 1:
                st.error("There is chance of Heart disease")
            else:
                st.success("There is no chance of Heart disease")
    if __name__=='__main__':
        main()
    
                
    
	
