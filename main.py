#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from tkinter.ttk import Style
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()


from PIL import Image
from helper import *


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0 


st.title('Welcome To Car Model Classifier!')
instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
st.write(instructions)
#TODO : Sidebar

uploaded_file = st.file_uploader('Upload An Image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        display_image = display_image.resize((500,300))
        st.image(display_image)
        prediction = predictor(os.path.join('uploaded',uploaded_file.name))
        print(prediction)
        os.remove('uploaded/'+uploaded_file.name)
        # drawing graphs
        st.text('Predictions :-')

        st.write(prediction)


        # fig, ax = plt.subplots()
        # #ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)
        # ax  = sns.barplot(y = 'name',x='values', data = prediction)
        # ax.set(xlabel='Confidence %', ylabel='Breed')

        # st.pyplot(fig)



