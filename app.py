import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from PIL import Image


st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

with st.sidebar:
    
    
   
       
        image = Image.open('logo_cropify.png')
        st.image(image, width=300)
        st.markdown("<h1 style='text-align: center;'>Crop Recommendation System </h1>", unsafe_allow_html= True)
      
        st.markdown("""
        <h4 style='text-align: left;'>
       ----------------------------------------
        </h4>
        """, unsafe_allow_html=True)
   
        

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")

    
    col1,col2  = st.columns([2,2])
    
    
    with col1: 
         N = st.number_input("Nitrogen", 1,10000)
         P = st.number_input("Phosporus", 1,10000)
         K = st.number_input("Potassium", 1,10000)
         rainfall = st.number_input("Rainfall in mm",0.0,100000.0)
       


    with col2:
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended for your farm.")
      #code for html 


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()