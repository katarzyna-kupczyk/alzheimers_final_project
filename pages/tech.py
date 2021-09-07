import streamlit as st

def app():
    st.subheader('Tech Stack - Under the Hood')
    st.write('''The model used for the predictions made by this website was built using supervised learning with
             a Convolutional Neural Network (CNN) model which has a trained Densenet121 base layer.
              ''')
    st.write('''The dataset used for this project was obtained from the Kaggle database and consisted
             of 6400 MRI scans distributed among 4 different classes (Non Demented, Very Mild Demented,
             Mild Demented and Moderate Demented)''')
    st.write('[Link to Dataset](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images)')
