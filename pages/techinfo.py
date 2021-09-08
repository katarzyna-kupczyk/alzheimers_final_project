import streamlit as st

def app():
    st.subheader('Tech Stack - Under the Hood')
    st.subheader('The Problem')
    st.write('''
             Detecting Alzheimer's is resource intensive for medical professionals and unfortunatley
             many individulas with Alzheimer's go undiagnosed for many years.
             Therefore having a tool that can assist medical professionals in the diagnosis stage of Alzheimer's at mass
             scale would be very beneficial.
             ''')

    st.subheader('The Data')
    st.write('''
             The dataset used for this project was obtained from Kaggle and consisted
             of 6400 MRI scans distributed among 4 different classes (Non Demented, Very Mild Demented,
             Mild Demented and Moderate Demented). The dataset was split into test and train folders by default
             on kaggle with an unbalaced distribution; Non Demented and Very-Mild Demented containing far more scans.
             For example in our Train set we had 2560 files for Non Demented yet only 52 for Moderately Demented -
             (this being our greatest data discrepancy.)
             ''')
    st.write('[Link to Dataset](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images)')

    st.subheader('The Tech')
    st.write('''
             The model used for making predictions on this website is a Deep Learning algortihm that
             was built with using a supervised learning approach.
             The model contains Convolutional Neural Networks (CNN's) and leverages transfer learning by having
             DenseNet121 as a base layer.
              ''')
    st.write('[Further reading on DenseNet121](https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8)')

    st.subheader('The Results')
    st.write('The best resutls our model was able to obtain were as follows:')
    st.write('ROC-AUC: 0.89')
    st.write('Accuracy: 0.71')
    st.write('Recall: 0.70')
    st.write('Precision: 0.71')
