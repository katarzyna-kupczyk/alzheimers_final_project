import streamlit as st
import streamlit.components.v1 as components



def app():
    # components.html(
    #     '<h1 style="text-align:center; color:White; font-family:IBM Plex Sans Condensed">  How this website works</h1>'
    # )
    columns = st.columns((1,5,1))
    columns[1].title('How this website works')
    columns = st.columns((1, 2, 1))
    columns[1].subheader('1: Upload MRI scan')
    columns = st.columns((1, 2, 1))
    columns[1].subheader('2: Model detects stage of Alzheimer\'s Disease')
    columns = st.columns((1, 2, 1))
    columns[1].subheader('3: Generation of medical report')
