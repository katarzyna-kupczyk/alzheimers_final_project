import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


def app():
    st.title(f"Patient Form")

    columns = st.columns(2)

    first_name = columns[0].text_input("Patient First Name: ", '')
    columns[0].write(first_name)

    last_name = columns[1].text_input("Patient Last Name: ", '')
    columns[1].write(last_name)

    dob = st.date_input("DOB: ")

    report_date = st.date_input("Report Date: ")


    uploaded_file = st.file_uploader("Upload MRI File", type=["jpg", "jpeg", "png"], )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=False, width=300 )



    def report(model_):
        if model_ == 'Non Demented':
            st.write('Result: No Dementia')
            st.write('''Findings: The ventricles, cisterns and sulci are normal for the patient’s age. There is no midline shift.
                     There is no extra-axial fluid collection. There is no evidence of intracranial haemorrhage.
                     There is no evidence of restricted diffusion to suggest an acute infarct. The basal ganglia and thalami are unremarkable.
                     The brainstem and cerebellum are within normal limits*.
                    Impression: No evidence of any pathology.''')
        elif model_ == 'Very Mild Demented':
            st.write('Result: Very Mild Dementia')
            st.write('''Findings: The ventricles, cisterns and sulci show very slight atrophy. There is no midline shift.
                     There is no extra-axial fluid collection. There is no evidence of intracranial haemorrhage.
                     There is no evidence of restricted diffusion to suggest an acute infarct. The basal ganglia and thalami are unremarkable.
                     The brainstem and cerebellum are within normal limits*.
                ''')
            st.write('''Impression: Possible sign of early stage Alzheimer’s disease.''')
            st.write('''Recommendation: GP to carry out cognitive assessment test and monitor for progression.''')
        elif model_ == 'Mild Demented':
            st.write('Result: Mild Dementia')
            st.write('''Findings: The ventricles, cisterns and show mild atrophy. There is no midline shift. There is no extra-axial fluid collection.
                     There is no evidence of intracranial haemorrhage. There is no evidence of restricted diffusion to suggest an acute infarct.
                     The basal ganglia and thalami are unremarkable. The brainstem and cerebellum are within normal limits*.
                ''')
            st.write('Impression: Early stage Alzheimer’s disease.')
            st.write('''Recommendation: GP to carry out cognitive assessment test, monitor for progression and discuss treatment options.''')
        elif model_ == 'Moderate Demented':
            st.write('Result: Moderate Dementia')
            st.write('''Findings: The ventricles, cisterns and sulci show moderate atrophy. There is no midline shift. There is no extra-axial fluid collection.
                     There is no evidence of intracranial haemorrhage. There is no evidence of restricted diffusion to suggest an acute infarct.
                     The basal ganglia and thalami are unremarkable. The brainstem and cerebellum are within normal limits*.
                ''')
            st.write('Impression: Alzheimer’s disease.')
            st.write('''Recommendation: GP to carry out cognitive assessment test, monitor for progression and discuss treatment options.''')
        else:
            st.write('Report Not Found')

    def medical_report():
        st.write('Patient Full Name: ', first_name + ' ' + last_name)
        st.write('Date of Birth: ', dob)
        st.write('Report Date:' , report_date)
        st.write('Examination: MRI')
        report(real_classification)

    def predict_img(path_to_prediction_data):
        image = Image.open(path_to_prediction_data).convert('RGB')
        image_array  = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image_array, (224, 224))
        image = image / 255
        image = tf.expand_dims(image, axis = 0)

        model = tf.keras.models.load_model('alz_model_h5.h5')
        prediction = model.predict(image)
        prediction_array = prediction[0]
        max_value = np.max(prediction_array)
        if max_value == prediction_array[0]:
            classification = 'Mild Demented'
        elif max_value == prediction_array[1]:
            classification = 'Moderate Demented'
        elif max_value == prediction_array[2]:
            classification = 'Non Demented'
        else:
            classification = 'Very Mild Demented'

        return {'prediction': classification}


    if st.button('Generate Report'):
        st.write("Classifying...")
        classification = predict_img(uploaded_file)
        real_classification = classification['prediction']
        medical_report()
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write('''\* These findings are a generalised description of what a brain would look like at
                 the predicted stage of Alzheimer\'s and were not generated by the model itself''')
