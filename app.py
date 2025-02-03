import streamlit as st
import numpy as np
import joblib

@st.cache_resource

def load_model():
    model=joblib.load('GBD_Model.pkl')
    return model

st.cache_resource.clear()

st.title('DIABETES PREDICTION APP')
st.subheader('The model will predict diabetes based on the user\'s input')

model=load_model()

if model:
    st.header('Please enter your details')

gender=st.selectbox('GENDER',options=[(0,'Female'),(1,'Male'),(2,'Other')],format_func=lambda x: x[1])
gender_values=gender[0]

age=st.number_input('AGE',min_value=0,max_value=100,value=0)

hypertension=st.selectbox('HYPERTENSION',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[1])
hypertension_values=hypertension[0]

heart_disease=st.selectbox('HEART_DISEASE',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[1])
heart_disease_values=heart_disease[0]

smoking_history=st.selectbox('SMOKING HISTORY',options=[(0,'No Info'), (1,'Current'),(2,'Ever'),(3,'Former'),(4,'Never')],format_func=lambda x: x[1])
smoking_history_values=smoking_history[0]

bmi=st.number_input('BMI',value=0)

HbA1c_level=st.number_input('HbA1c_l Level',value=0)

blood_glucose_level=st.number_input('Blood Glucose Level',value=0)

input_data=np.array([gender_values,age,hypertension_values,heart_disease_values,smoking_history_values,bmi,HbA1c_level,blood_glucose_level])

if st.button('Predict Diabetes Status',key='predict_button'):
    prediction=model.predict(input_data.reshape(1,-1))
    diabetes_status = 'Yes' if prediction[0] == 1 else 'No'
    st.subheader(f'Predicted Diabetes Status:{diabetes_status}')

st.write('Recommendation: Use these results to make better health choices')
