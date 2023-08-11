# import libraries
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler

# load the model from disk
model = load('titanic_survival.joblib')
# model = load('../myModelTrain/titanic_survival.joblib')

# Create Streamlit Web app
st.title('Titanic Survival Predictions')

# Sidebar with menu options
st.sidebar.title('Menu')

# Menu options
menu = ['Home','Prediction']
st.sidebar.selectbox('', menu)

# Input components
age = st.slider('Age', 0.42, 80.0, 30.0)
sibsp = st.slider('SibSp', 0, 8, 0)
parch = st.slider('Parch', 0, 9, 0)
fare = st.slider('Fare', 0.0, 512.30, 32.20)

# Add predict button
predict_button = st.button('Predict')

#Prediction logic
if predict_button:
    # scaler = StandardScaler()
    # input_data = scaler.fit_transform([[age], [sibsp], [parch], [fare]])
    # st.write([age, sibsp, parch,fare])
    # st.write(input_data)
    # prediction = model.predict([input_data][0])
    
    # if prediction[0] == 1:
    #     st.write('Survived')
    # else:
    #     st.write('Died')

    input_data = [[age, sibsp, parch,fare]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display prediction
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.write('Survived')
    else:
        st.write('Died')
    

