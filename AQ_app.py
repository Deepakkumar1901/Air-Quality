import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv("Air_Quality.csv")

# Vectorize the text data using TfidfVectorizer

X=df[['SOi','Noi','Rpi','SPMi']]
Y=df['AQI']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)

# Train a Multinomial Naive Bayes classifier
DT=DecisionTreeRegressor()
DT.fit(X_train,Y_train)

X2 = df[['SOi','Noi','Rpi','SPMi']]
Y2 = df['AQI_Range']

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.33, random_state=70)

#fit the model on train data
DT2 = DecisionTreeClassifier().fit(X_train2,Y_train2)

st.header(" Air Quality Predictor ")
st.write("Enter the level of following contents in air :")

soi = st.number_input('Enter the level of SO2 content in air',min_value = 0,max_value=1000,step=1)
noi = st.number_input('Enter the level of NO2 content in air',min_value = 0,max_value=1000,step=1)
rpi = st.number_input('Enter the level of Respirable Suspended Particulate Matter (RSPM) content in air',min_value = 0,max_value=1000,step=1)
spmi = st.number_input('Enter the level of Suspended Particulate Matter (SPM) content in air',min_value = 0,max_value=1000,step=1)

input_data = [soi,noi,rpi,spmi]


features=np.array([input_data])



pred1 = DT.predict(features)
pred2 = DT2.predict(features)

if pred2=="Good":
    pos_health_imp = "Air quality is satisfactory and poses little or no risk."
elif pred2=="Moderate":
    pos_health_imp = "Sensitive individuals should avoid outdoor activity as they may experience respiratory symptoms."
elif pred2=="Poor":
    pos_health_imp = "General public and sensitive individuals in particular are at risk to experience irritation and respiratory problems."
elif pred2=="Unhealthy":
    pos_health_imp = "Increased likelihood of adverse effects and aggravation to the heart and lungs among general public"
elif pred2=="Very Unhealthy":
    pos_health_imp = "General public will be noticeably affected. Sensitive groups should restrict outdoor activities"
elif pred2=="Hazardous":
    pos_health_imp = "General public at high risk of experiencing strong irritations and adverse health effects. Should avoid outdoor activities."

if st.button("Predict"):
    st.write("Air Quality Index (AQI)  =  ",pred1[0])
    st.write("Air Quality Level  =  ",pred2[0])
    st.write("Health recommends  =  ", pos_health_imp)
