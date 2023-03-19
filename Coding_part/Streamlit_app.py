import pandas as pd
import streamlit as st
import pickle

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/artificial-cybernetic-circuit-brain-inside-human-nerves-system_163855-21.jpg?size=626&ext=jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title("ALZHEIMER DISEASE PREDICTION")
st.markdown("----------------------")
d = {}
visit = st.number_input("**Enter no.of times you visited hospital**",0,10,step = 1)
d['Visit'] = visit
mr_delay = st.slider("**Enter MR Delay**",0,3000)
d['MR Delay'] = mr_delay
gender = st.radio(
    "**Select your Gender**",
    ('Male','Female'))
d['M/F'] = gender
age = st.slider("**Enter your age**",60,100)
d['Age'] = age
educ = st.number_input("**Enter your Education(lowest-6 to 23-highest)**",6,23,step = 1)
d['EDUC'] = educ
ses = st.selectbox(
    "**Enter your Socio Economic Status**",
    [1,2,3,4,5])
d['SES'] = ses
mmse = st.slider("**Enter your Mini-Mental State Examination Score**",0,30)
d['MMSE'] = mmse
cdr = st.selectbox(
    "**Select your Clinical Dementia Rating**",
    [0,0.5,1,2])
d['CDR'] = cdr
etiv = st.slider("**Select the Estimated total intracranial volume**",1000,2000)
d['eTIV'] = etiv
nwbv = st.slider("**Select the Normalized whole-brain volume**",0.6,0.9)
d['nWBV'] = nwbv
asf = st.slider("**Select your Atlas scaling factor**",0.8,1.6)
d['ASF'] = asf

if st.button("submit"):
    data = pd.DataFrame(data = [d.values()],columns=d.keys())
    data['M/F'] = data['M/F'].map({'Male':1,'Female':0})
    with open("Gradientboosting.pkl", 'rb') as file_obj:
        gb=pickle.load(file_obj)
    #gb = pickle.load("Gradientboosting.pkl")
    pred = gb.predict(data)
    if pred[0] == 0:
        st.success( "PREDICTED AS CURED")
    if pred[0] == 1:
        st.warning("PREDICTED AS PRESENCE OF ALZHEIMER DISEASE")
    if pred[0] == 2:
        st.info("PREDICTED AS NORMAL")
st.button('Clear')
