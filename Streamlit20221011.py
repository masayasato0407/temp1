#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
#from google_drive_downloader import GoogleDriveDownloader as gdd
from pysurvival.utils import load_model


# In[ ]:


load_model('rsf.zip')


# In[3]:


st.title('Prediction model for post-SVR HCC (SMART model)') 
st.sidebar.markdown("Enter the following items to display the predicted HCC risk")
with st.sidebar:
    with st.form('user_inputs'): 
        gender = st.selectbox('gender', options=['female', 'male']) 
        age=st.slider("age", 0, 100, 50, 1)
        BMI=st.slider("BMI", 0.0, 50.0, 25.0, 0.1)
        alc60 = st.selectbox('Daily alcoholic consumption', options=['Less than 60g', '60g or more']) 
        DM= st.selectbox('Diabetes', options=['absent', 'present'])
        PLT=st.slider("Platelet count (×10^4/µL)", 0.0, 100.0, 20.0, 0.1)
        AFP=st.slider("AFP (ng/mL)", 0.0, 50.0, 5.0, 0.1) 
        ALB=st.slider("Albumin (g/dL)", 0.0, 10.0, 4.0, 0.1)
        TBil=st.slider("Total bilirubin (mg/dL)", 0.0, 10.0, 1.0, 0.1)
        AST=st.slider("AST (IU/L)", 0, 300, 50, 1)
        ALT=st.slider("ALT (IU/L)", 0, 300, 50, 1)
        GGT=st.slider("GGT (IU/L)", 0, 1000, 50, 1)
        st.form_submit_button() 


# In[4]:


from google_drive_downloader import GoogleDriveDownloader as gdd


# In[ ]:




