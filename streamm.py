import streamlit as st
import numpy as np
import pandas as pd
import subprocess
import cv2
import numpy as np
from deepface import DeepFace
import sys

st.title('Real time emotion and Gender detection')

st.text('Press q to close camera')
st.text('')

def file():
  subprocess.run([f"{sys.executable}", "Untitled-1.py"])

st.button('open camera',key='open camera')
if st.session_state.get("open camera"):
    st.text('please wait we are loading.......')
    file()
    

