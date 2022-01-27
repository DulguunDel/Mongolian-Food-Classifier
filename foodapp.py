import streamlit as st
from fastai.vision.all import *
import gdown

#Part 1: Designing the Streamlit app
st.markdown("""# Mongolian Food Classifier

This app is designed to classify what type of Mongolian food it is based on a picture. There are four types of Mongolian food that is given: Buuz, Khuushuur, Tsuivan and Niislel Salat. This will come in handy during Tsagaan Sar if you are not sure which food is on the table. Try it out.""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader(" ", type = ['png','jpg','jpeg'])

st.markdown("""### or Take a picture""")

picture = st.camera_input (" ")

#Part 2: Making sure it is small enough to fit into Github
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! /n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1ZpwaMzm7fATJ6T1WML8-kcSyNvKI0HcV'
        output = 'export.pkl'
        gdown.download(url, output, quiet = False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

#Part 3: Deeplearning section
if image_file is not None:
    img = PILImage.create(image_file)
    st.image(img)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.markdown(f"""**Your food is {pred}**""")
    st.markdown(f""" Confidence level: {max(probs.tolist())} """)

if picture is not None:
    img = PILImage.create(picture)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.markdown(f"""**Your food is {pred}**""")
    st.markdown(f""" Confidence level: {max(probs.tolist())} """)