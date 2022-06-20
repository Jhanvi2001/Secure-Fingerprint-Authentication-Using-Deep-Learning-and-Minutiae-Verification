# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:48:13 2022

@author: sjhan
"""
import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle

icon = Image.open('bisag2.jpg')
st.set_page_config(
    page_title="Fingerprint Detection",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded"
    
)
import webbrowser

model=tensorflow.keras.models.load_model('binary_cnn.h5')
autoencoder=tensorflow.keras.models.load_model('Reconstructing.h5')
st.sidebar.subheader("Choose the option ")
menu = ["Home","Details","Fingerprint Spoofing","Reconstructing Fingerprints","Fingerprint Matching","Help"]
choice = st.sidebar.selectbox("Menu",menu) 
uploaded_file = st.sidebar.file_uploader("Choose an image")    

if choice=="Home":
    st.title("Liveness detection and Reconstruction of Fingerprint")
    st.image("fingerprint.png")
elif choice=="Details":
    # Embed a youtube video
    st.title("What is liveness detection?")
    st.write("Liveness detection in biometrics is the ability of a system to detect if a fingerprint or face (or other biometrics) is real (from a live person present at the point of capture) or fake (from a spoof artifact or lifeless body part).It comprises a set of technical features to counter biometric spoofing attacks where a replica imitating a person’s unique biometrics (like a fingerprint mold or 3D mask made of silicone) is presented to the biometric scanner to deceive or bypass the identification and authentication steps given by the system. Liveness check uses algorithms that analyze data - after they are collected from biometric scanners and readers - to verify if the source is coming from a fake representation.")
    st.title("What are the known fingerprint spoofs?")
    st.write("Fingerprint spoof attacks (also known as Presentation Attacks or PAs in the technical jargon) have been realized using imitations (Presentation Attack Instruments or PAIs) made of all sorts of easy-to-find materials like glue, wax, plasticine, and silicone.The first spoofing of fingerprint readers was reported in 1998 by NETWORK computing.These attacks have been the subject of many tests and publications on spoof detection fingerprint methodologies and techniques (also known as Presentation Attack Detection - PAD) in the last decades. ")
    st.title("How to detect liveness in fingerprint presentation attacks?")
    st.write("Technics to overcome presentation attacks on fingerprints biometric systems are divided into static and dynamic methods.")
    st.image('fingerprint.jpg')
    st.header("Fingerprint recognization")
    st.write("This fingerprint recognition capture shows linear valleys (white) and ridges (black). Minutiae are specific spots such as ridge bifurcations and endings (in yellow and red). The tiny circular dots are sweat pores. (Source Gemalto at Milipol 2017).Solutions will exploit both counterfeit detection and liveness detection methods.")
    st.header("Static methods")
    st.write("They compare a single fingerprint capture with others.They can detect the lack of details in fake fingerprints such as sweat pores, pattern differences, and unnatural features (such as air bubbles) compared to the real ones.Searching for noise and alteration fingerprint marks such as stains has been known for years in forensics. Besides, the skin flexibility is so high that no two fingerprints are ever the same, even when captured immediately after each other. A person may also alter his/her fingerprint pattern intentionally to sidestep identification. Counterfeit detection methods will also detect this type of attempt.Extracted from a single fingerprint capture, static features such as skin elasticity, perspiration-based features, textural characteristics such as smoothness (aka surface coarseness), and morphology can be exploited. For example, natural skin is usually smoother than materials such as gelatin and silicone polymers made of agglomerated molecules. A live finger will also have more ridge distortion than a fake.Interestingly, the spectrum of light reflected by the finger when illuminated is very distinctive of human skin. In the short-wave infrared spectrum, skin reflection is also independent of the skin tone, making it ubiquitous.Another “liveness” signature is sweat. Sweat starts from pores and unevenly diffuses along the ridge. In contrast, spoof captures tend to show high uniformity. Let alone the fact that pores are very tiny and challenging to incorporate in artifacts as they are usually not visible in lifted fingerprints.Multimodal scanners can also combine both finger vein and fingerprint images. ")
    st.header("Dynamic methods")
    st.write("They process multiple fingerprint frames (aka fusion) and perform a more in-depth analysis to detect life signs in acquired fingerprints.")
    st.write("Skin distortion analysis: Skin turns whiter under pressure. This effect becomes visible when a fingertip is pressed against a surface, and the blood flow is held back due to tissue compression.  Besides, the user may be asked to move the finger while pressing it against the scanner surface, thus intentionally amplifying the skin deformation.Blood flow detection.  The idea here is to capture the blood movement beneath the skin to differentiate live fingers from artificial ones. Active sweat pore detection: Active pores with ionic sweat fluid are only available on live fingers and are tough to replicate.")
    st.header("AI-enhanced methods")
    st.write("Today, anti-spoofing measures are leveraging deep learning convolutional neural networks (CNN), most commonly applied to visual imagery analysis. CNN models can be trained to distinguish a live finger from a fake. They can, for instance, identity forged fingerprints with known materials. Artificial neural network algorithms are helping liveness detection algorithms to be more accurate.As a result, tricking fingerprint scanners and readers now requires a markedly higher level of expertise that goes far beyond rudimentary silicone spoofs. Most of all, creating a high-quality artifact from a latent fingerprint requires skill and knowledge similar to that of a forensic specialist with the appropriate lab equipment.The truth is that making functional Fingerprint recognization:This fingerprint recognition capture shows linear valleys (white) and ridges (black). Minutiae are specific spots such as ridge bifurcations and endings (in yellow and red). The tiny circular dots are sweat pores. (Source Gemalto at Milipol 2017).Solutions will exploit both counterfeit detection and liveness detection methods.")
    st.title("Fingerprint Reconstruction")
    st.image("reconstruction.png")
    st.write("Recent studies have shown that it is indeed possible to reconstruct fingerprint images from their minutiae representations. Reconstruction techniques demonstrate the need for securing fingerprint templates, improve the template interoperability and improve fingerprint synthesis. But, there is still a large gap between the matching performance obtained from original fingerprint images and their corresponding reconstructed f ingerprint images. In this paper, the prior knowledge about f ingerprint ridge structures is encoded in terms of orientation patch and continuous phase patch dictionaries to improve the f ingerprint reconstruction. The orientation patch dictionary is used to reconstruct the orientation field from minutiae, while the continuous phase patch dictionary is used to reconstruct the ridge pattern. Experimental results on three public domain databases (FVC2002 DB1 A, FVC2002 DB2 A and NIST SD4) demonstrate that the proposed reconstruction algorithm outperforms the stateof-the-art reconstruction algorithms in terms of both i) spurious and missing minutiae and ii) matching performance with respect to type-I attack (matching the reconstructed fingerprint against the same impression from which minutiae set was extracted) and type-II attack (matching the reconstructed fingerprint against a different impression of the same finger).")
    
elif choice=="Fingerprint Spoofing":

    if uploaded_file is not None:
        my_img = Image.open(uploaded_file)
        my_img=my_img.convert('RGB')
        my_img=my_img.resize((128,128))
        st.image(my_img, caption='Fingerprint')
        frame = np.array(my_img)
        frame = np.expand_dims(frame, axis = 0)
        st.write(frame.shape)
        result = model.predict(frame)
        if result[0][0] == 1:
            prediction = 'Live'
        else:
            prediction = 'Fake'
        st.write(prediction)
            
elif choice=="Reconstructing Fingerprints":
    if uploaded_file is not None:
        my_img = Image.open(uploaded_file)
        my_img=my_img.resize((224,224))
        frame = np.array(my_img)
        frame=frame[...,0]
        frame = np.expand_dims(frame, axis = 0)
        st.write(frame.shape)
        frame = frame.reshape(-1, 224,224, 1)
        frame = frame / np.max(frame)    
        frame= frame.reshape(-1, 224,224, 1)
        pred1 = autoencoder.predict(frame)
        fig=plt.figure(figsize=(5, 5))
        print("Test Images")
        for i in range(1):
            plt.subplot(1, 2, i+1)
            plt.imshow(frame[i, ..., 0], cmap='gray')
        
        st.write("Original Img")
        st.pyplot(fig)
        
        print("Reconstruction of Test Images")
        for i in range(1):
            plt.subplot(1, 2, i+1)
            plt.imshow(pred1[i, ..., 0], cmap='gray')  
        
        st.write("Reconstructed Img")
        st.pyplot(fig)

elif choice=="Fingerprint Matching":
    from fingerprint_matching import show_fingername,test_single_sample
    matching_state, fname, ID = test_single_sample(uploaded_file)
        # printing result
    st.image(uploaded_file)
    st.write(matching_state,'|',fname,'|',ID)
    
elif choice=="Help":
        link = '[Bisag-N](https://bisag-n.gov.in/)'
        st.markdown(link, unsafe_allow_html=True)
        from streamlit_folium import st_folium
        import folium

    # center on Liberty Bell, add marker
        m = folium.Map(location=[23.1897602,72.6347305], zoom_start=16)
        folium.Marker(
            [23.1897602,72.6347305], 
            popup="Bisag-N", 
            tooltip="Bisag-N"
            ).add_to(m)

# call to render Folium map in Streamlit
        st_data = st_folium(m, width = 725)
        