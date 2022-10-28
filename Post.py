from pydub import AudioSegment
import model
import streamlit as st

st.title('New Post')

option = st.selectbox(
    'Select the Media type',
    ('Choose', 'Text', 'Voice'))

if option == 'Text':
    pred = st.text_area('Text Area', height=70, max_chars=1000, help='Text area to enter the text',
                        placeholder='Enter the text')
    col1, col2 = st.columns(2)
    with col1:
        a = st.button("Detect", key='ab', help='Click to check', disabled=False)
    with col2:
        b = st.button("Post", key='bb', help='Click to Post')
    col3, col4 = st.columns(2)
    if a:
        de = model.detect([pred])
        st.markdown(de[0])
    if b:
        st.write("Posted Successfully")

if option == 'Voice':
    fileObject = st.file_uploader(label="Please upload your file ",type=['wav', 'mp3'])
    if fileObject:
        with open(fileObject.name, 'wb') as f:
            f.write(fileObject.getbuffer())
            ogg_version = AudioSegment.from_mp3(fileObject.name)
            ogg_version.export(fileObject.name, format='wav')
            col5,col6,col7 = st.columns(3)
            with col5:
                d = st.button("Preview",help="Click to preview")
            with col6:
                e = st.button("Detect")
            with col7:
                f = st.button("Post")
            if d:
                audio_file = open(fileObject.name, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
            if e:
                st.write(model.detect([model.voice_file(fileObject.name[:-4]+".wav")])[0])
            if f:
                st.write("Posted Successfully")



