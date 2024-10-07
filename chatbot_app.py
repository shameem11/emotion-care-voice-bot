import streamlit as st 
import os
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings 
import cv2
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.models import model_from_json
from keras.models import Sequential

#from tensorflow.keras.models import model_from_json

import numpy as np
from langchain.llms import HuggingFacePipeline
from streamlit_chat import message
from pinecone import Pinecone
import TextProcessor.text_type as tp
import random
import nltk
os.environ['CURL_CA_BUNDLE'] = ''
pc = Pinecone(api_key="bbe4eb3a-3ffc-45b0-9827-685273a18c88")
index_name = "emotinalcare"
st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = "MBZUAI/LaMini-T5-738M"
#print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)


with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights("emotiondetector.h5")
# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
analyzer = SentimentIntensityAnalyzer()





def get_sentiment(text):

    scores = analyzer.polarity_scores(text)
    sentiment = "None"
    if scores['neu'] > scores['pos']  and scores['neu']  > scores['neg']:
        return sentiment
    elif scores['pos']  > scores['neg'] :
        return "Positive"
    else : 
        return "Negative"



def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

@st.cache_resource
def data_ingestion():

    
    return

@st.cache_resource
def llm_pipeline(meta , text):
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.2,
        top_p= 0.20,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = meta , question = text)
    return ans



def process_answer(instruction):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index = pc.Index(index_name)

    text = instruction
    
    text_embed = embeddings.embed_query(text)
    get_response = index.query(
        namespace = "np10",
        vector = text_embed,
        top_k =  5,
        includeMetadata = True

    )
    
    meta = [ i.metadata['text'] for i in  get_response.matches]
    print(meta)
    
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.20,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = meta , question = text)
    
    return ans

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
FRAME_WINDOW = st.image([])
def detect_emotion() :
    webcam=cv2.VideoCapture(0)
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    ans = "neutral"
    prediction_label = []
    for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))   
            ans = prediction_label
            FRAME_WINDOW.image(img ,ans)
    print(ans)
    return ans

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


def main():
    
    positive = ["its good to hear" , "wow" , "Amazing" , "Its Fantastic" , "Its so amazing"]
    negative = ["Sorry to hear that" , "Dont worry" , "We can solve them" , "dont lose hope buddy" , "Be confident Dont worry"]
    nonetype = ["ohh.. ok" , "then" , "yep" , "Okayeee" , "Hmmm" , "uhhh" ]
        
    sad = ["What happened today ?" , "Dont feel I hear to talk" , "Hey are you Okay ? " , "Tell me I will Solve whatever you need..."]
    
    neutral = ["Bro What about today ?" , "Tell me how are you ? " , "Anything Special happened" , "Do you need any answers from me ?"]
    
    greeting = ["hello" , "hi" , "hey"] 
    col1, col2 , col3= st.columns([1,2 , 1])
    with col2:
        with st.spinner('Chat is warming up...'):
            ingested_data = data_ingestion()
            st.success('Chatbot is ready...')
            
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

        
        user_input = st.text_input("", key="input")
        
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there !"]
        
        
        
        
        if user_input:
            if user_input.strip().lower() in greeting :
                emotion = detect_emotion()
                if emotion is not "happy" or "neutral" :
                    answer = sad[random.randrange(len(sad))]
                else :
                    answer = neutral[random.randrange(len(neutral))]
            elif  not tp.predict(user_input ).lower().startswith("wh"):
                if get_sentiment(user_input) == 'Positive' :
                    answer = positive[random.randrange(len(positive))]
                elif get_sentiment(user_input) == 'Negative' :
                    answer = negative[random.randrange(len(negative))]
                else :
                    answer = nonetype[random.randrange(len(nonetype))]
                    
            else :
                answer = process_answer(user_input)
            st.session_state["past"].append(user_input)
            response =  answer
            st.session_state["generated"].append(response)
            
            
        if st.session_state["generated"]:
            display_conversation(st.session_state)
        

        




if __name__ == "__main__":
    main()


