import pickle 
import random 
import string 
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model

import streamlit as st
import base64


champ_names = pickle.load(open("data files/champ_names.pkl",'rb'))
responses = pickle.load(open("data files/responses.pkl",'rb'))

data = pd.read_csv('data files/Train_data.csv')

# Tokenizing the inputs

tokenzier = Tokenizer(num_words=2000)
tokenzier.fit_on_texts(data['inputs'])
train = tokenzier.texts_to_sequences(data['inputs'])

# Appling Padding

x_train = pad_sequences(train)

# Encoding

labelEnc = LabelEncoder()
y_train = labelEnc.fit_transform(data['tags'])

# Setting the input shape

input_shape = x_train.shape[1]

# Setting the URLs
build_url = "https://u.gg/lol/champions/{}/build"
patchhis_url = "https://leagueoflegends.fandom.com/wiki/{}/LoL/Patch_history"


def name_find(list1, list2):
    for i in list1: 
        if i not in list1 or i not in list2:
            name = i
    return name

# Defining Vocabulary 

vocabulary = len(tokenzier.word_index)
#print("Unique word count:", vocabulary)
output_length = labelEnc.classes_.shape[0]
#print("Final output classes:", output_length)

# Defining the model

inp = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(inp)
x = LSTM(20,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(inp,x)

model.load_weights("data files/model_weights.h5")



def predictor(input_text):
    
    texts_pred = []
    
    # Preprocessing
    user_input = [wrd.lower() for wrd in input_text if wrd not in string.punctuation]
    user_input = ''.join(user_input)
    texts_pred.append(user_input)

    # Tokenizing and Padding
    user_input = tokenzier.texts_to_sequences(texts_pred)
    user_input = np.array(user_input).reshape(-1)
    user_input = pad_sequences([user_input],input_shape)

    # Prediction
    output = model.predict(user_input, verbose=0)
    output = output.argmax()

    # Assigning the right tag for the prediction
    response_tag = labelEnc.inverse_transform([output])[0]
    
    return response_tag

def responder(pred,input_text):
    
    static_tags = ['greetings', 'tierlist', 'patchnotes', 'bored',  'fun', 'interesting']
    ignore_terms = ["build", "runes", "patches", "patch", "history", "previous", "of", "rune"]
    
    bot_name = "LOL Bot : "
    
    response_tag = pred
    input_text = input_text
    
    name = name_find(input_text.split(), ignore_terms)
    
    if response_tag == "runes":
        
        if name.capitalize() in champ_names:        
            st.write(bot_name, random.choice(responses[response_tag]))
            st.write(build_url.format(name))
            
        else:  
            st.write(bot_name, "OOPS! Give me a valid champ name")
    
    if response_tag == "patchhistory":
       
        if name.capitalize() in champ_names:                 
            st.write(bot_name, random.choice(responses[response_tag]))
            st.write(patchhis_url.format(name))
                
        else:
            st.write(bot_name, "OOPS! Give me a valid champ name")
         
    if response_tag == "broken":
        for i in responses[response_tag]:
            st.write(i)

    if response_tag in static_tags:
        st.write(bot_name, random.choice(responses[response_tag]))
        

# Function for setting background image
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

    
# Setting Background image
set_bg_hack('data files/lolbg.png')


def main():
    
    # Title of the Page
    st.title('The League of Legends Chat Bot')
    

    
    # Adding github url
    gburl = '[GitHub](https://github.com/PropaneHD/League-of-Legends-Chat-bot)'
    st.markdown(gburl, unsafe_allow_html=True)
   
    st.subheader('An early prototype.. More to come!')
    
    st.write("You can ask me things regarding LOL. Like champ runes, patch notes, tierlist and champ's patch history")
    
    text = st.text_input("Hi, Im LOL Chatbot")
    
    if text:
        #st.write("LOL Bot: ")
        with st.spinner('Fetching'):
            res = predictor(text)
            final = responder(res, text)
    
if __name__ == '__main__':
    main()