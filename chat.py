import streamlit as st
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

data = pd.read_csv('Samsung_Dialog.txt', sep =':', header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answer'] = sales


# Define a function for text processing (includiing lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)


    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turn to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        # The code above does the following:
        # Identifies every word in the sentence
        # Turns it to a lower case
        # Lemmatizes it if the word is alphameric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)


new_data['ProcessedQuestions'] = new_data['Question'].apply(preprocess_text)


xtrain = new_data['ProcessedQuestions'].to_list()


# Vectorize Corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)




bot_greeting = ['Hello user, I am a creation of Chamar.... Ask your question', 
                'Hey you! Tell me what yhou want',
                'I am like a genie in a kettle. Hit me with your question and I will answer',
                'Search yhour heart and ask me that question that hits you hard']


bot_farewell = ['Thanks for your usage... bye',
                'Hope to see you soon',
                'Glad i was able to help']


human_greeting = ['hi', 'hey there', 'hello', 'hello there','holla', 'salute']

human_exit = ['ok', 'okay', 'thanks', 'thank you', 'thanks bye', 'quit', 'bye bye', 'close']


import random
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell)


#----------------------------------------STREAMLIT------------------------------------
st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Charity</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ. ")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
# col2.image('pngwing.com (11).png')

userPrompt = st.chat_input('Ask Your Question')
if userPrompt:
    col1.chat_message("ai").write(userPrompt)

    userPrompt = userPrompt.lower()
    if userPrompt in human_greeting:
        col1.chat_message("human").write(random_greeting)
    elif userPrompt in human_exit:
        col1.chat_message("human").write(random_farewell)
    else:
        proUserInput = preprocess_text(userPrompt)
        vect_user = tfidf_vectorizer.transform([proUserInput])
        similarity_scores = cosine_similarity(vect_user, corpus)
        most_similarity_index = np.argmax(similarity_scores)
        col1.chat_message("human").write(new_data['Answer'].iloc[most_similarity_index])



# if userPrompt:
#     print(userPrompt)

#     userPrompt = userPrompt.lower()
#     if userPrompt in human_greeting:
#         print(random_greeting)
#     elif userPrompt in human_exit:
#         print(random_farewell)
#     else:
#         proUserInput = preprocess_text(userPrompt)
#         vect_user = tfidf_vectorizer.transform([proUserInput])
#         similarity_scores = cosine_similarity(vect_user, corpus)
#         most_similarity_index = np.argmax(similarity_scores)
#         print(data['Answer'].iloc[most_similarity_index])