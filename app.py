import streamlit
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def text_preprocessing(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))





streamlit.title("Email Spam  Classifier")

input_text=streamlit.text_input("Enter the message")


if streamlit.button('Predict'):

#1. preprocessing
  preprocessed_text=text_preprocessing(input_text)

#2.Vectorize

  vector_input=tfidf.transform([preprocessed_text])

#3.predict
  result=model.predict(vector_input)[0]
# 4.Display
  if result==1:
    streamlit.header("Spam")
  else:
    streamlit.header("Not spam")



