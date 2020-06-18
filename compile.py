#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## Video to audio conversion

import subprocess

command = "ffmpeg -i tech.mp4 -ab 160k -ac 2 -ar 44100 -vn vid.wav"

subprocess.call(command, shell=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####### STEREO TO MONO ########

from pydub import AudioSegment
sound = AudioSegment.from_wav("example.wav")
sound = sound.set_channels(1)
sound.export("mono.wav", format="wav")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# [START speech_transcribe_async_gcs]
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import os

credential_path = r"C:\Users\HP\project.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def sample_long_running_recognize(storage_uri):
    f = open("ODA_tech.txt","w+")
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition
    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech_v1.SpeechClient()

    
    
    enable_automatic_punctuation = True

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100

    # The language of the supplied audio
    language_code = "en-US"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "sample_rate_hertz": sample_rate_hertz,
        "enable_automatic_punctuation": enable_automatic_punctuation,
        "language_code": language_code,
        "encoding": encoding,
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()

    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        f.write("{}".format(alternative.transcript))


# [END speech_transcribe_async_gcs]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage_uri",
        type=str,
        default="gs://iosoda/vid.wav",
    )
    args = parser.parse_args()

    sample_long_running_recognize(args.storage_uri)

main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
np.random.seed(400)
stemmer = SnowballStemmer("english")


# In[ ]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


# In[ ]:


f=open("ODA_tech.txt", "r")
text=f.read()
print(text)


# In[ ]:


processed_docs=[]
processed_docs.append(preprocess(text))
processed_text=preprocess(text)
print(processed_text)


# In[ ]:


dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[ ]:


document_num = 0
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))


# In[ ]:


lda_model = gensim.models.LdaModel(bow_corpus,num_topics = 3,id2word = dictionary,passes = 50)
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


# In[ ]:


file=open("vid.txt", "r")


# In[ ]:


text2 = file.read()
#3print(text2)


# In[ ]:


# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(text2))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###### NAMED ENTITY RECOGNITION ######


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from google.cloud import language_v1
from google.cloud.language_v1 import enums
import os

credential_path = r"C:\Users\HP\project.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def sample_analyze_sentiment(gcs_content_uri):
    """
    Analyzing Sentiment in text file stored in Cloud Storage

    Args:
      gcs_content_uri Google Cloud Storage URI where the file content is located.
      e.g. gs://[Your Bucket]/[Path to File]
    """

    client = language_v1.LanguageServiceClient()

    # gcs_content_uri = 'gs://cloud-samples-data/language/sentiment-positive.txt'

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"gcs_content_uri": gcs_content_uri, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_sentiment(document, encoding_type=encoding_type)
    # Get overall sentiment of the input document
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
        u"Document sentiment magnitude: {}".format(
            response.document_sentiment.magnitude
        )
    )
    # Get sentiment for all sentences in the document
    for sentence in response.sentences:
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(u"Language of the text: {}".format(response.language))
    
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage_uri",
        type=str,
        default="gs://iosoda/ODA_tech.txt",
    )
    args = parser.parse_args()

    sample_analyze_sentiment(args.storage_uri)

main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# coding: utf-8

# In[ ]:


from flask import *  
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename)  
  
if __name__ == '__main__':  
    app.run(debug = True)

