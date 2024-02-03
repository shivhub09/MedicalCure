from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tensorflow
import tensorflow as tf
import random
import json
import pickle
from tensorflow.keras.models import load_model


main = Flask(__name__)

with open('intents.json') as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
  for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(pattern)
    words.extend(wrds)
    docs_x.append(wrds)
    docs_y.append(intent["tag"])


  if intent["tag"] not in labels:
    labels.append(intent["tag"])  

words = [stemmer.stem(w.lower()) for w in words if  w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
  bag = []
  wrds = [stemmer.stem(w) for w in doc]

  for w in words:
    if w in wrds:
      bag.append(1)
    else:
      bag.append(0)

  output_row = out_empty[:]
  output_row[labels.index(docs_y[x])] = 1

  training.append(bag)
  output.append(output_row)

def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]

  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1

  return np.array(bag)

model = load_model('model2.h5')
@main.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        user_input = request.json.get('user_input')
        
        # Perform chat logic
        results = model.predict([bag_of_words(user_input, words).reshape(1, -1)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

                # Formatting the responses
                response_list = nltk.sent_tokenize(str(responses[0]))

                

                # Return the formatted responses as JSON
                return jsonify({'responses': response_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    main.run(debug=True)
