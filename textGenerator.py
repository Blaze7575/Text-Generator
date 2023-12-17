import random
import pickle
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation # longShortTermMemory to predict next character or even stock prediction
from tensorflow.keras.optimizers import RMSprop


text_df = pd.read_csv("fake_or_real_news.csv")
# text_df prints all the text

text = list(text_df.text.values) # turns the all text in a list
joined_text = " ".join(text) # on this space character join all the text

partial_text =  joined_text[:10000] # train them on 10000 characters 

tokenizer = RegexpTokenizer(r"\w+") # tokenize individual words
tokens = tokenizer.tokenize(partial_text.lower()) # text all of it in lower case
# tokens 
# prints tokens

unique_tokens = np.unique(tokens) # prevent duplicate tokens
unique_tokens_index = {token: idx for idx, token in enumerate(unique_tokens)} # map words with index or making a dictionary

# unique_tokens_index
# prints tokens with certain index


# how may words we want to look at and predict next word

# we get 10 words and predict next word and then get next word and predict next word and so and so
n_words = 10 # training data
input_words = []
next_words = []


for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])


x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype = bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)


for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i,j, unique_tokens_index[word]] = 1
    y[i, unique_tokens_index[next_words[i]]] = 1 # because y requires only 1 index
        
        
# x

# now we have  values in x and y


# now can go for training the model


model = Sequential()
model.add(LSTM(128, input_shape = (n_words, len(unique_tokens)), return_sequences = True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))
 

model.compile(loss = "categorical_crossentropy", optimizer = RMSprop(learning_rate = 0.01),metrics=["accuracy"])
model.fit(x,y, batch_size=128, epochs = 30, shuffle = True)

# # here run above

model.save("mymodel.h5")
model = load_model("mymodel.h5")

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1,n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0,i, unique_tokens_index[word]] = 1
    
    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]


possible = predict_next_word("He will go to ", 5)
print([unique_tokens[idx] for idx in possible])



def generate_text(input_text, text_length, creativity = 2):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice[predict_next_word(sub_sequence, creativity)]]
            
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


print(generate_text("Trump and joe biden were talking about", 100,10))
