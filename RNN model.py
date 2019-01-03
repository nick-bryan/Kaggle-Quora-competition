## Import libraries, training/test data, and Google News embedding
import numpy as np
import pandas as pd
import operator
import os
import re
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Dropout, Embedding, CuDNNLSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

## Remove punctuation
def clean_text(x):

    x = str(x)
    for punct in "/-_":
        x = x.replace(punct, ' ')
    for punct in '&$%*+=>^@~':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"\'()/:;<[\\]`{|}' + "“”’'":
        x = x.replace(punct, '')
    return x

## Change numbers to ##
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

## Change spelling (misspelling and UK->US)
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'dont': 'do not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'World War 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'Colour':'Color',
                'Centre':'Center',
                'Didnt':'Did not',
                'Doesnt':'Does not',
                'Dont': 'Do not',
                'Isnt':'Is not',
                'Shouldnt':'Should not',
                'Favourite':'Favorite',
                'Travelling':'Traveling',
                'Counselling':'Counseling',
                'Theatre':'Theater',
                'Cancelled':'Canceled',
                'Labour':'Labor',
                'Organisation':'Organization',
                'WWII':'World War 2',
                'Citicise':'Criticize',
                'Instagram': 'social medium',
                'Whatsapp': 'social medium',
                'Snapchat': 'social medium',
                'WhatsApp': 'social medium',
                'SnapChat': 'social medium'
                }

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

## Clean question text data using previous functions
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))

test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

## Create a dictionary of words and their occurence
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

## Check overlap between vocab and embedding
def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

## Run functions to determine overlap and words outside of the embeddings
#sentences = train_df["question_text"].apply(lambda x: x.split())
#vocab = build_vocab(sentences)
#oov = check_coverage(vocab,embeddings_index)

## Training-validation split
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## Fill missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Parameters for tokenizer, padding, and embedding
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Create embedding matrix
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0

for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix[i] = embedding_vector

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

## Define RNN Model and fit it to the training data
model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], input_shape=(maxlen,)))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=500, epochs=3, validation_data=(val_X, val_y))

## Make predictions on validation and test set
pred_val_lstm_y = model.predict([val_X], batch_size=1000, verbose=1)
pred_test_lstm_y = model.predict([test_X], batch_size=1000, verbose=1)

## Determine best threshold to use for predicting insincere
thresholds = []
for thresh in np.arange(0.25, 0.751, 0.01):
    thresh = np.round(thresh, 2)
    f1 = f1_score(val_y, (pred_val_lstm_y > thresh).astype(int))
    acc = accuracy_score(val_y, (pred_val_lstm_y > thresh).astype(int))
    thresholds.append([thresh, f1, acc])
    print("Threshold {0}, F1 score {1}, Accuracy {2}".format(thresh, f1, acc))

## Create submissions
sub_pred = (pred_test_lstm_y > 0.5).flatten().astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": sub_pred})
submit_df.to_csv("submission.csv", index=False)
