from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import re

tokenizer = None

def verwerk_invoerdata(X_train, X_test, y_train, y_test, maxlen=100, maxwords=5000):
    global tokenizer 
    tokenizer = Tokenizer(num_words=maxwords)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)

def maak_klaar_voor_voorspelling(tekst, maxlen):
    global tokenizer
    tekst  = tekst_opschonen_hidden(tekst)
    tokens = tokenizer.texts_to_sequences([tekst])
    tokens = pad_sequences(tokens, padding='post', maxlen=maxlen)
    return tokens.astype(np.float32)

opkuisregel1 = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') # Selecteert alle html elementen
opkuisregel2 = re.compile('[^a-zA-Z]') # Selecteer alle karakters die geen letters zijn
opkuisregel3 = re.compile('\s+[a-zA-Z]\s+') # Selecteer alle enkelvoudige karakters
opkuisregel4 = re.compile('\s+') # Selecteer alle meervoudige spaties

def verwijder_html(tekst):
    return re.sub(opkuisregel1, ' ', tekst)

def vervang_leestekens(tekst):
    return re.sub(opkuisregel2, ' ', tekst)

def verwijder_enkelvoudige_karakters(tekst):
    return re.sub(opkuisregel3, ' ', tekst)

def verwijder_meervoudige_spaties(tekst):
    return re.sub(opkuisregel4, ' ', tekst)

def maak_staafdiagram_polariteiten(polariteiten, aantallen):
    plt.bar([0, 1], aantallen, align='center') # Maak een staafdiagram 
    plt.xticks([0, 1], polariteiten) # Benoem de x-as
    plt.show() # Toon de grafiek
    
def tekst_opschonen_hidden(tekst):
    tekst = verwijder_html(tekst) # Verwijdert alle html elementen uit de tekst
    tekst = vervang_leestekens(tekst) # Vervangt alle leestekens door spaties
    tekst = verwijder_enkelvoudige_karakters(tekst) # Verwijdert alle enkelvoudige letters
    propere_tekst = verwijder_meervoudige_spaties(tekst) # Vervangt meervoudige spaties door één spatie
    return propere_tekst
    
def kuis_dataset_op(dataset):
    return np.apply_along_axis(lambda x: tekst_opschonen_hidden(x[0]), 1, dataset[:, 0, None]) 

def zet_polariteiten_om_naar_getallen(dataset):
    return np.apply_along_axis(lambda x: 1 if x[0] == 'positive' else 0, 1, dataset[:, 1, None])

def stel_deeplearning_model_op(maximum_aantal_woorden):
    model = Sequential()
    model.add(layers.Embedding(maximum_aantal_woorden, 100, input_length=100, name='Omzetting_woord_naar_eigenschappen')) #The embedding layer
    model.add(Flatten(name='Samenvoegen_eigenschappen_zin'))
    model.add(layers.Dense(1,activation='sigmoid', name='Voorspellen_sentiment'))
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=0, save_best_only=True, save_freq='epoch', save_weights_only=False)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test),callbacks=[checkpoint1])
    
def plot_learning_curves(history):   
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax2.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.legend(['train', 'val'], loc='upper left')

    plt.show()
    
