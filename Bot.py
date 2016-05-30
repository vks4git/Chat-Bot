import numpy
from convert import Loader
from sys import stdin
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.optimizers import Adagrad
from keras.layers import LSTM, Embedding
from keras.preprocessing import text

__author__ = 'vks & sashenka228'

batch_size = 500
nb_epoch = 5
vec_size = 1600
layer1 = 1200
layer2 = 800
layer3 = 1200
lstm1 = 64
lstm2 = 64
lstm3 = 256

loader = Loader(vec_size)
data = loader.get()

X_train = numpy.array([data[i] for i in range(len(data) - 1)])
Y_train = numpy.array([data[i + 1] for i in range(len(data) - 1)])

model = Sequential()

model.add(Embedding(vec_size, lstm1, input_length=batch_size))
model.add(LSTM(lstm2, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(TimeDistributedDense(lstm2, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(LSTM(lstm3, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(vec_size))
model.add(Activation('softmax'))

adagrad = Adagrad(lr=0.01, epsilon=1e-6, clipnorm=1.)

model.compile(loss='binary_crossentropy', optimizer=adagrad)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)

print("Talk to me!")

while True:
    rep = stdin.readline()
    rep = loader.toint(rep)[1]
    rep = numpy.reshape(rep, (1, vec_size)).astype("float32")
    ans = model.predict(rep, batch_size=1)
    print(loader.tostring(ans))
