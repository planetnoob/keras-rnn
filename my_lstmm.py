from functools import partial

from keras.callbacks import *
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.models import *
from keras.optimizers import *
from losswise.libs import LosswiseKerasCallback


class BaseRNN:

    def __init__(self, num_classes=1, batch_size=44, num_steps=7,
                 lstm_size=5, learning_rate=1e-3, dropout=0.0,
                 stateful=True, embeddinglength=50, metric=None):
        # config
        self.emdedlength = embeddinglength  # embedding length
        self.lstm_units = int(lstm_size)
        self.lr = learning_rate
        self.classes = num_classes
        self.dropval = dropout
        self.bsize = batch_size
        self.stepsnum = num_steps
        self.expand = lambda x: np.expand_dims(x, axis=0)
        self.embname = lambda: ''.join((''.join(['iter: ',str(9),
                                                 ': ', str(45),'-',
                                                 str(time.ctime()),
                                                 '.emb'])).split(' '))
        # model definitions
        self.batchSize = (self.bsize, self.stepsnum, num_classes)
        self.inp_Size = (self.stepsnum, num_classes)

        # state
        self.count = 0
        self.stateful = stateful
        self.metric = metric or 'mean_squared_error'

        # modules
        self.sequential = partial(Sequential, )
        self.dropout = partial(Dropout, rate=self.dropval)
        self.flatten = partial(Flatten, )
        self.timedist_wrap = partial(TimeDistributed, )
        self.Embed = partial(Embedding, output_dim=self.emdedlength)
        self.GRU = partial(CuDNNGRU, units=self.lstm_units,
                           stateful=self.stateful)
        self.CuLSTM = partial(CuDNNLSTM, units=self.lstm_units,
                              stateful=self.stateful)
        self.cnnlstm = partial(ConvLSTM2D, filters=self.lstm_units,
                               kernel_size=(25, 25), strides=(25, 25),
                               padding='same')
        self.cnn1D = partial(Conv1D,
                             filters=self.lstm_units,
                             kernel_size=self.lstm_units // 8,
                             padding='same',
                             strides=1,
                             activation='relu')
        self.cnn2D = partial(Conv2D, filters=self.lstm_units // 4,
                             kernel_size=(8, 8), strides=(4, 4),
                             padding='same')
        self.pool1D = partial(AveragePooling1D,
                              pool_size=8,
                              strides=1)

        self.pool2D = partial(AveragePooling2D,
                              pool_size=(7, 7),
                              strides=(12, 12))
        self.pool2D_global = partial(GlobalMaxPooling2D, )
        self.Dense = partial(Dense, units=num_classes, activation='linear')


class Part1_RNN(BaseRNN):

    def __init__(self, num_classes=1, batch_size=88, num_steps=7,
                 lstm_size=5, learning_rate=1e-3, dropout=0,
                 stateful=True, embeddinglength=50, metric=None):

        super(Part1_RNN, self).__init__(num_classes, batch_size, num_steps,
                                        lstm_size, learning_rate, dropout,
                                        stateful, embeddinglength, metric)
        # rnn model details.
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        # layer 1: LSTM's
        if self.stateful:
            model.add(self.CuLSTM(batch_input_shape=self.batchSize))
        else:
            model.add(self.CuLSTM(shape=self.inp_Size))

        # layer 1.1 -a: Dense <- CuLSTM
        # model.add(self.Dense(units=self.lstm_units)) #<- disabled due to Udacity passing

        # layer 2: linear Predictor <- Dense
        model.add(self.Dense())
        return model

    def makemodel(self, optimizer, lossweights=None):

        if self.model is None:
            return None
        else:
            if optimizer == 'rmsprop':
                optimizer_ = rmsprop(lr=self.lr, decay=0.01, )
            elif optimizer == 'sgd':
                optimizer_ = sgd(lr=self.lr, momentum=0.9, nesterov=True, decay=0.01)
            elif optimizer == 'nadam':
                optimizer_ = nadam(lr=self.lr, schedule_decay=0.01)
            else:
                return None

        self.model.compile(optimizer=optimizer_,
                           loss='categorical_crossentropy',
                           loss_weights=lossweights,
                           metrics=['mean_squared_error'])
        return self.model

    def getcallbacks(self, checkpointstr=None, eval_cb=None):

        # defaults
        lw_sess_tag = 'Part1_RNN' + time.asctime().replace(" ", '')
        chkptname = os.getcwd() + '\\checkpoints\\' + (checkpointstr or 'rnn1_unnamed_model.hdf5')
        basedir = os.getcwd() + '\\checkpoints\\'
        callbacks = []

        modelcheckpoint = ModelCheckpoint(chkptname, monitor='loss',
                                          verbose=1, save_best_only=True)

        lrop = ReduceLROnPlateau(monitor='loss', factor=0.8,
                                 patience=8, verbose=1, mode='min',
                                 min_delta=1e-8, cooldown=4, min_lr=1e-10)

        earlystopping = EarlyStopping(monitor='loss', min_delta=1e-8,
                                      patience=30, verbose=1, mode='min')

        tboard = TensorBoard(log_dir=basedir, histogram_freq=1,
                             batch_size=self.bsize, write_graph=True,
                             write_grads=True, write_images=True)

        losswise_ = LosswiseKerasCallback(tag=lw_sess_tag,
                                          display_interval=1)

        history = History()

        if eval_cb == 'tb':
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history, tboard]
        elif eval_cb == 'lw':
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history, losswise_]
        elif eval_cb is None:
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history]
        return callbacks


class Part2_RNN(BaseRNN):

    def __init__(self, num_classes=33, batch_size=50, num_steps=100,
                 lstm_size=200, learning_rate=1e-3, dropout=0.0,
                 stateful=True, embeddinglength=50, metric=None):

        super(Part2_RNN, self).__init__(num_classes, batch_size, num_steps,
                                        lstm_size, learning_rate, dropout,
                                        stateful, embeddinglength, metric)
        # rnn model details.
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        # 1:layer 1: LSTM
        if self.stateful:
            model.add(self.CuLSTM(units=200,
                                  unit_forget_bias=True,
                                  batch_input_shape=self.batchSize))
        else:
            model.add(self.CuLSTM(units=200,
                                  unit_forget_bias=True,
                                  shape=self.inp_Size))

        # 2:layer 2: Dense Linear
        model.add(self.Dense(units=self.classes))

        # 3:layer 0: Dense + Classifier
        # model.add(self.Dense(activation='softmax')) # -> disabled for udacity passing
        model.add(Activation(activation='softmax'))

        return model

    def makemodel(self, optimizer, lossweights=None):
        if self.model is None:
            return None
        else:
            if optimizer == 'rmsprop':
                optimizer_ = rmsprop(lr=self.lr, decay=0.01, )
            elif optimizer == 'sgd':
                optimizer_ = sgd(lr=self.lr, momentum=0.9, nesterov=True, decay=0.01)
            elif optimizer == 'nadam':
                optimizer_ = nadam(lr=self.lr, schedule_decay=0.01)
            else:
                return None

        self.model.compile(optimizer=optimizer_,
                           loss='categorical_crossentropy',
                           loss_weights=lossweights,
                           metrics=['categorical_crossentropy',
                                    'categorical accuracy'])
        return self.model

    def getcallbacks(self, checkpointstr=None, eval_cb=None):

        # defaults
        lw_sess_tag = 'Part2_RNN' + time.asctime().replace(" ", '')
        chkptname = os.getcwd() + '\\checkpoints\\' + (checkpointstr or 'rnn2_unnamed_model.hdf5')
        basedir = os.getcwd() + '\\checkpoints\\'
        callbacks = []

        modelcheckpoint = ModelCheckpoint(chkptname, monitor='loss',
                                          verbose=1, save_best_only=True)

        lrop = ReduceLROnPlateau(monitor='loss', factor=0.8,
                                 patience=8, verbose=1, mode='min',
                                 min_delta=1e-8, cooldown=4, min_lr=1e-10)

        earlystopping = EarlyStopping(monitor='loss', min_delta=1e-8,
                                      patience=30, verbose=1, mode='min')

        tboard = TensorBoard(log_dir=basedir, histogram_freq=1,
                             batch_size=self.bsize, write_graph=True,
                             write_grads=True, write_images=True)

        losswise_ = LosswiseKerasCallback(tag=lw_sess_tag,
                                          display_interval=1)

        history = History()

        if eval_cb == 'tb':
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history, tboard]
        elif eval_cb == 'lw':
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history, losswise_]
        elif eval_cb is None:
            callbacks = [modelcheckpoint, lrop,
                         earlystopping,
                         history]
        return callbacks


if __name__ == '__main__':
    import losswise

    losswise.set_api_key("5FPT5QCHE")
