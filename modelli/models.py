"""
Model definition.
"""
import json

import importlib
import modelli.layers 
importlib.reload(modelli.layers)
import tensorflow as tf
from keras.layers import Dense, LSTM, Bidirectional, Embedding,  Dropout, TimeDistributed
from keras import Input
from keras.layers import Concatenate
from keras import Model
from keras.models import model_from_json
import tensorflow_addons as tfa
from keras.metrics import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.losses import categorical_crossentropy


from modelli.layers import CRF


def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model


class BiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        inputs = [word_ids]
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=pred)

        return model, loss


class ELModel(object):
    """
    A Keras implementation of ELMo BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
        """
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_embedding_dim,
                                    mask_zero=True,
                                    name='char_embedding')(char_ids)
        char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)

        elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

        word_embeddings = Concatenate()([word_embeddings, char_embeddings, elmo_embeddings])

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation='tanh')(z)

        crf = CRF(self._num_labels, sparse_target=False)
        loss = crf.loss_function
        pred = crf(z)

        model = Model(inputs=[word_ids, char_ids, elmo_embeddings], outputs=pred)

        return model, loss


class Bi(Model):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        super().__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels

        # build word embedding
        if self._embeddings is None:
            self.word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')
        else:
            self.word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')

        # build character based word embedding
        if self._use_char:
            self.char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')
            self.char_bilstm = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))
            self.concat = Concatenate()

        self.dropout = Dropout(self._dropout)
        self.bilstm = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))
        self.dense = Dense(self._fc_dim, activation='tanh')

        if self._use_crf:
            self.last_layer = CRF(self._num_labels,sparse_target=False)
            self.loss = self.last_layer.loss_function

        else:
            self.loss = 'categorical_crossentropy'
            self.last_layer = Dense(self._num_labels, activation='softmax')


    def call(self, inputs, training=False):
        # word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
            # inputs = [word_ids]
        word_embeddings=self.word_embeddings(inputs[0])
        if self._use_char:
                # char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
                # inputs.append(char_ids)
            char_embeddings=self.char_embeddings(inputs[1])
            char_embeddings=self.char_bilstm(char_embeddings)
            word_embeddings = self.concat([word_embeddings, char_embeddings])
        word_embeddings=self.dropout(word_embeddings)
        z=self.bilstm(word_embeddings)
        self.z=self.dense(z)
        if self._use_crf:
            pred=self.last_layer(self.z,training=training)
        else:
            pred=self.last_layer(self.z)
        return pred

    def compute_loss(self, x, y, y_pred, sample_weight):
        return self.loss(y,y_pred, self.z)
    
    