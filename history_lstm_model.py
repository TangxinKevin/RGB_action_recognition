import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import Model
import numpy as np

def create_model(sequence_length, input_dimension, hidden_dimensions, class_number, tau):
    
    feature_input = layers.Input(shape=(sequence_length, input_dimension,))

    x = feature_input
    # multi layers lstm
    for k, dim in enumerate(hidden_dimensions):
        x = layers.CuDNNLSTM(dim,
                kernel_regularizer=regularizers.l2(l=0.0004),
                return_sequences=True, name='lstm_{}'.format(k))(x)
   
    # lstm_state: batch_size x seq_length x dimension
    lstm_state = x

    # y: batch_size x seq_length x class_number
    y = layers.TimeDistributed(layers.Dense(class_number,
                               activation='softmax'))(lstm_state)

    model = Model(inputs=feature_input, 
                  outputs=y)
    return model


if __name__ == '__main__':
    
    sequence_length = 50
    input_dimension = 2048
    hidden_dimensions = [120, 120, 120]
    class_number = 12
    tau = 5

    model = create_model(sequence_length,
                        input_dimension,
                        hidden_dimensions,
                        class_number,
                        tau)
    x = np.random.random((2, 50, 2048))
    y = model.predict(x, batch_size=2)

    z = model.get_layer('time_distributed').get_weights()
    print(y.shape)


    

    

    

