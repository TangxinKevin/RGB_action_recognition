
import math
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import functools
import numpy as np

def create_new_model(seqence_length, input_dimension, hidden_dimensions, class_number,tau):
    feature_input = tf.keras.layers.Input(shape=(seqence_length, input_dimension,))
    x = feature_input
    for k in hidden_dimensions:
        x = tf.keras.layers.CuDNNLSTM(k, 
            kernel_regularizer=tf.keras.regularizers.l2(l=0.0004),
            return_sequences=True)(x)
    layeroutput=[]
    layererror=[]
    herror=[]
    houtput=[]
    x = tf.transpose(x, [0, 2, 1])
    y = tf.keras.layers.Dense(class_number, activation='softmax')(x)      
    for i in range(0,seqence_length-1):
        last = tf.gather(x, [i],axis=-1)
        label = tf.gather(y,[i],axis=1)
        layeroutput.append(last)
        llogits = layers.fully_connected(last, int(label.get_shape()[1]),activation_fn='softmax')
        mistakes =tf.nn.softmax_cross_entropy_with_logits_v2(logits=llogits, labels=label)
        layererror.append(mistakes)  
        if i==0:
            houtput.append(last)
            herror.append(layererror[i])
        if ((i>0) and (herror[i-1]<=layererror[i])) is not None:
             alpha=herror[i-1]/(herror[i-1]+layererror[i])
             houtput.append(alpha*last+(1-alpha)*houtput[i-1])          
        if ((i>0) and (herror[i-1]>layererror[i])) is not None:
             temp=layeroutput[0]*0
             for ii in range(0,i):
                 if (ii>=tau) is not None:
                     temp=temp+layeroutput[ii]
             houtput.append(temp/(i-tau))            
        hlast=houtput[i]
        hlogits = layers.fully_connected(hlast,    int(label.get_shape()[1]),    activation_fn='softmax')
        hmistakes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hlogits, labels=label)
        herror.append(tf.reduce_mean(tf.cast(hmistakes, tf.float32)))  

    y = tf.keras.layers.Dense(class_number, activation='softmax')(hlogits)      
    model = tf.keras.Model(inputs=feature_input, 
                           outputs=y)
    print(model)








    return model, 

if __name__ == '__main__':
    inputs = np.random.random((2, 50, 2048))
    inputs.dtype = 'float32'
    model = create_new_model(50, 2048, [90, 90, 90], 12,3)
#    y = model(inputs)
#    print(y[0].shape)
#    print(y[1].shape)