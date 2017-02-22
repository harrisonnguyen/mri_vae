from __future__ import print_function

import tensorflow as tf
import edward as ed
import os
import numpy as np

from edward.models import Normal
from edward.models import Bernoulli
from tensorflow.python.framework import tensor_shape
"""
Helper methods
"""
def _weight_variable(name, shape):
    """
    Creates a weight variable initialised with truncated normal of stddev = 0.1
    :param name: string, name of the weight
    :param shape: array of int, shape of the weight, 5D for 3D convolutions
    :return: tensor
    """
    w = tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.02))
    tf.add_to_collection('vars',w)
    return w

def _weight_variable_weight(name, shape,weight):
    """
    Creates a weight variable initialised with truncated normal of stddev = 0.1
    :param name: string, name of the weight
    :param shape: array of int, shape of the weight, 5D for 3D convolutions
    :param weight: array of float, values of initialised variable
    :return: tensor
    """
    w = tf.get_variable(name, shape, tf.float32, initializer=tf.constant_initializer(weight))
    tf.add_to_collection('vars',w)
    return w


def _bias_variable(name, shape):
    """
    Creates a bias variable initialised with truncated normal of stddev = 0.1
    :param name: string, name of the bias
    :param shape: array of int, shape of the bias
    :return: tensor
    """
    #tf.constant_initializer(0.1, dtype=tf.float32)
    b = tf.get_variable(name, shape,tf.float32,tf.constant_initializer(0.1, dtype=tf.float32))
    tf.add_to_collection('vars',b)
    return b

def _bias_variable_weight(name, shape,weight):
    """
    Creates a bias variable initialised with truncated normal of stddev = 0.1
    :param name: string, name of the bias
    :param shape: array of int, shape of the bias
    :return: tensor
    """
    #tf.constant_initializer(0.1, dtype=tf.float32)
    b = tf.get_variable(name, shape,tf.float32,tf.constant_initializer(weight))
    tf.add_to_collection('vars',b)
    return b

def _get3d_deconv_output_size(input_height, input_width,input_depth, filter_height,
                             filter_width,filter_depth, row_stride, col_stride,depth_stride,
                             padding_type):
    """
    Returns the number of rows and columns in a convolution/pooling output.
    :param input_height: int, input layer height
    :param input_width: int, input layer width
    :param input_depth: int, input layer depth
    :param filter_height: int, filter height
    :param filter_width: int, filter width
    :param filter_depth: int, filter depth
    :param row_stride: int, row stride
    :param col_stride: int, column stride
    :param depth_stride: int, depth stride
    :param padding_type: string, either 'VALID' or 'SAME'
    :return: int
    """
    input_height = tensor_shape.as_dimension(input_height)
    input_width = tensor_shape.as_dimension(input_width)
    input_depth = tensor_shape.as_dimension(input_depth)
    filter_height = tensor_shape.as_dimension(filter_height)
    filter_width = tensor_shape.as_dimension(filter_width)
    filter_depth = tensor_shape.as_dimension(filter_depth)
    row_stride = int(row_stride)
    col_stride = int(col_stride)
    depth_stride = int(depth_stride)

    # Compute number of rows in the output, based on the padding.
    if input_height.value is None or filter_height.value is None:
        out_rows = None
    elif padding_type == "VALID":
        out_rows = (input_height.value - 1) * row_stride + filter_height.value
    elif padding_type == "SAME":
        out_rows = input_height.value * row_stride
    else:
        raise ValueError("Invalid value for padding: %r" % padding_type)

    # Compute number of columns in the output, based on the padding.
    if input_width.value is None or filter_width.value is None:
        out_cols = None
    elif padding_type == "VALID":
        out_cols = (input_width.value - 1) * col_stride + filter_width.value
    elif padding_type == "SAME":
        out_cols = input_width.value * col_stride
        
    # Compute number of columns in the output, based on the padding.
    if input_depth.value is None or filter_depth.value is None:
        out_depth = None
    elif padding_type == "VALID":
        out_depth = (input_depth.value - 1) * depth_stride + filter_depth.value
    elif padding_type == "SAME":
        out_depth = input_depth.value * depth_stride


    return out_rows, out_cols,out_depth


## creates a deconvolution layer with specific activation function
def _3D_deconv_layer(name, kernel_shape, biases_shape,prev_layer, activation_fn = tf.nn.elu,
                  strides = [1, 1, 1,1,1], padding = 'VALID',kernel_weight = None,bias_weight = None):
    """
    Wrapper for a 3D deconvolutional layer
    :param name: string, name of the layer
    """
    
    with tf.variable_scope(name) as scope:

        #if a weight variable is given, initialise kernels accordingly
        if kernel_weight is None:
            kernel = _weight_variable('weights',kernel_shape)
        else:
            kernel = _weight_variable_weight('weights',kernel_shape,kernel_weight)

        #add tensor to collection
        tf.add_to_collection('deconv_weights',kernel)

        if bias_weight is None:
            biases = _bias_variable('biases',biases_shape)
        else:
            biases = _bias_variable_weight('biases',biases_shape,bias_weight)
        
        out_row,out_col,out_depth = _get3d_deconv_output_size(prev_layer.shape[1],prev_layer.shape[2],prev_layer.shape[3],kernel_shape[0],kernel_shape[1],kernel_shape[2],strides[1],strides[2],strides[3],padding)
        output_shape = [batch_size,out_row,out_col,out_depth,kernel_shape[3]]
        #apply weights and biases        
        conv = tf.nn.conv3d_transpose(prev_layer,kernel,output_shape, strides,padding=padding,name=scope.name)
        conv = tf.nn.bias_add(conv,biases)
        #apply activation function
        if activation_fn is not None:
            conv = activation_fn(conv)
        #add output to collection
        tf.add_to_collection('deconv_output',conv)
    return conv

##creates a convolution layer with specific activation function
def _3D_conv_layer(name, kernel_shape, biases_shape,prev_layer, activation_fn = tf.nn.elu,
                  strides = [1, 1, 1, 1, 1], padding = 'VALID',kernel_weight = None,bias_weight=None):
    with tf.variable_scope(name) as scope:
        
        #if a weight variable is given, initialise kernels accordingly
        if kernel_weight is None:
            kernel = _weight_variable('weights',kernel_shape)
        else:
            kernel = _weight_variable_weight('weights',kernel_shape,kernel_weight)

        #add tensor to collection
        tf.add_to_collection('conv_weights',kernel)

        if bias_weight is None:
            biases = _bias_variable('biases',biases_shape)
        else:
            biases = _bias_variable_weight('biases',biases_shape,bias_weight)

        #apply weights and biases
        conv = tf.nn.conv3d(prev_layer,kernel,strides,padding=padding)
        conv = tf.nn.bias_add(conv,biases)
        #apply activation function
        if activation_fn is not None:
            conv = activation_fn(conv,name = scope.name)
        # add output to collection
        tf.add_to_collection('conv_output',conv)
    return conv


# we take a random variable z~N(0,1)
# pass through deconvolutional network
# to produce a mean and sigma for each voxel
def generative_network(z):
    """Generative network to parameterize the generative model
    It takes latent variables as input and outputs the likelihood parameters
        
    mu = neural_network(z) sigma = neural_network(z)
    """
    reshape_z = tf.reshape(z,[batch_size,1,1,1,latent_dimension])
    prev_layer = reshape_z
    in_filter = latent_dimension
   
    #deconv layer 1
    out_filter=42
    kernel_shape = [4,8,4,out_filter,in_filter]
    biases_shape = [out_filter]
    kernel_weights = np.load('deconv_weights0.npy')
    bias_weights = np.load('deconv_bias0.npy')
    prev_layer = _3D_deconv_layer('deconv1',kernel_shape,biases_shape,prev_layer,kernel_weight = kernel_weights,bias_weight = bias_weights)
    in_filter = out_filter

    #deconv layer 2
    out_filter=38
    kernel_shape = [4,8,4,out_filter,in_filter]
    biases_shape = [out_filter]
    kernel_weights = np.load('deconv_weights1.npy')
    bias_weights = np.load('deconv_bias1.npy')
    prev_layer = _3D_deconv_layer('deconv2',kernel_shape,biases_shape,prev_layer,kernel_weight = kernel_weights,bias_weight = bias_weights)
    in_filter = out_filter

    #deconv layer 3
    out_filter=24
    kernel_shape = [3,5,3,out_filter,in_filter]
    biases_shape = [out_filter]
    kernel_weights = np.load('deconv_weights2.npy')
    bias_weights = np.load('deconv_bias2.npy')
    prev_layer = _3D_deconv_layer('deconv3',kernel_shape,biases_shape,prev_layer,kernel_weight = kernel_weights,bias_weight = bias_weights)
    in_filter = out_filter

    #deconv layer 4
    out_filter=20
    kernel_shape = [3,3,3,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv4',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1],padding = 'SAME')
    in_filter = out_filter

    #deconv layer 5
    out_filter=16
    kernel_shape = [6,1,6,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv5',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    
    #deconv layer 6
    out_filter = 12
    kernel_shape  =[6,1,6,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv6',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter

    #output_layer
    out_filter = 1
    kernel_shape = [4,0,4,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('gen_output',kernel_shape,biases_shape,prev_layer,activation_fn=tf.nn.sigmoid)

    mu = tf.reshape(prev_layer,[batch_size,x_shape[0],x_shape[1],x_shape[2],1])
    #mu = tf.reshape(fully_connected,[batch_size,x_shape[0],x_shape[1],x_shape[2],1])
    #sigma =  tf.reshape(tf.nn.softplus(output[:,:,:,:,1]),[batch_size,x_shape[0],x_shape[1],x_shape[2],1]) + 1e-10

    return mu#,sigma

def inference_network(x,latent_dimension = 100):
    """Inference network to parameterize the variational family
    it takes data as input and outputs the variational parameters
    
    
    the network follows the architecture given by 
    Deep MRI brain extraction: A 3D convolutional neural network for
skull stripping Kleesiek, J 2016

    mu,sigma = encoder_neural_network(x)
    """
    
    latent_dimension = latent_dimension
    in_filter = 1

    #layer 1
    out_filter = 24
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    kernel_weights = np.load('conv_weights0.npy')
    bias_weights = np.load('conv_bias0.npy')
    prev_layer = _3D_conv_layer('conv1',kernel_shape,biases_shape,x,kernel_weight = kernel_weights,bias_weight = bias_weights)
    in_filter = out_filter
    
    #do max pooling
    pool1 = tf.nn.max_pool3d(prev_layer,ksize = [1,2,2,2,1],
                             strides = [1,2,2,2,1],
                            padding= 'SAME')
    
    prev_layer = pool1
    
    #layer 2
    out_filter = 28
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    kernel_weights = np.load('conv_weights1.npy')
    bias_weights = np.load('conv_bias1.npy')
    prev_layer = _3D_conv_layer('conv2',kernel_shape,biases_shape,prev_layer,kernel_weight = kernel_weights,bias_weight = bias_weights)
    in_filter = out_filter
 
    #layer 3
    out_filter = 32
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv3',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    #layer 4
    out_filter = 34
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv4',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    """
    #layer 5
    out_filter = 42
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv5',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter

  
    #layer 6
    out_filter = 50
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv6',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    #layer 7
    out_filter = 50
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv7',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    """

    #fully connected layer
    dim = tensor_shape.as_dimension(prev_layer.shape[1])*tensor_shape.as_dimension(prev_layer.shape[2])*tensor_shape.as_dimension(prev_layer.shape[3])*tensor_shape.as_dimension(prev_layer.shape[4])
    dim = int(dim)
    prev_layer_flat = tf.reshape(prev_layer, [batch_size, dim])

    weights = _weight_variable('weights_full', [dim, latent_dimension*3])
    biases = _bias_variable('biases_full', [latent_dimension*3])
    fully_connected = tf.matmul(prev_layer_flat, weights) + biases
    
    mu = tf.reshape(fully_connected[:, :latent_dimension],[batch_size,latent_dimension])
    #logsigma2 = tf.reshape(fully_connected[:,latent_dimension:],[batch_size,latent_dimension]) + 1e-10
    sigma = tf.reshape(tf.nn.softplus(fully_connected[:, latent_dimension:latent_dimension*2]),[batch_size,latent_dimension])

    #h = tf.reshape(fully_connected[:, latent_dimension*2:latent_dimension*3],[batch_size,latent_dimension])
    """
    fully_connected = prev_layer
    mu = tf.reshape(fully_connected[:,:,:,:,0],[batch_size,latent_dimesnion])
    signa = tf.reshape(tf.nn.softplus(fully_connected[:,:,:,:,1]),[batch_size,latent_dimension])+1e-10
    """
    return mu,sigma 


def AutoregressiveNN(z,h, l):
    
    prev_layer = tf.concat([z,h],axis=1)
    dim = int(tensor_shape.as_dimension(latent.shape[1]))
    with tf.variable_scope('IAF_1') as scope:
        weights = _weight_variable('weights', [dim, 500])
        biases = _weight_variable('biases', [500])
        prev_layer = tf.nn.tanh(tf.matmul(prev_layer,weights) + biases)
    
    with tf.variable_scope('IAF_2') as scope:
        weights = _weight_variable('weights', [dim, latent_dimension*2])
        biases = _weight_variable('biases', [latent_dimension*2])
        prev_layer = tf.matmul(prev_layer,weights) + biases
    
    m = tf.reshape(fully_connected[:, :latent_dimension],[batch_size,latent_dimension])
    sigma = tf.reshape(tf.nn.sigmoid(fully_connected[:, :latent_dimension]),[batch_size,latent_dimension])
    z = tf.add(tf.multiply(sigma,z),tf.multiply(1-sigma,m))
    
    #l is logq(z_t|x)
    l = l - tf.reduce_sum(tf.log(sigma),1)
    #n_IAF = n_IAF - 1
    return z, sigma,l


def VAE_loss(x_ph,x_mu,z_mu,z_sigma,global_step,  learning_rate_initial = 0.01,learning_decay = 0.96):
    reconstruct_loss = -tf.reduce_sum(x_ph*tf.log(x_mu +1e-8) + \
                                    (1-x_ph)*tf.log(1-x_mu+1e-8),axis = 1)
    
    #latent_loss = -0.5*tf.reduce_sum(1+z_logsigma2
                               #- tf.square(z_mu)
                               #-tf.exp(z_logsigma2),axis=1)
    latent_loss = -0.5*tf.reduce_sum(1+tf.log(tf.square(z_sigma)+1e-10)
                               - tf.square(z_mu)
                               -tf.square(z_sigma),axis=1)
    #IAF_loss = l
    cost = tf.reduce_mean(reconstruct_loss + latent_loss)
    learning_rate = tf.train.exponential_decay(learning_rate_initial,global_step,10,learning_decay)
    optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,global_step = global_step)
    return cost,optimiser

def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'image': tf.FixedLenFeature([x_shape[0]*x_shape[1]*x_shape[2]], tf.float32)
        })
    # now return the converted data
    image = features['image']
    return image
"""
 Global variables
"""
ed.set_seed(42)
latent_dimension = 100
batch_size = 4
batch_size_test = 4
#x_shape = [91,109,91]
x_shape = [31,37,31]
#x_shape = [10,20,10]
#x_shape = [22,27,22]
n_IAF = 3

"""
The model
"""
"""
z = Normal(mu = tf.zeros([batch_size,latent_dimension]), sigma = tf.ones([batch_size,latent_dimension]))
#mu_x,sigma_x = generative_network(z)
mu_x = generative_network(z)
#x = Normal(mu = mu_x,sigma = sigma_x)
x = Bernoulli(logits = mu_x)
##inference
x_ph = tf.placeholder(tf.float32, [batch_size, x_shape[0],x_shape[1],x_shape[2],1])
mu, sigma = inference_network(x_ph)
#we use mean field approximation
qz = Normal(mu=mu, sigma=sigma)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz},data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)
"""
global_step = tf.Variable(0,trainable=False)
no_IAF = tf.Variable(n_IAF,trainable=False)

x_ph = tf.placeholder(tf.float32,[None, x_shape[0],x_shape[1],x_shape[2],1])
z_mu,z_sigma = inference_network(x_ph,latent_dimension = latent_dimension)
epsilon = tf.random_normal([batch_size,latent_dimension],0,1,dtype=tf.float32)
z = tf.add(tf.multiply(z_sigma,epsilon),z_mu)
#l = -tf.reduce_sum(tf.add(tf.log(z_sigma),0.5*tf.square(epsilon)),1) -latent_dimension * 0.5* np.log(2*np.pi)
#do the IAF
#z_1,sigma_1,l_1 = AutoregressiveNN(z,h,l)
#z_t,_,l_t =  tf.while_loop(condition,AutoregressiveNN,[z,h,l,])
mu_x = generative_network(z)
#cost,optimiser = VAE_loss(x_ph,mu_x,z_mu,z_logsigma2)
cost,optimiser = VAE_loss(x_ph,mu_x,z_mu,z_sigma,global_step)

## IMPORt the data
# returns symbolic label and image
image = read_and_decode_single_example("T1_BL_scale_train.tfrecords")
# groups examples into batches randomly
images_batch = tf.train.shuffle_batch(
    [image], batch_size=batch_size,
    capacity=20,
    min_after_dequeue=batch_size)

image_test = read_and_decode_single_example("T1_BL_scale_test.tfrecords")
# groups examples into batches randomly
images_batch_test = tf.train.shuffle_batch(
    [image_test], batch_size=batch_size_test,
    capacity=20,
    min_after_dequeue=batch_size_test)



saver = tf.train.Saver()
##Lets train the model!
with tf.Session() as session:
    
    session.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,coord=coord)
    n_epoch = 30
    n_iter_per_epoch =  int(86/batch_size)
    for epoch in range(n_epoch):
        avg_loss = 0.0
        for t in range(n_iter_per_epoch):
            x_train= session.run([images_batch])[0]
            x_train = np.reshape(x_train,(batch_size,x_shape[0],x_shape[1],x_shape[2],1))
            cost_val,_, temp = session.run([cost,optimiser,z],feed_dict={x_ph:x_train})
            #info_dict = inference.update(feed_dict={x_ph: x_train})
            #avg_loss += info_dict['loss']
            avg_loss +=cost_val
            #print(cost_val)
            #print(temp)
            # Print a lower bound to the average marginal likelihood for an
            # image.
        avg_loss = avg_loss / n_iter_per_epoch
        #avg_loss = avg_loss / batch_size
        #avg_loss = avg_loss / epoch
        print("log p(x) >= {:0.3f}".format(avg_loss))



        saver.save(session,'mri_vae_model/my-model-big.ckpt',write_meta_graph=True,global_step = epoch)
    
    #see if the reconstruction is any good
    x_test= session.run([images_batch_test])[0]
    x_test = np.reshape(x_test,(batch_size,x_shape[0],x_shape[1],x_shape[2],1))
    reconstruct_x = session.run(mu_x, feed_dict = {x_ph:x_test})
    np.save('x_test',x_test)
    np.save('reconstruct_x',reconstruct_x)

    #calculate the test reconstruction error
    avg_loss = 0.0
    x_test= session.run([images_batch_test])[0]
    x_test = np.reshape(x_train,(batch_size_test,x_shape[0],x_shape[1],x_shape[2],1))
    cost_val,_, temp = session.run([cost,optimiser,z],feed_dict={x_ph:x_test})
    avg_loss +=cost_val
    avg_loss = avg_loss / n_iter_per_epoch
    print("log p(x) >= {:0.3f}".format(avg_loss))

    coord.request_stop()
    coord.join(threads)
