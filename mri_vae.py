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
##creates a weight variable with name and shape
def _weight_variable(name, shape):
    w = tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    tf.add_to_collection('vars',w)
    return w


def _bias_variable(name, shape):
    b = tf.get_variable(name, shape,tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))
    tf.add_to_collection('vars',b)
    return b

def _get3d_deconv_output_size(input_height, input_width,input_depth, filter_height,
                             filter_width,filter_depth, row_stride, col_stride,depth_stride,
                             padding_type):
    """Returns the number of rows and columns in a convolution/pooling output.
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
                  strides = [1, 1, 1,1,1], padding = 'VALID'):
    
    with tf.variable_scope(name) as scope:
        kernel = _weight_variable('weights',kernel_shape)
        biases = _bias_variable('biases',biases_shape)
        
        out_row,out_col,out_depth = _get3d_deconv_output_size(prev_layer.shape[1],prev_layer.shape[2],prev_layer.shape[3],kernel_shape[0],kernel_shape[1],kernel_shape[2],strides[1],strides[2],strides[3],padding)
        output_shape = [batch_size,out_row,out_col,out_depth,kernel_shape[3]]
        conv = tf.nn.conv3d_transpose(prev_layer,kernel,output_shape, strides,padding=padding,name=scope.name)
        conv = tf.nn.bias_add(conv,biases)
        if activation_fn is not None:
            conv = activation_fn(conv)
    return conv

##creates a convolution layer with specific activation function
def _3D_conv_layer(name, kernel_shape, biases_shape,prev_layer, activation_fn = tf.nn.softplus,
                  strides = [1, 1, 1, 1, 1], padding = 'VALID'):
    with tf.variable_scope(name) as scope:
        kernel = _weight_variable('weights',kernel_shape)
        biases = _bias_variable('biases',biases_shape)
        conv = tf.nn.conv3d(prev_layer,kernel,strides,padding=padding)
        conv = tf.nn.bias_add(conv,biases)
        if activation_fn is not None:
            conv = activation_fn(conv,name = scope.name)
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
    out_filter=10
    kernel_shape = [17,19,17,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv1',kernel_shape,biases_shape,prev_layer,tf.nn.tanh)
    in_filter = out_filter
    #deconv layer 2
    out_filter=1
    kernel_shape = [15,19,15,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv2',kernel_shape,biases_shape,prev_layer,activation_fn=tf.nn.sigmoid)
    in_filter = out_filter
    """
    #deconv layer 3
    out_filter=5
    kernel_shape = [5,5,5,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv3',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1])
    in_filter = out_filter
    #deconv layer 4
    out_filter=1
    kernel_shape = [3,3,3,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv4',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1],padding = 'SAME')
    in_filter = out_filter
    """
    """
    #deconv layer 5
    out_filter=3
    kernel_shape = [5,7,5,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv5',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
   
    #deconv layer 6
    out_filter = 12
    kernel_shape  =[10,10,10,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv6',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    """
    """
    #output_layer
    mu = tf.reshape(fully_connected[:, :latent_dimension],[batch_size,latent_dimension])
    out_filter = 1
    kernel_shape = [2,4,2,out_filter,in_filter]
    biases_shape = [out_filter]
    output = _3D_deconv_layer('gen_output',kernel_shape,biases_shape,prev_layer,activation_fn=tf.nn.sigmoid)
    """
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
    #reshaped_x = tf.reshape(x,[batch_size,tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],in_filter])
    #layer 1
    out_filter = 24
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv1',kernel_shape,biases_shape,x)
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
    prev_layer = _3D_conv_layer('conv2',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    
    
    #layer 3
    out_filter = 24
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv3',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    
    #layer 4
    out_filter = 34
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv4',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1])
    in_filter = out_filter

    #layer 5
    out_filter = 42
    kernel_shape = [3,3,3,in_filter,out_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_conv_layer('conv5',kernel_shape,biases_shape,prev_layer,activation_fn=tf.nn.tanh)
    in_filter = out_filter
    """
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

    weights = _weight_variable('weights_full', [dim, latent_dimension*2])
    biases = _bias_variable('biases_full', [latent_dimension*2])
    fully_connected = tf.matmul(prev_layer_flat, weights) + biases
    
    mu = tf.reshape(fully_connected[:, :latent_dimension],[batch_size,latent_dimension])
    #logsigma2 = tf.reshape(fully_connected[:,latent_dimension:],[batch_size,latent_dimension]) + 1e-10
    sigma = tf.reshape(tf.nn.softplus(fully_connected[:, :latent_dimension]),[batch_size,latent_dimension])
    """
    fully_connected = prev_layer
    mu = tf.reshape(fully_connected[:,:,:,:,0],[batch_size,latent_dimesnion])
    signa = tf.reshape(tf.nn.softplus(fully_connected[:,:,:,:,1]),[batch_size,latent_dimension])+1e-10
    """
    return mu,sigma

def VAE_loss(x_ph,x_mu,z_mu,z_sigma,global_step,  learning_rate_initial = 0.01,learning_decay = 0.96):
    reconstruct_loss = -tf.reduce_sum(x_ph*tf.log(x_mu +1e-8) + \
                                    (1-x_ph)*tf.log(1-x_mu+1e-8),axis = 1)
    
    #latent_loss = -0.5*tf.reduce_sum(1+z_logsigma2
                               #- tf.square(z_mu)
                               #-tf.exp(z_logsigma2),axis=1)
    latent_loss = -0.5*tf.reduce_sum(1+tf.log(tf.square(z_sigma)+1e-10)
                               - tf.square(z_mu)
                               -tf.square(z_sigma),axis=1)
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
#x_shape = [91,109,91]
x_shape = [31,37,31]

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
x_ph = tf.placeholder(tf.float32,[None, x_shape[0],x_shape[1],x_shape[2],1])
#z_mu,z_logsigma2 = inference_network(x_ph)
z_mu,z_sigma = inference_network(x_ph)
epsilon = tf.random_normal([batch_size,latent_dimension],0,1,dtype=tf.float32)
#z = tf.add(tf.multiply(tf.sqrt(tf.exp(z_logsigma2)),epsilon),z_mu)
z = tf.add(tf.multiply(z_sigma,epsilon),z_mu)
mu_x = generative_network(z)
#cost,optimiser = VAE_loss(x_ph,mu_x,z_mu,z_logsigma2)
cost,optimiser = VAE_loss(x_ph,mu_x,z_mu,z_sigma,global_step)

## IMPORt the data
# returns symbolic label and image
image = read_and_decode_single_example("T1_mri_full_BL_normalizedscale_train.tfrecords")
# groups examples into batches randomly
images_batch = tf.train.shuffle_batch(
    [image], batch_size=batch_size,
    capacity=20,
    min_after_dequeue=batch_size)

##Lets train the model!
saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,coord=coord)
    n_epoch = 50
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



        saver.save(session,'/project/RDS-SMS-NEUROIMG-RW/harrison/mri_vae_model/my-model.ckpt',write_meta_graph=True,global_step = epoch)
    
    #see if the reconstruction is any good
    x_train= session.run([images_batch])[0]
    x_train = np.reshape(x_train,(batch_size,x_shape[0],x_shape[1],x_shape[2],1))
    reconstruct_x = session.run(mu_x, feed_dict = {x_ph:x_train})
    np.save('reconstruct_x',reconstruct_x)

    coord.request_stop()
    coord.join(threads)
