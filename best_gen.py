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
    kernel_shape = [4,5,4,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv1',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
    #deconv layer 2
    out_filter=38
    kernel_shape = [4,4,4,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv2',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter

    #deconv layer 3
    out_filter= 34
    kernel_shape = [3,3,3,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv3',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1],padding = 'SAME')
    in_filter = out_filter

    

    #deconv layer 4
    out_filter=28
    kernel_shape = [2,3,2,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv4',kernel_shape,biases_shape,prev_layer,strides = [1,2,2,2,1])
    in_filter = out_filter

    #deconv layer 5
    out_filter=24
    kernel_shape = [2,3,2,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv5',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
 
    #deconv layer 6
    out_filter = 12
    kernel_shape  =[2,2,2,out_filter,in_filter]
    biases_shape = [out_filter]
    prev_layer = _3D_deconv_layer('deconv6',kernel_shape,biases_shape,prev_layer)
    in_filter = out_filter
 
    #output_layer
    #mu = tf.reshape(fully_connected[:, :latent_dimension],[batch_size,latent_dimension])
    out_filter = 1
    kernel_shape = [2,2,2,out_filter,in_filter]
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
    prev_layer = _3D_conv_layer('conv5',kernel_shape,biases_shape,prev_layer)
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
