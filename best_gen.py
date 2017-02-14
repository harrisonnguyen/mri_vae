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
