import nibabel as nib
import numpy as np
import tensorflow as tf
from skimage.measure import block_reduce
from sklearn.preprocessing import normalize
from os import listdir
from os.path import isfile,join
from os import rename
from sklearn.preprocessing import minmax_scale


structural_path = '/home/hngu4068/Documents/T1_BL_BET_FLIRT/'
structural_files = [structural_path+f for f in listdir(structural_path) if isfile(join(structural_path,f))]

N = len(structural_files)

img_data = np.zeros((N,91,109,91))
#load data and store it in an array
data = np.zeros((N,31,37,31))

for i in range(N):
    print "Opening file %d" %i
    file_name = structural_files[i]
    epi_img = nib.load(file_name)
    img_data[i,:,:,:]= epi_img.get_data()
    epi_img.uncache()
    #down sample the image
    data[i,:,:,:] = block_reduce(img_data[i,:,:,:],block_size = (3,3,3),func = np.mean)

print "Scale the data set"
# we will normalize the data set
flatten_data = np.zeros((N,31*37*31))
for i in range(N):
    flatten_data[i,:] = data[i,:,:,:].flatten()
data = minmax_scale(flatten_data,axis=1)


print "Crop the data set"
cropped_data = np.zeros((N,22*27*22))
for i in range(N):
    cropped_data[i,:] = np.reshape(data[i,:],(31,37,31))[4:26,5:32,4:26].flatten()

data = cropped_data


   
print "writing to tfrecords train"
np.random.shuffle(data)
writer = tf.python_io.TFRecordWriter("T1_BL_scalecropped2_train.tfrecords")
#iterate over each example
for i in range(0,N-6):
    temp_data = data[i,:]
    #construct the example proto object
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'image':tf.train.Feature(float_list=tf.train.FloatList(value = temp_data.astype(float)))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)

print "writing to tfrecords test"
writer = tf.python_io.TFRecordWriter("T1_BL_scalecropped2_test.tfrecords")
#iterate over each example
for i in range(N-6,N):
    temp_data = data[i,:]
    #construct the example proto object
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'image':tf.train.Feature(float_list=tf.train.FloatList(value = temp_data.astype(float)))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)



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
            'image': tf.FixedLenFeature([22*27*22], tf.float32)
        })
    # now return the converted data
    image = features['image']
    return image

print "Now reading file"
# returns symbolic label and image
image = read_and_decode_single_example("T1_BL_scalecropped2_train.tfrecords")

# groups examples into batches randomly
images_batch = tf.train.shuffle_batch(
    [image], batch_size=4,
    capacity=20,
    min_after_dequeue=4)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
images= sess.run([images_batch])
