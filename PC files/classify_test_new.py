import tensorflow as tf
import numpy as np
import sys, os
import pandas
from sklearn import tree
from sklearn.externals import joblib

# 2 - dance, 1 - nondance, 0 - others
conf_matrix = np.zeros([3, 3])

def update_conf(predictions, i):

    ans = -1
    if(predictions[0][2] > 0.93):
    	ans = 2
    else:
    	if(predictions[0][1] <= 0.049):
    		ans = 0
    	else:
    		ans = 1

    conf_matrix[i][ans] += 1
    print('prediction: ' + str(ans))
    return ans

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.Session()

cnt = 0
dancedir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/COP-FinalModel/peacock_test/dance"
for file in os.listdir(dancedir):

    # change this as you see fit
    image_path = os.path.join(dancedir, file)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    update_conf(predictions, 2)

    cnt = cnt + 1
    print(cnt)

cnt = 0        
nondancedir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/COP-FinalModel/peacock_test/nondance"
for file in os.listdir(nondancedir):

    # change this as you see fit
    image_path = os.path.join(nondancedir, file)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    update_conf(predictions, 1)

    cnt = cnt + 1
    print(cnt)

cnt = 0
othersdir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/COP-FinalModel/peacock_test/nopeacock"
for file in os.listdir(othersdir):

    # change this as you see fit
    image_path = os.path.join(othersdir, file)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    update_conf(predictions, 0)

    cnt = cnt + 1
    print(cnt)

sess.close()

print("\nClassification accuracy is {x}".format(x = (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]) / np.sum(np.sum(conf_matrix))))
print("\nCONFUSION MATRIX IS: \n")
print(conf_matrix)