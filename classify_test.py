# import tensorflow as tf
# import sys

# # change this as you see fit
# image_path = sys.argv[1]

# # Read in the image_data
# image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# # Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line 
#                    in tf.gfile.GFile("/media/darpan/MIsc./Python Workspace/retrained_labels.txt")]

# # Unpersists graph from file
# with tf.gfile.FastGFile("/media/darpan/MIsc./Python Workspace/retrained_graph.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')

# with tf.Session() as sess:
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
#     predictions = sess.run(softmax_tensor, \
#              {'DecodeJpeg/contents:0': image_data})
    
#     # Sort to show labels of first prediction in order of confidence
#     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
#     for node_id in top_k:
#         human_string = label_lines[node_id]
#         score = predictions[0][node_id]
#         print('%s (score = %.5f)' % (human_string, score))

import tensorflow as tf
import sys, os

thres = 0.65; dcorrect = 0; ndcorrect = 0

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/media/darpan/MIsc./Python Workspace/tensorflow-master/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/media/darpan/MIsc./Python Workspace/tensorflow-master/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

dance_cnt = 0
dancedir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/peacock_test/dance"
for file in os.listdir(dancedir):

    dance_cnt = dance_cnt + 1
    print(dance_cnt)

    # change this as you see fit
    image_path = os.path.join(dancedir, file)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        #print(predictions[0][0])
        if(predictions[0][0] >= thres):
            dcorrect = dcorrect + 1

nondance_cnt = 0        
nondancedir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/peacock_test/nondance"
for file in os.listdir(nondancedir):

    nondance_cnt = nondance_cnt + 1
    print(nondance_cnt)
    
    # change this as you see fit
    image_path = os.path.join(nondancedir, file)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        #print(predictions[0][0])
        if(predictions[0][0] < thres):
            ndcorrect = ndcorrect + 1

print("\nCONFUSION MATRIX IS: \n")
print("\tactual/predicted\t\tdancing \t non-dancing\n")
print("\tdancing\t\t\t\t   {x}    \t\t{y}".format(x = dcorrect, y = dance_cnt - dcorrect))
print("\tnon-dancing\t\t\t   {x}    \t\t{y}\n".format(x = nondance_cnt - ndcorrect, y = ndcorrect))
print("Classification accuracy for total {x} frames is {y}\n".format(x = dance_cnt + nondance_cnt, y = (dcorrect + ndcorrect) / (dance_cnt + nondance_cnt)))
