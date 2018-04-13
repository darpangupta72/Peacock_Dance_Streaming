from sklearn.externals import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from os import path, system
import time
from time import sleep
import numpy as np
import cv2

# time (in sec) for which to stream at a time, additional stream time if system goes from 2 to 0/1, sleep if state is 1, 0
stream = 15; stream_buffer = 10; sleep_nondance = 0; sleep_nopeacock = 0
# no. of the video device as seen in "ls -ltrh /dev/video*""
num_device = 1; 

# predict system state using probabilities for each class
def gen_label(predictions):

	if(predictions[0][2] > 0.93):
		return 2
	else:
		if(predictions[0][1] <= 0.049):
			return 0
		else:
			return 1

# predict system state for given image (.jpg)
def predict(img):

	img_dir = "/media/darpan/MIsc./Python Workspace/Peacock_Dance_Streaming/COP-FinalModel"
	# img_dir = "/home/pi/Desktop"
	img_path = path.join(img_dir, "frame_latest.jpg")

	# Read in the image_data
	img_data = tf.gfile.FastGFile(img_path, 'rb').read()

	# Feed the img_data as input to the graph and get prediction
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img_data})
	
	# print(predictions)
	pred = gen_label(predictions)
	return pred

if __name__ == "__main__":

	# setting up tensorflow
	label_lines = [line.rstrip() for line 
	                   in tf.gfile.GFile("output_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    _ = tf.import_graph_def(graph_def, name='')

	sess = tf.Session()
	# print("VIDEO STARTED")

	# state of system in {0, 1, 2} where 2 - dance, 1 - nondance, 0 - nopeacock
	prev_state = 0; curr_state = 0
	# free - whether the device is free (or busy)
	free = True; success = True; count = 1

	# repeat until device captures feed successfully
	while success:

		# capture feed if device is free
		if(free):
			vidcap = cv2.VideoCapture(num_device)
			free = False
		
		# generating system state from the video frame
		success, img = vidcap.read()
		# print('Read a new frame: ', count)
		cv2.imwrite("frame_latest.jpg", img)
		prev_state = curr_state
		curr_state = predict(img)
		# print(curr_state)

		# stream the video for some time if peacock is dancing
		if (curr_state == 2):
			vidcap.release()
			cv2.destroyAllWindows()
			free = True
			print("state 2 - dance, streaming for 15")
			system("ffmpeg -thread_queue_size 512 -f v4l2 -i /dev/video" + str(num_device) + " -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -acodec aac -ab 128k -strict experimental -aspect 16:9 -vcodec h264 -preset veryfast -crf 25 -pix_fmt yuv420p -vb 820k -r 30 -t " + str(stream) + " -f flv rtmp://a.rtmp.youtube.com/live2/ehhq-6vjg-qype-03e0")
		# peacock not dancing now
		else:
			# if peacock was dancing as per previous state, stream for some additional time
			if (prev_state == 2):
				vidcap.release()
				cv2.destroyAllWindows()
				free = True
				print("previous state 2, current state " + str(curr_state) + " - dance, streaming for 10")
				system("ffmpeg -thread_queue_size 512 -f v4l2 -i /dev/video" + str(num_device) + " -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -acodec aac -ab 128k -strict experimental -aspect 16:9 -vcodec h264 -preset veryfast -crf 25 -pix_fmt yuv420p -vb 820k -r 30 -t " + str(stream_buffer) + " -f flv rtmp://a.rtmp.youtube.com/live2/ehhq-6vjg-qype-03e0")
			# if peacock is present in the frame but not dancing, sleep
			elif (curr_state == 1):
				print("state 1 - nondance, sleeping for " + str(sleep_nondance))
				sleep(sleep_nondance)
			# if no peacock is present in the frame, sleep
			else:
				print("state 0 - nopeacock, sleeping for " + str(sleep_nopeacock))
				sleep(sleep_nopeacock)	
		
		the device sleeps from 7 pm to 6 am
		now = datetime.now()
		if (now.hour >= 19 or now.hour < 6):
			print("night")
			sleep((timedelta(hours = 24) - (now - now.replace(hour = 6, minute = 0, second = 0, microsecond = 0))).total_seconds() % (24 * 3600))

		count += 1

	# release the resource (device)
	vidcap.release()
	cv2.destroyAllWindows()