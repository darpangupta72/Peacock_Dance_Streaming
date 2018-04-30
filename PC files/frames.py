# required module cv2
import cv2
# replace 'peacock.mp4' by the video name for which frames are to be extracted
vidcap = cv2.VideoCapture('peacock.mp4')
success,image = vidcap.read()
# frame no.
count = 0
success = True
# read while true
while success:
  # read next frame from the video
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
