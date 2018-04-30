# Peacock_Dance_Streaming

A device including a camera and a Raspberry Pi will be set up at a location where peacocks frequent. The aim of the project is to detect if there is a peacock present at the location and live stream a video (on the internet) when the peacock starts dancing at the location.


To begin with, you need to have installed python v3.5.3 and the latest version of pip.
And then for training the model and running the files you need to have the packages given in the "requirements.txt" installed.
Use the following command to do the same

sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python3.5

sudo apt-get install pip

sudo pip install -r requirements.txt

Now, you have all the pre-requisite files installed. 

Training/testing data can be created by extracting frames from relevant videos. Code is given in frames.py. It can be run using the following command:

python3.5 frames.py

Now train the model using the images from the training dataset using the command given in the "tf_train" file given in the PC train folder(Change the pwd with the location of training dataset)
Now for testing the model on the test dataset run the "classify_test_new.py" file in the PC train folder.
Edit the pwd in the "classify_test_new.py" with the current location of test dataset.
Then run

python3.5 classify_test_new.py

These are the requirements for training and testing your model. This can be done on a remote desktop or even the raspberry pi. Now for porting the model onto raspberry pi you need to run the same commands as given previously other than the training and testing of the model. This will install all the pre-requisites to run the model.
Now, run

sudo apt-get update

sudo apt-get upgrade

Now install ffmpeg(Normally it would already be installed, if you are using raspian on a raspberrypi3). In any case, use the following command:

sudo apt-get install ffmpeg

Now, the only part remaining is to hook your pi to the usb camera and internet and copy the "pd_stream.py" onto your pi. The whole code has been properly commented in case you run into any difficulties. Change the pwd with the current working directory and run the command

python3.5 pd_stream.py

This should start the model and it'll start detecting peacock dance images and correspondingly stream them online when a peacock is detected(Also, you have to use the stream key of your own youtube account and so edit it correspondingly).
If you still encounter any more problems, go to our blog for this project, in which every part of how to recreate the project is given in much detail. The link is

https://peacockdancecop315.wordpress.com/2018/04/29/peacock-dance-streaming/
