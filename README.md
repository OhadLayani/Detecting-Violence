# Detecting-Violence
Violence Detection Neural Network
The attached files are of a computer vision based violence detection project.
the models used a public dataset known as RWF-2000 compiled for a similiar project , hte project itself used tensorflow
libraries such as keras with optical flow and rgb principles to identify violence from avi video files .

The dataset used is composed of approximiatly 2000  violent and non violent videos with 400 set aside for accuracy estimation and validation.


the avi files are extracted into frames which goes various data augmentation and preprocessing methods before the model is trained on them.
of the three models pure rgb had gotten the best result with 	 loss value of  0.1696  and  accuracy of approximatily  0.9372 after ten training epochs.

in contrast the optical flow and combined models accuracy suffered sitting at around (0.55 and 0.60 respecectivly) due to computing constraints causing the frame sizes to be resized ino a far smaller size (70,70 ) causing loss of  valuable information for training.


## Built With

* Python
* Keras for model training
* Tensorflow
* Cv2 for image capture



## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 





## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
