# DeepFakes - Under Developement
Inspired from original /r/Deepfakes thread


Here contains, necessary scripting to implement your own deepfakes clone and tools necessary to train on your own.
Google Cloud environment is used to carry out training.

**If you like to contribute to the development of the project - feel free to join**

## Technologies and Frameworks Used
Pytorch


## What can be expected from this repo

### DeepFakes for Face Change
- [x] Adding Face Detection
- [ ] Adding Face Tracking through Video Frames
- [ ] Developing GAN to face style transfer
- [ ] Face Substitution
- [ ] Training on New data
- [ ] UI interface to carryout training

### DeepFakes for Object change
- [ ] Training Interface for Object Detection
- [ ] Object Tracking and Detection
- [ ] Developing GAN to style transfer
- [ ] Object Substitution

## How to get started

As the start make a virtual environment i've used python3 ( who use python2 anyway?) 
Make a virtual environment with whatever name you like, i've used env for simplicity, If you don't care you can directly install in the system as well.
Then Install the requirements as specified in requirements.txt

on linux or mac os
```
python3 -m venv env
pip3 install -r requirements.txt
```

on windows

```
py -m venv env
pip3 install -r requirements.txt
```
I am running on Linux so could'nt test on other platforms, but it should work on those platforms as well (If not make a issue? or help me fix it?)
## Credits

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - Thanks for Amazing library for face detection
