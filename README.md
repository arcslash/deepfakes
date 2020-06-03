# DeepFakes - Under Developement
Inspired from original /r/Deepfakes thread


Here contains, necessary scripting to implement your own deepfakes clone and tools necessary to train on your own.
Google Cloud environment is used to carry out training.

## Technologies and Frameworks Used
Pytorch


## What can be expected from this repo

### DeepFakes for Face Change
- [x] Adding Face Detection
- [ ] Developing GAN to generate faces
- [ ] Face / Object Substitution
- [ ] Training on New data
- [ ] UI interface to carryout training

### DeepFakes for Object change
Todo 

## How to get started

As the start make a virtual environment I'hv used python3 ( who use python2 anyway?) 
Make a virtual environment with whatever name you like, I'hv used env for simplicity, If you don't care you can directly install in the system as well.

on linux or mac os
```
python3 -m venv env
pip install -r requirements.txt
```

on windows

```
py -m venv env
pip install -r requirements.txt
```
I am running on Linux so could'nt test on other platforms, but it should work on those platforms as well (If not make a issue? or help me fix it?)
## Credits

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - Thanks for Amazing library for face detection