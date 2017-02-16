# Machine Learning for Computer Vision (M3) - Master in Computer Vision, UAB
Project for the Machine Learning for Computer Vision module in the Master in Computer Vision by Universitat Autònoma de Barcelona. 

## Authors

- [Arantxa Casanova](https://github.com/ArantxaCasanova)
- [Belén Luque](https://github.com/bluque)
- [Anna Martí](https://github.com/amartia)
- [Santi Puch](https://github.com/santipuch590)

## Project description

The goal of the project is to build an image classifier using widely used machine learning
and deep learning techniques.

The first half of the course will use a "classic" machine learning approach, primarily focused
on the Bag of Words framework, while the second half of the course will tackle the problem using 
one of the most famous techniques in deep learning: Convolutional Neural Networks (CNN).

## Dataset

The dataset consists of 1881 training images and 807 test images split into 8 classes:
1. coast
2.  forest
3.  highway
4.  inside city
5.  mountain
6. open country 
7. street
8. tall building

The whole dataset can be downloaded [here](https://drive.google.com/open?id=0ByrI9_WaU23FckdmRGprZjhub2c). Once extracted, the resulting 
`MIT_split` folder should be located inside the `dataset` folder.

## Structure of the project

| Folder | Purpose |
| :---: | :---: |
| mlcv | Store all the functions and classes used to run the experiments |
| scripts| Store all the scripts that import certain functions and classes from mlcv and execute the desired experiments |
| dataset | Store the .dat files containing the paths to the images and the corresponding labels, as well as the images themselves |
| models | Store all the pickle files with the objects obtained during training (classifiers, standard scalers, PCAs, etc.) |

## Dependencies 

Most of the dependencies of the project are specified in the `requirements.txt` file 
in the root of the project, which means that you can easily install the required dependencies with the 
following command:

```
$ pip install -r requirements.txt
```

*NOTE: It is recommended to use a separate virtual environment for this project, so that you don't have conflicts
with the versions of the dependencies with your global installation. [Here](https://virtualenv.pypa.io/en/stable/) you will find instructions about the 
usage of `virtualenv`, and [here](http://conda.pydata.org/docs/using/envs.html) for `conda` virtual environments if you're 
using the Anaconda distribution.*

This project also relies on **OpenCV 2.4.11**. Here you will find OpenCV setup instructions for all major OSes:
- [Windows](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html)
- [macOS](http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/)
- [Ubuntu](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)

On top of that this project uses the [Yael](http://yael.gforge.inria.fr/) library from INRIA.
The library was used on a Linux machine.
To install it, just [download it](https://gforge.inria.fr/frs/download.php/file/34217/yael_v438.tar.gz), then gunzip 
and untar it inside the `libraries` folder, and after that execute the following command on the uncompressed `yael` folder:
```
$ ./configure.sh --enable-numpy
```