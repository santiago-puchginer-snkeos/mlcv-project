#!/bin/bash

# This script assumes that the following packages are already installed in the system
#   - wget
#   - tar
#   - make
#   - virtualenv
#
# It also assumes that OpenCV 2.4.8 for Python is installed in the root of the filesystem, so that
# the file in /opencv-2.4.8/release/lib/cv2.so actually exists

# Local installation of Python 2.7.12
mkdir ~/python && cd ~/python
wget http://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
tar zxfv Python-2.7.12.tgz
cd Python-2.7.12
./configure --prefix=$HOME/python
make
make install
echo "Installed Python version"
~/python/bin/python -V

# Create virtualenv with Python 2.7.12
mkdir ~/virtualenvs && cd ~/virtualenvs
virtualenv --python=$HOME/python/bin/python mlcv

# Activate virtualenv and install dependencies in requirements.txt
source ~/virtualenvs/mlcv/bin/activate
pip install -r ././../requirements.txt

# Copy cv2 dependency
cp /opencv-2.4.8/release/lib/cv2.so ~/virtualenvs/mlcv/lib/python2.7/site-packages/cv2.so

printf "\nDONE!\n"
