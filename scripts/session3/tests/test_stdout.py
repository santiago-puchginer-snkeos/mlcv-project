from __future__ import print_function
import sys
import time


def log(msg, fd=sys.stdout):
    fd.write('{}\n'.format(msg))
    fd.flush()

if __name__ == '__main__':
    log('Hello world')
    log('Going to sleep for 1 min')
    time.sleep(60)
    log('Hi again, I am awake now')

    log('Error example', fd=sys.stderr)
