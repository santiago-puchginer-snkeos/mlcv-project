import mlcv.input_output as io
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arguments_parser = argparse.ArgumentParser()
    arguments_parser.add_argument('file')
    arguments_parser.add_argument('--dpi', type=int, default=50)
    arguments = arguments_parser.parse_args()
    filename = arguments.file
    dpi = arguments.dpi

    print(filename.upper())

    # Load file
    history = io.load_object(filename, ignore=True)

    # Plot
    plt.figure(dpi=dpi, facecolor='white')
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0, 1))
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    plt.close()

    plt.figure(dpi=dpi, facecolor='white')
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Categorical cross-entropy (loss)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.close()