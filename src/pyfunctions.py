import numpy as np
import sys
from scipy import signal
import scipy.io as sio

def conv2(input_data, kernel, mode, tag):
    
    input_channels, input_height, input_width = input_data.shape
    kernel_channels, kernel_height, kernel_width = kernel.shape


    if mode == "valid":
        result_width = input_width - kernel_width + 1
        result_height = input_height - kernel_height + 1

    elif mode == "full":
        result_width = input_width + kernel_width - 1
        result_height = input_height + kernel_height - 1

    else:
        print("ERROR: there is no such mode (%s)" % (mode))
        sys.exit(1)

    result = np.zeros((input_channels, result_height, result_width), dtype=np.float32)
    
    # print("PYTHON: tag", tag)
    # print("PYTHON: size of input", input_data.shape)
    # print("PYTHON: size of kernel", kernel.shape)
    # print("PYTHON: size of result", result.shape)

    num_conv_channels = int(input_channels / float(kernel_channels))

    if kernel_channels == 1:
        kernel_sequence = [0, 0, 0]
    else:
        kernel_sequence = [0, 1, 2]

    for channel, kernel_channel in enumerate(kernel_sequence):
        result[channel, :, :] = signal.convolve2d(input_data[channel,:,:],  \
                                                   kernel[kernel_channel,:,:], \
                                                   mode=mode)
    # plt.imshow(np.moveaxis(result, 0, -1))
    # plt.show()

    #save input and output data 
    save_folder = "data_from_python/"

    if (input_channels == kernel_channels):
        result = np.sum(result, axis=0, dtype=np.float32)

        container = {'input_' + tag: np.moveaxis(input_data, 0, -1),
                     'kernel_' + tag: np.moveaxis(kernel, 0, -1),
                     'results_' + tag: result}

    else:
        container = {'input_' + tag: np.moveaxis(input_data, 0, -1),
                     'kernel_' + tag: np.moveaxis(kernel, 0, -1),
                     'results_' + tag: np.moveaxis(result, 0, -1)}

    sio.savemat(save_folder + tag + '.mat', container)

    return result.flatten('C')


def save_matrix(key, matrix):

    container = {key: np.moveaxis(matrix, 0, -1)}
    save_folder = "../data_from_python/"
    sio.savemat(save_folder + key + '.mat', container)
