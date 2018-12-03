# python notebook for Make Your Own Neural Network
# working with the MNIST data set
#
# (c) Tariq Rashid, 2016
# license is GPLv2

import numpy
import matplotlib.pyplot

data_file = open("mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()


print (len(data_list))
print (data_list[1])


# take the data from a record, rearrange it into a 28*28 array and plot it as an image
all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.savefig('001.png')



