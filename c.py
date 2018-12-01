# python notebook for Make Your Own Neural Network
# (c) Tariq Rashid, 2016
# license is GPLv2


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
from neuralNetworkModule import neuralNetwork

# number of input, hidden and output nodes
input_nodes = 6
hidden_nodes = 1000
output_nodes = 3
# learning rate is 0.3
learning_rate = 0.1
# create instance of neural network
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


data_file = open("001.csv", 'r')
data_list = data_file.readlines()
data_file.close()

for n in range(1,1000):
    for line in data_list:
        line=line.replace('\n','')
        words=line.split(',')
        nn.train(words[:6],words[6:])


# test query (doesn't mean anything useful yet)
print (nn.query([1,0,0,1,0,0]))
print (nn.query([1,0,0,0,1,0]))

