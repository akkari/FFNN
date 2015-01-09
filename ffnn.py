#!/usr/bin/python
# -*- coding:utf-8 -*-
# Created on 2015/01/06

import numpy as np
import cPickle


class NN:
    def __init__(self, structure, input, target):
        """ Input: sequence of sequences.
            target: sequence of sequences."""
        if len(structure) < 3:
            raise ValueError("Invalid structure")
            return

        self.structure = structure
        self.dim_in, self.dim_out = structure[0], structure[-1]
        self.num_layer = len(structure) - 1 # number of hidden layers and output layer
        self.weight, self.bias = [], []
        self.transfer_function = []

    # Initialize weights and biases and set transfer function.
        for i in xrange(1, len(structure)):
            row, col = structure[i], structure[i-1]
            self.weight.append(np.random.random_sample((row, col)))
            self.bias.append(np.random.random((row, 1)))
            if i == len(structure) - 1:
                self.transfer_function.append(purelin)
            else:
                self.transfer_function.append(sigmoid)

    # Normalize input and target.
        self.input, self.target = np.copy(input), np.copy(target)
        self.norm()


    def save_net(self, filename):
        with open(filename, 'w') as f:
            cPickle.dump(self, f)


    def train(self, learning_rate=0.2, max_epoch=10000):
        for epoch in xrange(1, max_epoch+1):
        # Incremental training: weights and biases and updated after each input is presented.
            for i, t in zip(self.input, self.target):
                output, raw_input, net_input = self.propagate(np.array([i]), True)
                error = t - output

            # Calculate sensitivity.
                sensitivity = [None] * len(net_input)
                sensitivity[-1] = -2 * np.dot(np.eye(self.dim_out), error)
                for j in xrange(self.num_layer-2, -1, -1):
                    num_neuron = self.structure[j+1]
                    F = np.zeros((num_neuron, num_neuron))
                    for k in xrange(num_neuron):
                        F[k][k] = D_sigmoid(net_input[j][k])
                    sensitivity[j] = np.dot(np.dot(F, self.weight[j+1].T), sensitivity[j+1])

            # Update weights and biases.
                for j in xrange(self.num_layer):
                    self.weight[j] -= learning_rate * np.dot(sensitivity[j], raw_input[j].T)
                    self.bias[j] -= learning_rate * sensitivity[j]


    def __call__(self, input):
        """ Return a list of numbers or a list of vectors.
        Input: sequence of sequence."""
        i = np.copy(input)
    # Normalize input.
        for index in xrange(len(i)):
            i[index] = i[index] * self.input_norm_coef[0] + self.input_norm_coef[1]

        output = self.propagate(i)

    # Backward transformation of output.
        for index in xrange(len(output)):
                output[index] = output[index] * self.output_norm_coef[0] + self.output_norm_coef[1]

        return output


    def propagate(self, input, save_path=False):
        """ Input: array of arrays."""

        output = []
        raw_input, net_input = [], []

        for i in input:
            into_layer = np.array([i]).T
            for j in xrange(self.num_layer):
                n = np.dot(self.weight[j], into_layer) + self.bias[j]
                if save_path:
                    raw_input.append(into_layer)
                    net_input.append(n)
                outof_layer = self.transfer_function[j](n)
                into_layer = outof_layer

            output.append(outof_layer.flatten())

        if save_path:
            return np.array(output), raw_input, net_input
        else:
            return np.array(output)


    def norm(self):
        self.input_norm_coef = normalize(self.input, direction='forward')
        self.output_norm_coef = normalize(self.target, direction='backward')


def normalize(a, direction, low_end=-1.0, high_end=1.0):
    """ Given a( a p*r array), returns coefficients of transformation for each column and
    normalizes each column at the same time."""

    p_min, p_max = np.min(a, axis=0), np.max(a, axis=0)
    slope = (high_end - low_end) / (p_max - p_min)
    intercept = low_end - p_min * slope

    for i in xrange(len(a)):
        a[i] = slope * a[i] + intercept

    if direction == 'forward':
        return slope, intercept
    else:
        return 1.0 / slope, -intercept / slope


def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))


def D_sigmoid(n):
    return np.exp(-n) / (1.0 + np.exp(-n))**2


def purelin(n):
    return n


def load_net(filename):
    with open(filename, 'r') as f:
        net = cPickle.load(f)
    return net


def save_net(net, filename):
    with open(filename, 'w') as f:
        cPickle.dump(net, f)


def main():
    pass


if __name__=="__main__":main()


