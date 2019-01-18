# -*- coding: utf-8 -*-

import copy, numpy as np
np.random.seed(0)


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output*(1-output)


int2Binary = {}
binary_dim = 8
largest_number = pow(2, binary_dim)


def generate_data_set():
    binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
    for i in range(largest_number):
        int2Binary[i] = binary[i]


alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


def add_with_simple_rnn():
    generate_data_set()

    # initialize weights
    # weights from input to hidden layer
    synapse_0 = 2*np.random.random((input_dim, hidden_dim))-1
    # weights from hidden layer to output layer
    synapse_1 = 2*np.random.random((hidden_dim, output_dim))-1
    # weights from hidden layer to hidden layer
    synapse_h = 2*np.random.random((hidden_dim, hidden_dim))-1

    synapse_0_update = np.zeros_like(synapse_0)
    synapse_1_update = np.zeros_like(synapse_1)
    synapse_h_update = np.zeros_like(synapse_h)

    # train
    for j in range(50000):
        a_int = np.random.randint(largest_number/2)
        a = int2Binary[a_int]
        b_int = np.random.randint(largest_number/2)
        b = int2Binary[b_int]
        # true answer
        c_int = a_int + b_int
        c = int2Binary[c_int]
        # predict answer
        d = np.zeros_like(c)

        overall_error = 0

        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(hidden_dim))

        # forward propagation
        for position in range(binary_dim):
            bit_idx = binary_dim-position-1
            # input
            x = np.array([[a[bit_idx], b[bit_idx]]])  # (1, input_dim)
            # right answer
            y = np.array([[c[bit_idx]]]).T  # (1, 1)

            # hidden layer (input ~+ prev_hidden)
            layer_1 = sigmoid(np.dot(x, synapse_0) + np.dot(layer_1_values[-1], synapse_h))  # (1, hidden_size)
            # output layer
            layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # (1, 1)

            layer_2_error = y - layer_2
            layer_2_deltas.append(layer_2_error*sigmoid_output_to_derivative(layer_2))
            overall_error += np.abs(layer_2_error[0])

            d[bit_idx] = np.round(layer_2[0][0])

            layer_1_values.append(copy.deepcopy(layer_1))

        # back propagation
        future_layer_1_delta = np.zeros(hidden_dim)
        for position in range(binary_dim):
            x = np.array([[a[position], b[position]]])
            layer_1 = layer_1_values[-position-1]
            prev_layer_1 = layer_1_values[-position-2]
            layer_2_delta = layer_2_deltas[-position-1]
            layer_1_delta = \
                (future_layer_1_delta.dot(synapse_h.T)+layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += x.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        synapse_0 += synapse_0_update*alpha
        synapse_1 += synapse_1_update*alpha
        synapse_h += synapse_h_update*alpha
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

        if j%1000 == 0:
            print 'error:'+str(overall_error)
            print 'pred:'+str(d)
            print 'true:'+str(c)
            out = 0
            for index, x in enumerate(reversed(d)):
                out += x*pow(2,index)
            print str(a_int) + '+' + str(b_int)+'='+str(out)+'['+str(c_int)+']'
            print '-----------'


def main():
    add_with_simple_rnn()


if __name__ == '__main__':
    main()