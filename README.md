# FFNN
Feedforward Neural Network in Python

##Example
    #Train a 2-layer network to fit the sine function in the interval [0.0, 4.0]
    
    #Import everything you need for training a neural network and plotting.
    from ffnn import *
    import numpy as np
    from pylab import *
    
    #Prepare the training data.
    func = np.sin
    input = np.array([np.linspace(0.0, 2.0)]).T
    target = func(input)

    #Define the structure of the network and parameters of the training.
    structure = (1, 8, 1)
    learning_rate = 0.1
    max_epoch = 10000

    #Create a network and train it.
    net = NN(structure, input, target)
    net.train(learning_rate, max_epoch)

    #Plot the output of the network together with and the sine function we attempt to fit.
    net_output = net(input)
    plot(input, target, 'b--', input, net_output, 'k-')
    legend(('target', 'net_output'), loc=0)
    grid(True)
    title('Network output')
    show()
