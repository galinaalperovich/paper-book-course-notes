## 1st course: "Neural Networks and Deep Learning"

Link to the course: https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome

Key points:

* Logistic regression is the simplest neural net with one layer and one neuron with sigmoid activation function

* You can think of a neuron as a function which creates more abstract feature from its input

* Deeper layers represent more complicated features. For example, the first layer in CNN detects edges; next layer detects parts of the face, next layer detects faces, etc. In speech recognition, the first layer can detect low and high frequencies; next layer detects specific sounds, next layer phonemes, etc. 

* The architecture of the network depends on the task. CNN for images, RNNs for sequence data, hybrid for complicated tasks such as self-driving cars. 

* Why Deep learning is taking off? Several reasons: neural nets have a property of improving with bigger and bigger amount of data (unlike other algorithms). 
Today we have a big amount of data, and the neural net can help with solving such tasks. Also, computational power, GPUs allows to relatively quickly experiment with NN. Deep NN showed better results in many fields than classic approaches.

* Often other algorithms than NN can give better results for small- and moderate size data. 

* "AI is the new electricity," means AI is transforming many industries like electric power.

* An activation function can't be linear because if it is, such neural network will be a linear function and all neurons would be useless (composition of linear functions is a linear function). We want to build more complex function.

* It is better to choose RELU (or Leaky RELU) activation function over sigmoid or tanh. The reason is when the values of activation function are extreme (close 0 and 1 in case of sigmoid and close to 1 and -1 in case of tanh), then it's derivative is changing very insignificant, and the training process (gradient descent) is very slow. RELU activation function has a good derivative (1 for x >= 0 and ~0 for x < 0) and the training is much faster.

* Sigmoid is used in the output layer in case of binary classification

* The tanh activation function usually works better than sigmoid for hidden units because the mean of the output is closer to zero. It centers the data for the next layer.

* It is important to initialize weights randomly because if you initialize them with zeros, on every iteration, the neurons will give the same output and learning process will not make sense. 

* If you use sigmoid or tanh activation function, and initialize the weight with the big random numbers, then the training process will be slow. So it is better to initialize with the small random numbers which are close to zero.

* It is important to keep in mind dimensions you are working on when building a neural net or implementing forward and backprop. You can debug it and double check dimensions of W, b, activation output, etc, it will prevent from bugs. 

* Deep representation (many layers) can explain very complex data, means model a complex mapping X -> y. 

* Two main stages in NN is forward propagation (going from the first layer to the last one calculating the output Y) and back propagation (calculating the derivatives for the weights W and b to change current weights) algorithms. You should implement them in order to understand the mechanism of training the neural network.
