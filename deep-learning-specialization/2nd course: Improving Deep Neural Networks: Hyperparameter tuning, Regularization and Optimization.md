## 2nd course: "Improving Deep Neural Networks: Hyperparameter tuning, Regularization, and Optimization"

Link to the course: https://www.coursera.org/learn/deep-neural-network/home/welcome

### Setting up your Machine Learning Application

* Usually you have three sets to tune parameters and hyper-parameters: train, development 
(also known as validation, cross-validate set) and test set. Train set is used for training model for current configuration,
dev set is used to estimated accuracy for current configuration (compare different settings to choose the best one), 
the test set is used to make a final unbiased estimation of your model.

* It might be ok to have only train and dev set if you don't need a final estimation of the model

* If you have slightly different distribution for your train and test data (for example you train on images from the internet, but your model
actually will work with images from the users) it is better to have the same distribution for dev and test sets. 

* We got used to dividing train/dev/test sets in proportion 60/20/20 or similar. With the big data, it is usually a different ratio.
You don't need much data to compare your models (dev set) or to make a final estimation (test), but you need a lot of data for training. 
That's why the proportion for big datasets is close to 98/1/1 or even more extreme. 

* If you work on one task and build one neural net, it would be hard or impossible to move your knowledge for a new task. 
Creating a neural net is a highly iterative process, and you need several iterations to understand what your parameters are. 

* Usually during your experiments you try different configurations, estimate your bias and variance, 
and realize where to move further with your experiments. 

### Bias/Variance ratio
* The **bias** is an error from incorrect assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
* The **variance** is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).

* How to determine if you have problems with bias/variance: 
  1. train error is low, dev error is high 
  -> overfitting -> high variance 

  2. train error is high, dev error is high (approximately the same)
  -> underfitting -> high bias

  3. train error is high, dev error is even higher
  -> underfitting -> high bias and high variance

  4. train error is low, dev error is low
  -> good model -> low bias and low variance

* Notion of Bayes error (optimal error) - minimal possible error for the current problem, any classifier can't perform better. 

* Depending on estimated bias and variance you can do different things to improve your algorithm

* Basic recipe:

  1. Firstly, get rid of bias problem. High bias -> bigger network, train longer, other NN architecture.

  2. Then, reduce high variance -> get more data, regularization techniques, other NN architecture.

* Before deep learning era, there are no tools which reduce both bias and variance, that's why it people often talked about "bias-variance trade-off."

* Today with deep learning you can reduce both bias and variance (complicated network, a lot of data, regularization). That's one more reason why deep learning is so popular; you have a powerful tool.

### Regularization with modifying the cost function 

* You add a regularized term to the cost function, to constraint the norms of W. You don't need to have constraints on b because W is a multidimensional vector and its norm grows bigger than a norm of scalar b. 

* If L1 regularization, then W will be sparse (a lot of zeros), it can help to compress the model, but it is used not very often. 

* The most popular regularization - L2. Parameter lambda - regularization parameter. You need to tune it find it with dev set. 

* When you add regularization term to the cost function, gradient descent update rule stays almost the same: 
  Old rule: w = w - alpha * derivative 
  
  New rule: w = w * (1 - alpha * lambda/m) - alpha * derivative 

That means we always multiply W on the coefficient which is smaller than 1. 

* Why regularization prevents overfitting? If lambda is large, then the norm of W is small => Z is close to linear => decision boundary is simpler, not complicated => don't overfit data. 

### Regularization with Dropout 

* Dropout is very powerful technique for regularization, you switch of hidden units with some probability. The most popular implementation is inverted dropout, it allows to keep an expected value of Z the same for a next layer after drop out. Z /= keep_prob

* When you make a prediction on a test set, you don't use dropout. 

* Dropout is similar to L2 regularization, it keeps W weights small (it shrinks them).

* keep_prob is set for every layer. More connections between nodes -> less probability keep_prob to prevent overfitting for this layer. If you have a lot of parameters -> set keep_prob lower. keep_prob = 1 -> the same as you don't have a dropout. 

* keep_prob is another hyper param which needs to be tuned. 

* Dropout is often used in Computer vision field.

* Unless algorithm is overfitting you don't need to use dropout

* When using dropout, you can't rely on decreasing trend of the cost function anymore (for debugging purposes). What you can do is running NN without dropout, be sure the cost function is monotonically decreasing, and then switch on dropout and hope it is still decreasing.

### Other techniques for regularization and preventing overfitting

* Data augmentation: random distortion, syntactic data. It doesn't give a lot of new information, but it is cheap. More data -> it also regularizes the network. 

* Early stopping - you track train and test errors at the same time and stop when the test error is start growing. 

* Usually, in ML you solve two problems: 
  1. Optimize cost function J (Gradient descent, etc.)
  
  2. Not overfit (Regularization, more data, etc.)

  You want to solve these two tasks separately. Firstly minimize J as much as possible. And then try not to overfit (this separation is called "Orthogonalization")

* Early stopping works on these two tasks at the same time, it is not a good thing. It is better to use L2 regularization and train as need as possible and not overfit. But for L2 you need to find lambda. Early stopping gives a similar effect to L2 regularization, but without searching over different lambdas.

### Techniques to speed up your training

* Normalizing your data (the same mu and sigma_sq for both train and test data): cost function looks more symmetric, and it is easier to search minimum.

* Vanishing gradient problem: if all weights are slightly bigger than identity matrix, then for deep NN, activations are exploding. If smaller than identity matrix, then it decreases exponentially (W ^ {L}).  The same holds about derivatives. 

* Number of layers L can be > 150 => gradients is very small or very big => training is slow and challenging. For a long time, it was a huge problem for training deep neural network. To reduce this issue, weight initializing is very important.

* If g(x) = RELU(x), then you initialize weights np.random.randn(..) * np.sqrt(2/n[l-1]), you want variance of w be equals 2/n. It helps to reduce the problem of vanishing a bit.
  
* If g(x) = tanh(x), then you multiply by np.sqrt(1/n[l-1]) or np.sqrt(2/{n[l-1] + n[l]}).

* Variance can also ba a hyperparameter. 
