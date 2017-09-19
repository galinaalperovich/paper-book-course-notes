## 2nd course: "Improving Deep Neural Networks: Hyperparameter tuning, Regularization, and Optimization"

Link to the course: https://www.coursera.org/learn/deep-neural-network/home/welcome

### Key points:

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
