Download link :https://programming.engineering/product/cs440-ece448-mp05-neural-networks/

# CS440-ECE448-MP05-Neural-Networks
CS440/ECE448 MP05: Neural Networks
This tutorial and its materials are based on and adapted from the PyTorch Official

Tutorial and mrdbourke/pytorch-deep-learning, revised to meet the requirements of this assignment.

We encourage you to explore the original tutorials for additional topics not covered in this assignment.

This notebook will walk you through the whole MP, giving you instructions and debugging tips as you go.

If you are already familiar with PyTorch, you can jump to the implement this sections.

Goal

The goal of this assignment is to employ neural networks, nonlinear and multi-layer

5 categories



extensions of the linear perceptron, to classify images into :



ship (0)



automobile (1)



dog (2)



frog (3)



horse (4)

Table of Contents

Goal

What is PyTorch

Datasets

Dataloaders

Tensors

Neural Net Layers

You need to implement this (1)

Build a Model

Train a Model

You need to implement this (2)

Gradients

2/19/24, 3:42 PM mp5

You need to implement this (3)

Extra Credit

What is PyTorch?

PyTorch is an open source machine learning framework that accelerates the path from research prototyping to production deployment.

PyTorch allows you to manipulate and process data and write machine learning algorithms using Python code (user-friendly!).

PyTorch also offers some domain-specific libraries such as TorchText, TorchVision, and TorchAudio.

You can install PyTorch with conda or pip.

The exact installation command varies based on your system setup and desired PyTorch version. For example, to install PyTorch with pip, you can use the following command:

pip install torch

For this assignment, GPU support is not required since the autograder doesn’t have

any GPUs. Therefore, you can install PyTorch without CUDA support. However, learning how to use PyTorch with CUDA might be beneficial for your future projects.

Let’s verify the installation by printing PyTorch version. The code block below should run without error if PyTorch was installed correctly.

In [1]: import torch # the library name is torch print(“PyTorch version:”, torch.__version__)

PyTorch version: 1.13.1

2/19/24, 3:42 PM mp5

PyTorch Workflow

Source: mrdbourke/pytorch-deep-learning

Machine learning is a game of two parts:

Transform your data, whether it’s images, text, or any other form, into a numerical format (a representation).

. Pick or build a model to learn the representation as best as possible.

In this MP, you will mostly work on the second step: building a model.

2/19/24, 3:42 PM mp5

print(“Shape of train set:”, train_set.shape)

print(“Shape of test set:”, test_set.shape)

Shape of train set: (2813, 2883)

Shape of test set: (937, 2883)

In [3]: # Use the helper function to visualize training set.

The third argument is the index of image to visualize.

Feel free to change the index. Note that the size of training set is 2813 helper.show_train_image(train_set, train_labels, 2812)

2/19/24, 3:42 PM mp5

Dataloaders

Source: PyTorch Tutorial

Code for processing data samples can get messy and hard to maintain; we ideally want our data processing code to be decoupled from our model training code for better

readability and modularity. PyTorch provides two data primitives:

torch.utils.data.DataLoader torch.utils.data.Dataset

and that allow

you to use pre-loaded datasets as well as your own data. Dataset stores the samples

and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

A dataloader is a way for you to handle loading and transforming data before it enters your network for training or prediction. It lets you write code that looks like you’re just looping through the dataset, with the division into batches happening automatically. In this MP, you don’t need to write the dataset and dataloader part, we have provided one for you, but you need to understand how to use it.

For more information about datasets and dataloaders, please refer to Source.

2/19/24, 3:42 PM mp5

train_set_processed, test_set_processed = helper.Preprocess(train_set, test_

# Generate dataloaders

train_loader, test_loader = helper.Get_DataLoaders( train_set_processed,

train_labels,

test_set_processed,

test_labels,

batch_size=100,

)

2/19/24, 3:42 PM mp5

3,

4, 3, 2, 4, 3, 3, 2, 0, 2, 0, 3, 4, 0, 4, 3, 1, 2, 2, 3, 4, 1, 4, 1,

2,

4, 1, 3, 1, 4, 0, 4, 2, 4, 3, 4, 0, 1, 3, 1, 4, 2, 3, 1, 2, 2, 1, 0,

2,

3, 2, 2, 4, 3, 1, 0, 2, 2, 2, 4, 3, 3, 1, 1, 0, 2, 2, 4, 3, 4, 1, 0,

3,

3, 0, 0, 3])

Each batch contains features and labels. The features are tensors with shape

(batch_size, feature_size)

and the labels are tensors with shape

(batch_size)

.

Tensors

Source: mrdbourke/pytorch-deep-learning

Tensors are a specialized data structure that are very similar to numpy arrays and

matrices. Their job is to represent data in a numerical way. Tensors are similar to

NumPy’s arrays, except that tensors can run on GPUs or other hardware accelerators

(better performance!). In PyTorch, we use tensors to encode the inputs and outputs of a

model, as well as the model’s parameters.

The code cell below may give you some idea of how to use tensors.

2/19/24, 3:42 PM mp5

Random Tensor:

tensor([[0.7706, 0.2286, 0.2136],

[0.6712, 0.4824, 0.2182]])

Zeros Tensor:

tensor([0., 0., 0., 0., 0.])

My Tensor:

tensor([[7, 7, 5],

[1, 3, 0],

[2, 2, 1],

[9, 4, 8]])

Shape of tensor: torch.Size([4, 3])

Datatype of tensor: torch.int64

Device tensor is stored on: cpu

Element-wise multiplication:

tensor([1, 2, 3]) * tensor([2, 3, 4]) = tensor([ 2, 6, 12])

One of the most common errors you’ll run into in deep learning is shape mismatches, because matrix multiplication has a strict rule about what shapes and sizes can be combined.

The code cell below is such an example.

In [8]: # matrix multiplication

tensor_A = torch.tensor([[1, 2],

[3, 4],

[5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],

[8, 11],

[9, 12]], dtype=torch.float32)

print(“tensor_A, shape =”, tensor_A.shape)

print(tensor_A, “\n”)

print(“tensor_B, shape =”, tensor_B.shape)

print(tensor_B)

# torch.matmul() is a built-in matrix multiplication function

torch.matmul(tensor_A, tensor_B) # this will error because of shape misma

tensor_A, shape = torch.Size([3, 2])

tensor([[1., 2.],

[3., 4.],

[5., 6.]])

tensor_B, shape = torch.Size([3, 2])

tensor([[ 7., 10.],

8., 11.],

9., 12.]])

2/19/24, 3:42 PM mp5

—————————————————————————

RuntimeError Traceback (most recent call last)

Cell In[8], line 15

12 print(tensor_B)

14 # torch.matmul() is a built-in matrix multiplication function

—> 15 torch.matmul(tensor_A, tensor_B) # this will error because of sha

pe mismatch

RuntimeError: mat1 and mat2 shapes cannot be multiplied (3×2 and 3×2)

In [9]: # tensor_A and tensor_B cannot be multiplied

However, multiplying tensor_A with the transpose of tensor_B is legal (3×2

transpose of a tensor: tensor.T – where tensor is the desired tensor to print(“tensor_A, shape =”, tensor_A.shape)

print(tensor_A, “\n”)

print(“transpose of tensor_B, shape =”, tensor_B.T.shape)

print(tensor_B.T, “\n”) # transpose of tensor_B

print(“tensor_A * tensor_B.T equals”)

print(torch.matmul(tensor_A, tensor_B.T))

tensor_A, shape = torch.Size([3, 2])

tensor([[1., 2.],

[3., 4.],

[5., 6.]])

transpose of tensor_B, shape = torch.Size([2, 3])

tensor([[ 7., 8., 9.],

[10., 11., 12.]])

tensor_A * tensor_B.T equals

tensor([[ 27., 30., 33.],

61., 68., 75.],

95., 106., 117.]])

Neural Net Layers

So far we have looked into the tensors, their properties and basic operations on tensors. These are especially useful to get familiar with if we are building the layers of our network from scratch. PyTorch also provides some built-in blocks in the torch.nn module.

We can use nn.Linear(in_features, out_features) to create a a linear layer that applies a linear transformation to the incoming data x:

y = x AT + b , where A and b are initialized randomly.



This will take an input of size (∗, in_features) where ∗ means any number of dimensions including none and in_features is the size of each input sample. It will yield an output of size (∗, out_features) where all but the last dimension are the same shape as the input and out_features is the size of each output sample.

2/19/24, 3:42 PM mp5

Input:

tensor([[-2.1427, -0.7382, 0.3335, 0.3865, -0.6146], [-0.8864, 0.7096, 0.9242, -0.4327, -0.1272],

0.9784, 0.2195, 0.1283, -2.1508, 0.8675], [-0.2716, 0.4867, -0.6735, 2.4885, -0.1850],

[-0.2660, -0.7077, 2.2901, -0.3677, 0.5883],

0.3426, -1.5650, 0.0481, 0.4113, -0.2566], [-0.4732, -0.3190, -0.8887, 0.5200, -0.9329], [-2.9853, 0.1357, 1.3753, -0.4170, 0.5080]])

Input size: torch.Size([8, 5])

Block output:

tensor([[0.3037,

0.7349,

0.1701,

0.5356,

0.8148, 0.1889,

0.6813],

[0.3476,

0.5693,

0.2838,

0.4702,

0.5560,

0.4595,

0.4932],

[0.3449,

0.2760,

0.4551,

0.6195,

0.3657,

0.7563,

0.3839],

[0.5403,

0.7303,

0.6199,

0.3546,

0.7415,

0.1658,

0.5843],

[0.1345,

0.4733,

0.1670,

0.5443,

0.5537,

0.5599,

0.5445],

[0.3013,

0.5081,

0.3476,

0.5320,

0.6801,

0.3028,

0.6909],

[0.5304,

0.6252,

0.4428,

0.4566,

0.7025,

0.2153,

0.6614],

[0.1900,

0.7365,

0.1062,

0.6096,

0.7908,

0.3465,

0.5088]],

grad_fn=<SigmoidBackward0>)

Block output size: torch.Size([8, 7])

Note the output values produced by the two methods are not the same, because the

weights and biases were initialized randomly. But the shape of the output tensors are the

same.

You need to implement this (1)

create_sequential_layers() submitted.py

Implement the function in the file.

nn.Sequential



This function should return a object.

Once you have implemented the function, you can run the code cell below to test your implementation.

In [13]: !python -m unittest tests.test_visible.Test.test_sequential_layers

.

———————————————————————-

Ran 1 test in 0.040s

OK

Build a Model

Source: mrdbourke/pytorch-deep-learning and PyTorch Tutorial

Now that we have covered some fundamentals, let’s focus on how to build a model. In this MP, you will implement a neural network.

2/19/24, 3:42 PM mp5

2/19/24, 3:42 PM mp5

“””

super().__init__() # call the initialization function of the base class

network architecture, please try to relate the code to the picture self.hidden = torch.nn.Linear(4, 3) # input has 4 values self.relu = torch.nn.ReLU() # activation function

self.output = torch.nn.Linear(3, 2) # output has 2 values

def forward(self, x):

“””

In the forward function we accept a Tensor of input data (the variable x We can use Modules defined in the __init__() as well as arbitrary (diffe

“””

x_temp = self.hidden(x)

# input data x flows through

the hid

x_temp = self.relu(x_temp)

# use relu as the

activation

functio

y_pred = self.output(x_temp)

# predicted value

return y_pred

Create an instance of the SimpleNet model (this is a subclass of nn.Module model = SimpleNet()

Create inputs, here we use a random tensor, but in reality, the input shou x = torch.rand(3, 4) # 3 samples, each sample of size 4

Forward pass: compute predicted y by passing x to the model

Note that the model is randomly initialized, so this prediction probably d

We need to train our model and teach it to make reasonable predictions (we y_pred = model(x)

print(“y_pred.shape: “, y_pred.shape)

# since our output layer has 2 value

print(y_pred)

y_pred.shape: torch.Size([3, 2])

tensor([[-0.4393,

0.4056],

[-0.4375,

0.4294],

[-0.4397,

0.3995]], grad_fn=<AddmmBackward0>)

Name

What does it do?

torch.nn

Contains all of the building blocks (network layers) for computational

graphs (essentially a series of computations executed in a particular

way).

The base class for all neural network modules, all the building blocks

torch.nn.Module

for neural networks are subclasses. If you’re buildi

ng a neural

network in

PyTorch, your

models should subclass

nn.Module

.

Requires a

forward()

method be implemented.

is your network’s initialization function, where you will

__init__()

__init__()

initialize the neural network layers.

All

subclasses (e.g., your own network) require a

forward()

nn.Module

forward()

method, this defines the computation that will take

place on the

data passed to the particular nn.Module . Simply put,

forward()

should perform a forward pa

ss through your

network.

Note that you should NOT directly call the

forward(x)

method,

though. To use the model, you should call the whole model itself and

model(x)

pass it the input data, as in to perform a forward pass

2/19/24, 3:42 PM mp5

Once you have implemented the function, you can run the code cell below to test your implementation.

In [15]: !python -m unittest tests.test_visible.Test.test_loss_fn

.

———————————————————————-

Ran 1 test in 0.039s

OK

Gradients

When training neural networks, the most frequently used algorithm is back propagation. In this algorithm, parameters (model weights, biases, …) are adjusted according to the gradient of the loss function with respect to the given parameter. To compute those

torch.autograd

gradients, PyTorch has a built-in differentiation engine called . It supports automatic computation of gradient for any computational graph. Take a look at this for more information about autogradient in PyTorch (you should be able to understand everything that is covered in the linked tutorial by now).

Time for an example:

2/19/24, 3:42 PM mp5

print(model.hidden.weight, “\n”)

# Preform back propagation

optimizer.zero_grad() # Clear previous gradients, will see more about this

loss.backward() # back propagation

Here we only print the gradients of hidden.weight,

but backward() updates gradients of all related parameters print(“gradients of weights of the hidden layer:”) print(model.hidden.weight.grad, “\n”)

Update parameters

optimizer.step()

print(“Weights of hidden linear layer, after back propagation:”)

print(model.hidden.weight) # You can verify the results, after = before – gr

Predicted values:

tensor([[-0.4393,

0.4056],

[-0.4375,

0.4294],

[-0.4397,

0.3995]], grad_fn=<AddmmBackward0>)

True values:

tensor([[1., 1.],

[1., 1.],

[1., 1.]])

MSE: tensor(1.2084, grad_fn=<MseLossBackward0>)

Weights of hidden

linear layer, before back propagation:

Parameter

containing:

tensor([[-0.4284,

0.2926, -0.4868,

-0.1456],

[

0.4332,

0.2243, 0.4980,

0.2780],

[

0.1151,

-0.1169, -0.1383,

-0.2282]], requires_grad=True)

gradients of weights of the hidden layer:

tensor([[ 0.0000, 0.0000, 0.0000, 0.0000],

[-0.0840, -0.0754, -0.0986, -0.0919],

[ 0.0000, 0.0000, 0.0000, 0.0000]])

Weights of hidden

linear layer, after back propagation:

Parameter

containing:

tensor([[-0.4284,

0.2926, -0.4868,

-0.1456],

[

0.5172,

0.2997, 0.5966,

0.3699],

[

0.1151,

-0.1169, -0.1383,

-0.2282]], requires_grad=True)

Hyperparameters

Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence

rates.

We define the following hyperparameters for training:

2/19/24, 3:42 PM mp5



Number of Epochs – the number of times to iterate over the dataset (one epoch = one forward pass and one backward pass of all the training samples)



Batch Size – the number of data samples propagated through the network before the parameters are updated



Learning Rate – how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

Batch size

Batch size might be confusing if this is your first time hearing it. You can find a wonderful explanation here.



Let’s say you have 1050 training samples and you want to set up a

batch_size equal to 100. The algorithm takes the first 100 samples (from

1st to 100th) from the training dataset and trains the network. Next, it

takes the second 100 samples (from 101st to 200th) and trains the

network again. We can keep doing this procedure until we have

propagated all samples through of the network. Problem might happen

with the last set of samples. In our example, we’ve used 1050 which is not

divisible by 100 without remainder. The simplest solution is just to get the

final 50 samples and train the network.

Advantages of using a batch size < number of all samples:



It requires less memory. Since you train the network using fewer

samples, the overall training procedure requires less memory. That’s especially important if you are not able to fit the whole dataset in your machine’s memory.



Typically networks train faster with mini-batches. That’s because we

update the weights after each propagation. In our example we’ve

propagated 11 batches (10 of them had 100 samples and 1 had 50

samples) and after each of them we’ve updated our network’s

parameters. If we used all samples during propagation we would make

only 1 update for the network’s parameter.

Disadvantages of using a batch size < number of all samples:



The smaller the batch the less accurate the estimate of the gradient will be.

Annotated Code

2/19/24, 3:42 PM mp5

Please note that the code snippets below do not make use of batch size, instead it uses one single “batch” with all training data. Take a look here for an example with batch size. In this MP, you will write a training loop which operates data in batches i.e., using

dataloaders.

You need to implement this (3)

Now it’s time for you to create a neural network. You will implement a neural network in

Class NeuralNet train()

and write a training loop in the function .

In each training iteration, the input of your network is a batch of preprocessed image data of size (batch size, 2883). The number of neurons in the output layer should be equal to the number of categories.

For this assignment, you should use Cross Entropy Loss. Notice that because

2/19/24, 3:42 PM mp5

PyTorch’s CrossEntropyLoss incorporates a softmax function, you do not need to explicitly include an normalization function in the last layer of your network.

0.61 on the

To get a full score, the accuracy of your network must be above

visible testing set above 0.57 on the hidden testing set

, and . The structure of the neural network is completely up to you. You should be able to get around 0.62 testing-set accuracy with a two-layer network with no more than 200

hidden neurons.

If you are confident about a model you have implemented but are not able to pass the accuracy thresholds on gradescope, try adjusting the learning rate. Be aware, however, that using a very high learning rate may worse performance since the model may begin to oscillate around the optimal parameter settings.

Once you have implemented the function, you can run the code cell below to test your

implementation.

In [1]: !python grade.py

Total number of network parameters: 138677

Accuracy: 0.6264674493062967

Confusion Matrix =

[[146. 24. 8. 5. 7.]

27. 123. 11. 22. 19.]

13. 13. 87. 45. 28.]

9. 4. 34. 122. 17.]

14. 10. 23. 17. 109.]]

+5 points for accuracy above 0.15

+5 points for accuracy above 0.25

+5 points for accuracy above 0.48

+5 points for accuracy above 0.55

+5 points for accuracy above 0.57

+5 points for accuracy above 0.61

…

———————————————————————-

Ran 3 tests in 2.395s

OK

Extra credit

You can earn extra credits worth 10% of this MP if the accuracy of your network is above

0.66



Some tips:

. Choose a good activation function.

2/19/24, 3:42 PM mp5

. L2 Regularization.

. Convolutional neural network.

In [ ]:

https://courses.grainger.illinois.edu/ece448/s /mp/mp05_notebook.html 22/22
