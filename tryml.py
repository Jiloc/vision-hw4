from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print inputs
    l = [   make_layer(inputs, 32, LRELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)


def neural_net(inputs, outputs):
    print inputs
    l = [   make_layer(inputs, 64, LRELU),
            make_layer(64, 32, LRELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)


print("loading data...")
train = load_classification_data("mnist.train", "mnist.labels", 1)
test  = load_classification_data("mnist.test", "mnist.labels", 1)
print("done")
print

print("training model...")
batch = 128
iters = 3000
rate = .1
momentum = .9
decay = .01

models = []
decays = (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
rates = (1, 0.1, 0.01, 0.001, 0.0001)
for decay in decays:
        m = neural_net(train.X.cols, train.y.cols)
        models.append(m)
        train_model(m, train, batch, iters, rate, momentum, decay)
        print("done")
        print
for i, decay in enumerate(decays):
        print("evaluating model... for lr {}, decay {}".format(rate, decay))
        print("training accuracy: {}".format(accuracy_model(models[i], train)))
        print("test accuracy:     {}".format(accuracy_model(models[i], test)))


## Questions ##

# 2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
# the training accuracy tells how good the model performs with the data that it already processed, 
# while the testing accuracy shows how good the model performs with the data that it didn't see yet. 

# 2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
# Best solution:
# batch = 128
# iters = 1000
# rate = .1
# momentum = .9
# decay = .0


# 2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
# Best solution:
# batch = 128
# iters = 1000
# rate = .1
# momentum = .9
# decay = .01


# 2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
# LRELU

# 2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
# evaluating model... for lr 1, decay 0.01
# training accuracy: 0.0987166666667
# test accuracy:     0.098
# evaluating model... for lr 0.1, decay 0.01
# training accuracy: 0.9555
# test accuracy:     0.9489
# evaluating model... for lr 0.01, decay 0.01
# training accuracy: 0.9183
# test accuracy:     0.9225
# evaluating model... for lr 0.001, decay 0.01
# training accuracy: 0.864316666667
# test accuracy:     0.8717
# evaluating model... for lr 0.0001, decay 0.01
# training accuracy: 0.4483
# test accuracy:     0.4504

# 2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
# evaluating model... for lr 0.1, decay 1
# training accuracy: 0.917666666667
# test accuracy:     0.9215
# evaluating model... for lr 0.1, decay 0.1
# training accuracy: 0.952883333333
# test accuracy:     0.948
# evaluating model... for lr 0.1, decay 0.01
# training accuracy: 0.958133333333
# test accuracy:     0.9564
# evaluating model... for lr 0.1, decay 0.001
# training accuracy: 0.9635
# test accuracy:     0.9589
# evaluating model... for lr 0.1, decay 0.0001
# training accuracy: 0.955683333333
# test accuracy:     0.9503
# evaluating model... for lr 0.1, decay 1e-05
# training accuracy: 0.9617
# test accuracy:     0.9586
# it causes just a slight chagne because the network is made of 2 layers only.

# 2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
# evaluating model... for lr 0.1, decay 1
# training accuracy: 0.927316666667
# test accuracy:     0.9285
# evaluating model... for lr 0.1, decay 0.1
# training accuracy: 0.976966666667
# test accuracy:     0.9683
# evaluating model... for lr 0.1, decay 0.01
# training accuracy: 0.976866666667
# test accuracy:     0.9639
# evaluating model... for lr 0.1, decay 0.001
# training accuracy: 0.983683333333
# test accuracy:     0.9708
# evaluating model... for lr 0.1, decay 0.0001
# training accuracy: 0.984366666667
# test accuracy:     0.9702
# evaluating model... for lr 0.1, decay 1e-05
# training accuracy: 0.981983333333
# test accuracy:     0.9714

# 3.2.1 How well does your network perform on the CIFAR dataset?


