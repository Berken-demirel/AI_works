from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
import copy

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

train_images_reshaped = np.reshape(train_images, (30000, 28, 28))
test_images_reshaped = np.reshape(test_images, (test_images.shape[0], 28, 28))

scaled_images = np.empty(shape = (30000, 28, 28))
scaled_images_test = np.empty(shape=(5000, 28, 28))
# In order to scale pixel values to [-1, 1], The MaxAbsScaler was used.
counter = 0
scaler = MinMaxScaler(feature_range=(-1, 1))

for x in train_images_reshaped:
    scaled_images[counter, :, :] = scaler.fit_transform(x)
    counter += 1

counter = 0
for x in test_images_reshaped:
    scaled_images_test[counter, :, :] = scaler.fit_transform(x)
    counter += 1

# In order to split 10 percent of the training data set to validation
X_train, X_validate, y_train, y_validate = train_test_split(scaled_images,
                                                    train_labels, test_size=0.1,
                                                    random_state=40, stratify=train_labels)

arch_1 = (128,)
arch_2 = (16, 128,)
arch_3 = (16, 128, 16,)
arch_5 = (16, 128, 64, 32, 16,)
arch_7 = (16, 32, 64, 128, 64, 32, 16,)

arch = [arch_1, arch_2, arch_3, arch_5, arch_7]

list_of_dict = []
for m in arch:
    relu_loss = np.zeros(10)
    sigmoid_loss = np.zeros(10)
    relu_grad = np.zeros(10)
    sigmoid_grad = np.zeros(10)
    arc_dict = {}
    counter_for_list = 0

    mlp_relu = MLPClassifier(hidden_layer_sizes=m, activation='relu',
                            solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.01, momentum=0.0)

    mlp_sigmoid = MLPClassifier(hidden_layer_sizes=m, activation='logistic',
                            solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.01, momentum=0.0)

    # Flatten input data
    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples, nx*ny))

    nsamples, nx, ny = X_validate.shape
    d2_validate_dataset = X_validate.reshape((nsamples, nx*ny))

    nsamples, nx, ny = scaled_images_test.shape
    d2_test_dataset = scaled_images_test.reshape((nsamples, nx*ny))

    valid_accuracy = np.empty(shape=(10,))
    training_accuracy = np.empty(shape=(10,))
    Loss = np.empty(shape=(10,))

    # For loop to 100 epochs
    counter_10_epoch = 0
    for i in range(1, 101):
        # Shuffle training set after each epoch
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0

        if i != 1:
            weights_relu_before = copy.deepcopy(mlp_relu.coefs_)
            weights_relu_before = weights_relu_before[0]
            weights_sigmoid_before = copy.deepcopy(mlp_sigmoid.coefs_)
            weights_sigmoid_before = weights_sigmoid_before[0]
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + 500]
            mlp_relu.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mlp_sigmoid.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mini_batch_index += 500

            if mini_batch_index >= X_train.shape[0]:
                break

        # Record the parameters for every 10 steps
        if i % 10 == 0:
            weights_relu_after = mlp_relu.coefs_[0]
            weights_sigmoid_after = mlp_sigmoid.coefs_[0]
            relu_grad[counter_10_epoch] = np.linalg.norm((weights_relu_before - weights_relu_after) * 100)
            sigmoid_grad[counter_10_epoch] = np.linalg.norm((weights_sigmoid_before - weights_sigmoid_after) * 100)
            relu_loss[counter_10_epoch] = mlp_relu.loss_
            sigmoid_loss[counter_10_epoch] = mlp_sigmoid.loss_
            counter_10_epoch += 1


    arc_dict['name'] = m
    arc_dict['relu_loss_curve'] = relu_loss
    arc_dict['sigmoid_loss_curve'] = sigmoid_loss
    arc_dict['relu_grad_curve'] = relu_grad
    arc_dict['sigmoid_grad_curve'] = sigmoid_grad
    list_of_dict.append(arc_dict)

# Visualization Part
from utils import part3Plots
part3Plots(list_of_dict, save_dir='', filename='', show_plot=True)

