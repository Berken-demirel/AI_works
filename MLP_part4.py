from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np


train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

train_images_reshaped = np.reshape(train_images, (30000, 28, 28))
test_images_reshaped = np.reshape(test_images, (test_images.shape[0], 28, 28))

scaled_images = np.empty(shape=(30000, 28, 28))
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
number_of_epoch = 20

arch_fav = (16, 128, 16)

average_loss_1 = np.zeros(number_of_epoch)
average_loss_01 = np.zeros(number_of_epoch)
average_loss_001 = np.zeros(number_of_epoch)

average_valid_scheduled = np.zeros(number_of_epoch)

average_valid_accuracy_1 = np.zeros(number_of_epoch)
average_valid_accuracy_01 = np.zeros(number_of_epoch)
average_valid_accuracy_001 = np.zeros(number_of_epoch)

counter_for_list = 0
arc_dict = {}

for x in range(0, 1):
    mlp_1 = MLPClassifier(hidden_layer_sizes=arch_fav, activation='relu',
                        solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.1, momentum=0.0)

    mlp_01 = MLPClassifier(hidden_layer_sizes=arch_fav, activation='relu',
                        solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.01, momentum=0.0)

    mlp_001 = MLPClassifier(hidden_layer_sizes=arch_fav, activation='relu',
                        solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.001, momentum=0.0)

    mlp_scheduled = MLPClassifier(hidden_layer_sizes=arch_fav, activation='relu',
                    solver='sgd', max_iter=1, shuffle=True, learning_rate_init=0.1, momentum=0.0)

    # Flatten input data
    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples, nx*ny))

    nsamples, nx, ny = X_validate.shape
    d2_validate_dataset = X_validate.reshape((nsamples, nx*ny))

    nsamples, nx, ny = scaled_images_test.shape
    d2_test_dataset = scaled_images_test.reshape((nsamples, nx*ny))

    valid_accuracy_1 = np.empty(shape=(number_of_epoch,))
    valid_accuracy_01 = np.empty(shape=(number_of_epoch,))
    valid_accuracy_001 = np.empty(shape=(number_of_epoch,))

    valid_accuracy_scheduled = np.empty(shape=(number_of_epoch))

    Loss_1 = np.empty(shape=(number_of_epoch,))
    Loss_01 = np.empty(shape=(number_of_epoch,))
    Loss_001 = np.empty(shape=(number_of_epoch,))

    # For loop to 100 epochs
    counter_10_epoch = 0
    for i in range(1, 201):
        # Shuffle training set after each epoch
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0

        if i == 70:
            mlp_scheduled.set_params(learning_rate_init=0.01)

        if i == 120:
            mlp_scheduled.set_params(learning_rate_init=0.001)

        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + 500]
            mlp_1.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mlp_01.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mlp_001.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mlp_scheduled.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
            mini_batch_index += 500

            if mini_batch_index >= X_train.shape[0]:
                break

        if i % 10 == 0:
            valid_accuracy_1[counter_10_epoch] = mlp_1.score(d2_validate_dataset, y_validate)
            valid_accuracy_01[counter_10_epoch] = mlp_01.score(d2_validate_dataset, y_validate)
            valid_accuracy_001[counter_10_epoch] = mlp_001.score(d2_validate_dataset, y_validate)
            valid_accuracy_scheduled[counter_10_epoch] = mlp_scheduled.score(d2_validate_dataset, y_validate)

            Loss_1[counter_10_epoch] = mlp_1.loss_
            Loss_01[counter_10_epoch] = mlp_01.loss_
            Loss_001[counter_10_epoch] = mlp_001.loss_
            counter_10_epoch += 1

    average_loss_1 = (average_loss_1 + Loss_1)
    average_loss_01 = (average_loss_01 + Loss_01)
    average_loss_001 = (average_loss_001 + Loss_001)

    average_valid_accuracy_1 = (average_valid_accuracy_1 + valid_accuracy_1)
    average_valid_accuracy_01 = (average_valid_accuracy_01 + valid_accuracy_01)
    average_valid_accuracy_001 = (average_valid_accuracy_001 + valid_accuracy_001)
    average_valid_scheduled = (average_valid_scheduled + valid_accuracy_scheduled)



arc_dict['name'] = 'arch_3'
arc_dict['loss_curve_1'] = average_loss_1
arc_dict['loss_curve_01'] = average_loss_01
arc_dict['loss_curve_001'] = average_loss_001
arc_dict['val_acc_curve_1'] = average_valid_accuracy_1
arc_dict['val_acc_curve_01'] = average_valid_accuracy_01
arc_dict['val_acc_curve_001'] = average_valid_accuracy_001


from utils import part4Plots
part4Plots(arc_dict, save_dir='', filename='', show_plot=True)

plt.plot(average_valid_scheduled)
plt.ylabel('Validation Accuracy')
plt.show()

print(mlp_scheduled.score(d2_train_dataset, y_train))
