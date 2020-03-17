from sklearn.preprocessing import MaxAbsScaler
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
for x in train_images_reshaped:
    scaled_images[counter, :, :] = MaxAbsScaler().fit_transform(x)
    counter += 1

counter = 0
for x in test_images_reshaped:
    scaled_images_test[counter, :, :] = MaxAbsScaler().fit_transform(x)
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
    average_loss = np.zeros(10)
    average_valid_accuracy = np.zeros(10)
    average_training_accuracy = np.zeros(10)
    overall_score = np.zeros(10)
    weights_first_layer = []
    arc_dict = {}
    counter_for_list = 0
    for x in range(0, 10):
        mlp = MLPClassifier(hidden_layer_sizes=m, activation='relu',
                            solver='adam', max_iter=1, shuffle=True)

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

            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + 500]
                mlp.partial_fit(d2_train_dataset[indices], y_train[indices], np.unique(y_train))
                mini_batch_index += 500

                if mini_batch_index >= X_train.shape[0]:
                    break

            if i % 10 == 0:
                valid_accuracy[counter_10_epoch] = mlp.score(d2_validate_dataset, y_validate)
                training_accuracy[counter_10_epoch] = mlp.score(d2_train_dataset, y_train)
                Loss[counter_10_epoch] = mlp.loss_
                counter_10_epoch += 1

        average_loss = (average_loss + Loss)
        average_valid_accuracy = (average_valid_accuracy + valid_accuracy)
        average_training_accuracy = (average_training_accuracy + training_accuracy)
        overall_score[x] = mlp.score(d2_test_dataset, test_labels)
        weights_first_layer.append(mlp.coefs_[0])

    average_loss = average_loss / 10
    average_valid_accuracy = average_valid_accuracy / 10
    average_training_accuracy = average_training_accuracy / 10
    # Get the index and value of the best test accuracy
    best_test_accuracy_index = np.argmax(overall_score)
    best_accuracy = overall_score[best_test_accuracy_index]
    best_weights = weights_first_layer[best_test_accuracy_index]

    arc_dict['name'] = m
    arc_dict['loss_curve'] = average_loss
    arc_dict['train_acc_curve'] = average_training_accuracy
    arc_dict['val_acc_curve'] = average_valid_accuracy
    arc_dict['test_acc'] = best_accuracy
    arc_dict['weights'] = best_weights
    list_of_dict.append(arc_dict)


from utils import part2Plots, visualizeWeights
part2Plots(list_of_dict, save_dir='', filename='', show_plot=True)

a = list_of_dict[1]['weights']
visualizeWeights(a, save_dir='', filename='weigths')

a = list_of_dict[2]['weights']
visualizeWeights(a, save_dir='', filename='weigths')

a = list_of_dict[3]['weights']
visualizeWeights(a, save_dir='', filename='weigths')

a = list_of_dict[4]['weights']
visualizeWeights(a, save_dir='', filename='weigths')

