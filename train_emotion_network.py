import glob
import json
import os
import cv2
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle, resample
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from hyperopt import hp, fmin, tpe, Trials

pathfile = os.getcwd() + "\\path.txt"
f = open(pathfile, "r")
lines = f.readlines()
f.close()

model_path = lines[0]

with open(model_path + '\\global_settings.json') as inputfile:
    settings = json.load(inputfile)
classes = settings['emotion_classes']
class_to_remove = settings['class_to_remove']
test_classes = classes.copy()
classes.remove(class_to_remove)
num_classes = len(classes)
positive_class = settings['positive_class']
positive_index = classes.index(positive_class)
removed_index = test_classes.index(class_to_remove)
use_all_classes = settings['use_all_classes']
if not use_all_classes:
    test_classes.remove(class_to_remove)
num_test_classes = len(test_classes)

if not os.path.exists(model_path + "\\CNN_CV"):
    os.makedirs(model_path + "\\CNN_CV")

paths = []
for c in test_classes:
    paths.append(model_path + "\\Faces\\" + c)


def read_data():
    extension = 'jpg'

    images = []
    labels = []
    images_test = []
    labels_test = []
    for i, path in enumerate(paths):
        os.chdir(path)
        files = glob.glob('*.{}'.format(extension))
        for file in files:
            if int(file[-5]) != 2:  # only read in training and test images, not images from final test regions
                im = cv2.imread(file)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                print(file)
                if int(file[-5]) == 0:
                    images.append(im)
                    labels.append(test_classes[i])
                if int(file[-5]) == 1:
                    images_test.append(im)
                    labels_test.append(test_classes[i])

    return images, labels, images_test, labels_test


def format_image(image):
    global i
    i += 1
    print(i)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.reshape(image, (image.shape[0], image.shape[1], 3))
    image = tf.image.resize(image, (image_res[model_name], image_res[model_name]))  # /255.0
    image = preprocess[model_name](image)
    return image


def format_label(lab):
    lab = test_classes.index(lab)
    return lab


def resample_images(images, labels):
    images_ = {}
    labels_ = {}
    sizes = []
    for i, c in enumerate(test_classes):
        indices = np.where(labels == i)[0]
        sizes.append(len(indices))
        images_[c] = images[indices]
        labels_[c] = labels[indices]

    min_ = np.min(sizes)

    for i, key in enumerate(images_):
        images_[key], labels_[key] = resample(images_[key], labels_[key], n_samples=min_, replace=False)

        if i == 0:
            images = images_[key]
            labels = labels_[key]
        else:
            images = np.concatenate((images, images_[key]))
            labels = np.append(labels, labels_[key])

    return images, labels


def shuffle_select_process(images, labels, resample):
    (images, labels) = shuffle(images, labels)
    images = images[:n_samples]
    labels = labels[:n_samples]

    images = np.stack(list(map(format_image, images)))
    labels = np.array(list(map(format_label, labels)))

    if resample:
        images, labels = resample_images(images, labels)

    return images, labels


def remove_withheld_classes(im, lab):
    labels = np.array([test_classes[i] for i in lab])
    images_temp = []
    labels_temp = []
    for i, label in enumerate(labels):
        if label in classes:
            images_temp.append(im[i])
            labels_temp.append(classes.index(label))
    im = np.stack(images_temp)
    lab = np.array(labels_temp)

    return im, lab


def train_model(images_train, labels_train, images_test, labels_test, model_name, IMAGE_RES, index=str(0), n_epochs=100,
                final=False, alpha0=1e-3, decay=0.95, layers_back=0, threshold_check=False):
    if final:
        index = "Final"

    feature_extractors = {
        'emopy': tf.keras.models.load_model(emopypath),
        'vgg16': tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(IMAGE_RES, IMAGE_RES, 3), weights='imagenet'),
        'vgg19': tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=(IMAGE_RES, IMAGE_RES, 3), weights='imagenet'),
        'vggface2': tf.keras.models.load_model(vggface2path),
        'inception_resnet': tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(299, 299, 3), weights='imagenet'),
        'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=(IMAGE_RES, IMAGE_RES, 3), weights='imagenet')}

    feature_extractor = feature_extractors[model_name]

    for layer in feature_extractor.layers[:-layers_back]:
        layer.trainable = False

    if num_test_classes > num_classes:
        if final:
            images_final_test = images_test
            labels_final_test = labels_test
        images_train, labels_train = remove_withheld_classes(images_train, labels_train)
        images_test, labels_test = remove_withheld_classes(images_test, labels_test)

    featurewise_center = True
    featurewise_std_normalization = True
    rotation_angle = 45
    shift_range = 0.3
    shear_range = 0.2
    zoom_range = 0.2
    channel_shift_range = 0.3
    if model_name == 'emopy':
        featurewise_center = True
        featurewise_std_normalization = True
        rotation_angle = 10
        shift_range = 0.1
        zoom_range = 0.1
        channel_shift_range = 0.0
        shear_range = 0.0

    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=featurewise_center, samplewise_center=False,
                                                                featurewise_std_normalization=featurewise_std_normalization,
                                                                samplewise_std_normalization=False, zca_whitening=False,
                                                                zca_epsilon=1e-06, rotation_range=rotation_angle,
                                                                width_shift_range=shift_range, height_shift_range=shift_range,
                                                                brightness_range=None, shear_range=shear_range,
                                                                zoom_range=zoom_range, channel_shift_range=channel_shift_range,
                                                                fill_mode='nearest', cval=0.0,
                                                                horizontal_flip=True, vertical_flip=False, rescale=None,
                                                                preprocessing_function=None,
                                                                data_format='channels_last', validation_split=0.0,
                                                                dtype='float32')

    test_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=featurewise_center,
                                                                     samplewise_center=False,
                                                                     featurewise_std_normalization=featurewise_std_normalization,
                                                                     samplewise_std_normalization=False)

    augmentor.fit(images_train)
    test_augmentor.fit(images_train)
    mean = test_augmentor.mean
    std = test_augmentor.std

    if model_name == 'vggface2' or model_name == 'emopy':
        if model_name == 'vggface2':
            last_layer = feature_extractor.get_layer('avg_pool').output
            x = tf.keras.layers.Flatten()(last_layer)
        elif model_name == 'emopy':
            x = feature_extractor.get_layer('dropout_14').output
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(feature_extractor.input, out)
    else:
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha0, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: alpha0 * decay**epoch)
    #     lambda epoch: 1e-3 * 10 ** (epoch / 30))
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(augmentor.flow(images_train, labels_train, batch_size=BATCH_SIZE), epochs=n_epochs,
                                  verbose=1, shuffle=True,
                                  # validation_data=(images_test, labels_test),
                                  validation_data=test_augmentor.flow(images_test, labels_test, batch_size=BATCH_SIZE),
                                  steps_per_epoch=int(np.ceil(np.size(labels_train) / float(BATCH_SIZE))),
                                  use_multiprocessing=False, callbacks=[early_stopping, lr_schedule])

    if threshold_check and not final:
        return model, None, mean, std

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(np.size(acc))

    figure_size = (14, 8)
    fig = plt.figure(figsize=figure_size)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy: {}'.format(index))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss: {}'.format(index))

    if not final:
        plt.savefig('./CNN_CV/emotion_epochs_a={}_d={}_m={}_l={}_fold_{}_{}.png'.format(alpha0, decay, model_name, layers_back,
                                                                                        index, time.time()), dpi=fig.dpi, figsize=figure_size)
    else:
        plt.savefig('emotion_epochs_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    figure_size = (14, 12)
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure_size)

    train_probs = model.predict((images_train - mean) / std)
    train_results = np.argmax(train_probs, axis=1)
    conf_mx = confusion_matrix(labels_train, train_results)

    sns.heatmap(conf_mx, annot=True, ax=ax, cbar=False, cmap='binary')
    ax.set_ylim(conf_mx.shape[0] - 0, -0.5)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    s = 0
    for i in range(0, np.size(np.unique(classes))):
        s = s + conf_mx[i, i]
    accuracy = s / sum(sum(conf_mx)) * 100
    ax.set_title(index + ' Overall Training Accuracy: {0:.3f}%'.format(accuracy))
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    test_probs = model.predict((images_test - mean) / std)
    test_results = np.argmax(test_probs, axis=1)
    conf_mx = confusion_matrix(labels_test, test_results)

    sns.heatmap(conf_mx, annot=True, ax=ax2, cbar=False, cmap='binary')
    ax2.set_ylim(conf_mx.shape[0] - 0, -0.5)
    ax2.set_xlabel('Predicted labels')
    ax2.set_ylabel('True labels')
    s = 0
    for i in range(0, np.size(np.unique(classes))):
        s = s + conf_mx[i, i]
    accuracy = s / sum(sum(conf_mx)) * 100
    ax2.set_title(index + ' Overall Testing Accuracy: {0:.3f}%'.format(accuracy))
    ax2.xaxis.set_ticklabels(classes)
    ax2.yaxis.set_ticklabels(classes)

    if num_classes < 3:
        f1 = f1_score(labels_train, train_results, average='binary', pos_label=positive_index)
        [fpr, tpr, _] = roc_curve(labels_train, train_probs[:, positive_index], pos_label=positive_index)
        labels_temp = np.where(labels_train == positive_index, 1, 0)
        AUC = roc_auc_score(labels_temp, train_probs[:, positive_index])

        ax3.plot(fpr, tpr)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('Area Under Curve: {0:.3f}, F1 Score: {1:.3f}'.format(AUC, f1))

        f1 = f1_score(labels_test, test_results, average='binary', pos_label=positive_index)
        [fpr, tpr, _] = roc_curve(labels_test, test_probs[:, positive_index], pos_label=positive_index)
        labels_temp = np.where(labels_test == positive_index, 1, 0)
        AUC = roc_auc_score(labels_temp, test_probs[:, positive_index])

        ax4.plot(fpr, tpr)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('Area Under Curve: {0:.3f}, F1 Score: {1:.3f}'.format(AUC, f1))
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for k in range(num_classes):
            fpr[k], tpr[k], _ = roc_curve(labels_test, test_probs[:, k], pos_label=k)
            roc_auc[k] = auc(fpr[k], tpr[k])

        # Compute micro-average ROC curve and ROC area
        labels_test_binarized = label_binarize(labels_test, classes=range(0, num_classes))
        fpr["micro"], tpr["micro"], _ = roc_curve(labels_test_binarized.ravel(), test_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(num_classes):
            mean_tpr += interp(all_fpr, fpr[k], tpr[k])

        # Finally average it and compute AUC
        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        f1_micro = f1_score(labels_test, test_results, average='micro')
        f1_macro = f1_score(labels_test, test_results, average='macro')

        ax3.plot(fpr["micro"], tpr["micro"])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('Micro-average Area Under Curve: {0:.3f}, Micro F1: {1:.3f}'.format(roc_auc["micro"], f1_micro))

        ax4.plot(fpr["macro"], tpr["macro"])
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('Macro-average Area Under Curve: {0:.3f}, Macro F1: {1:.3f}'.format(roc_auc["macro"], f1_macro))

    if not final:
        plt.savefig('./CNN_CV/emotion_test_a={}_d={}_m={}_l={}_fold_{}_{}.png'.format(alpha0, decay, model_name, layers_back,
                                                                                 index, time.time()), dpi=fig.dpi, figsize=figure_size)
    else:
        plt.savefig('emotion_test_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    if final and num_test_classes > num_classes:
        test_probs = model.predict((images_final_test - mean) / std)
        test_results = []
        for prob in test_probs:
            j = np.argmax(prob)
            if prob[j] > best_hyperparameters['certainty_threshold']:
                re_index = test_classes.index(classes[j])
                test_results.append(re_index)
            else:
                test_results.append(removed_index)
        conf_mx = confusion_matrix(labels_final_test, test_results)

        figure_size = (12, 8)
        fig, ax = plt.subplots(1, 1, figsize=figure_size)

        sns.heatmap(conf_mx, annot=True, ax=ax, cbar=False, cmap='binary')
        ax.set_ylim(conf_mx.shape[0] - 0, -0.5)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        s = 0
        for i in range(0, np.size(np.unique(test_classes))):
            s = s + conf_mx[i, i]
        accuracy = s / sum(sum(conf_mx)) * 100
        ax.set_title(index + ' Overall Testing Accuracy: {0:.3f}%'.format(accuracy))
        ax.xaxis.set_ticklabels(test_classes)
        ax.yaxis.set_ticklabels(test_classes)

        plt.savefig('emotion_threshold_test_c={}_{}.png'.format(best_hyperparameters['certainty_threshold'], time.time()),
                    dpi=fig.dpi, figsize=figure_size)
        fig.clear()
        plt.close(fig)

    if num_classes < 3:
        metric = f1
    else:
        metric = f1_macro
    if resample_flag:
        metric = accuracy/100

    return model, metric, mean, std


def cross_val(args):
    alpha0 = args['alpha0']
    decay = args['decay']
    layers_back = args['layers_back']

    metrics = []
    k = 1
    for train_indices, test_indices in kf.split(images, y=labels):
        _, metric, _, _ = train_model(images[train_indices], labels[train_indices], images[test_indices],
                                      labels[test_indices], model_name, image_res[model_name], n_epochs=EPOCHS,
                                      index=str(k), alpha0=alpha0, decay=decay, layers_back=layers_back)
        tf.keras.backend.clear_session()
        metrics.append(1-metric)
        k += 1

    cv_mean = np.mean(metrics)

    return cv_mean


def random_search(n_iterations=10):
    best_hyperparameters = {}
    best_mean = 0

    cv_means = []
    best_means = []
    for j in range(0, n_iterations):

        k = 1

        min_exp, max_exp = -3, 0
        exponent = np.random.uniform(min_exp, max_exp)
        alpha0 = 10 ** exponent
        decay = np.random.rand(1)[0] * 0.1 + 0.9
        layers_back = np.random.randint(0, 5)
        # model_name = model_names[np.random.randint(0, 6)]

        metrics = []
        for train_indices, test_indices in kf.split(images, y=labels):
            _, metric, _, _ = train_model(images[train_indices], labels[train_indices], images[test_indices],
                                          labels[test_indices], model_name, image_res[model_name], n_epochs=EPOCHS,
                                          index=str(k), alpha0=alpha0, decay=decay, layers_back=layers_back)
            tf.keras.backend.clear_session()
            metrics.append(metric)
            k += 1

        cv_mean = np.mean(metrics)
        cv_means.append(1-cv_mean)
        if cv_mean > best_mean:
            best_hyperparameters = {'alpha0': alpha0, 'decay': decay, 'layers_back': layers_back}
            best_mean = cv_mean
            best_means.append(1-best_mean)

        print("{}/{}, best score: {}".format(j + 1, n_iterations, best_mean))

    figure_size = (14, 8)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.plot(cv_means, label='Iteration Loss')
    ax.plot(best_means, label='Best Loss')
    plt.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.savefig('emotion_tuning_loss_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    return best_hyperparameters


def bayesian_search(n_iterations=10):
    space = {'alpha0': 10 ** hp.uniform('alpha0', -4, -2),
             'decay': hp.uniform('decay', 0.7, 1.0),
             'layers_back': hp.choice('layers_back', [0]),  # hp.randint('layers_back', 3),
             }

    # minimize the objective over the space
    trials = Trials()
    best_hyperparameters = fmin(cross_val, space, algo=tpe.suggest, max_evals=n_iterations, trials=trials)
    best_hyperparameters['alpha0'] = 10 ** best_hyperparameters['alpha0']

    figure_size = (14, 8)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.plot(trials.losses(), label='Iteration Loss')
    trial_mins = [np.min(trials.losses()[:i + 1]) for i in range(0, len(trials.losses()))]
    ax.plot(trial_mins, label='Best Loss')
    plt.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.savefig('emotion_tuning_loss_{}.png'.format(time.time()), dpi=fig.dpi, figsize=figure_size)

    fig.clear()
    plt.close(fig)

    return best_hyperparameters


def threshold_search(grid_size=10):
    if num_test_classes > num_classes:
        k = 1
        models = {}
        for train_indices, test_indices in kf.split(images, y=labels):  #  caching cv models to save computation in grid search
            model, _, mean, std = train_model(images[train_indices], labels[train_indices],
                                              images[test_indices],
                                              labels[test_indices], model_name, image_res[model_name],
                                              n_epochs=EPOCHS, final=False,
                                              alpha0=best_hyperparameters['alpha0'],
                                              decay=best_hyperparameters['decay'],
                                              layers_back=best_hyperparameters['layers_back'],
                                              threshold_check=True)
            tf.keras.backend.clear_session()
            models[str(k)] = {'model': model, 'mean': mean, 'std': std}
            k += 1

        certainty_thresholds = np.linspace(1 / num_classes, 1, grid_size)
        best_score = 0
        best_threshold = 1 / num_classes
        for m, certainty_threshold in enumerate(certainty_thresholds):
            print("Certainty Threshold: {}".format(certainty_threshold))
            metrics = []
            k = 1
            for train_indices, test_indices in kf.split(images, y=labels):
                model = models[str(k)]['model']
                mean = models[str(k)]['mean']
                std = models[str(k)]['std']

                test_probs = model.predict((images[test_indices] - mean) / std)
                test_results = []
                for prob in test_probs:
                    j = np.argmax(prob)
                    if prob[j] > certainty_threshold:
                        re_index = test_classes.index(classes[j])
                        test_results.append(re_index)
                    else:
                        test_results.append(removed_index)
                conf_mx = confusion_matrix(labels[test_indices], test_results)

                figure_size = (12, 8)
                fig, ax = plt.subplots(1, 1, figsize=figure_size)

                sns.heatmap(conf_mx, annot=True, ax=ax, cbar=False, cmap='binary')
                ax.set_ylim(conf_mx.shape[0] - 0, -0.5)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                s = 0
                for i in range(0, np.size(np.unique(test_classes))):
                    s = s + conf_mx[i, i]
                accuracy = s / sum(sum(conf_mx)) * 100
                ax.set_title(str(k) + ' Overall Testing Accuracy: {0:.3f}%'.format(accuracy))
                ax.xaxis.set_ticklabels(test_classes)
                ax.yaxis.set_ticklabels(test_classes)

                plt.savefig('./CNN_CV/emotion_threshold_test_c={}_fold_{}_{}.png'.format(certainty_threshold, k,
                                                                                         time.time()),
                            dpi=fig.dpi, figsize=figure_size)
                fig.clear()
                plt.close(fig)
                tf.keras.backend.clear_session()

                print("Fold {} Accuracy: {}".format(k, accuracy / 100))
                metrics.append(accuracy / 100)
                k += 1

            cv_mean = np.mean(metrics)
            if cv_mean > best_score:
                best_threshold = certainty_threshold
                best_score = cv_mean

            print("{}/{}, best score: {}, best threshold: {}".format(m + 1, np.size(certainty_thresholds), best_score, best_threshold))

    else:
        best_threshold = 1 / num_classes
        print("Setting certainty threshold to default: {}".format(best_threshold))

    return best_threshold


if __name__ == '__main__':
    # model_names = ["emopy", "vgg16", "vgg19", "vggface2", "inception_resnet", "mobilenet"]
    model_name = 'emopy'

    BATCH_SIZE = 32
    EPOCHS = 100
    n_samples = 2 ** 14
    resample_flag = True

    image_res = {'emopy': 48, 'vgg16': 224, 'vgg19': 224, 'vggface2': 224, 'inception_resnet': 299, 'mobilenet': 224}
    preprocess = {'emopy': lambda image: tf.image.rgb_to_grayscale(image) / 255.0,
                  'vgg16': tf.keras.applications.vgg16.preprocess_input,
                  'vgg19': tf.keras.applications.vgg19.preprocess_input,
                  'vggface2': tf.keras.applications.resnet50.preprocess_input,
                  'inception_resnet': tf.keras.applications.inception_resnet_v2.preprocess_input,
                  'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input}

    vggface2path = os.getcwd() + "\\weights.h5"
    emopypath = os.getcwd() + "\\conv_model_0123456.h5"

    images, labels, images_test, labels_test = read_data()
    i = 0
    images, labels = shuffle_select_process(images, labels, resample_flag)
    images_test, labels_test = shuffle_select_process(images_test, labels_test, resample_flag)

    os.chdir(model_path)

    n_folds = 3
    n_iterations = 10
    kf = KFold(n_splits=n_folds, shuffle=True)
    best_hyperparameters = bayesian_search(n_iterations)
    # best_hyperparameters = random_search(n_iterations)

    best_hyperparameters['certainty_threshold'] = threshold_search(grid_size=20)

    model, _, mean, std = train_model(images, labels, images_test, labels_test, model_name, image_res[model_name],
                                      n_epochs=EPOCHS, final=True, alpha0=best_hyperparameters['alpha0'],
                                      decay=best_hyperparameters['decay'],
                                      layers_back=best_hyperparameters['layers_back'])

    model.save("cnn_model_{}.h5".format(time.time()))

    settings = {"model": model_name, "image_res": image_res[model_name], "certainty_threshold": best_hyperparameters['certainty_threshold'],
                "feature_mean": float(mean), "feature_std": float(std)}
    with open('cnn_settings_{}.json'.format(time.time()), 'w', encoding='utf-8') as outfile:
        json.dump(settings, outfile, ensure_ascii=False, indent=2)

    best_hyperparameters['layers_back'] = int(best_hyperparameters['layers_back'])
    with open('cnn_best_hyperparameters_{}.json'.format(time.time()), 'w', encoding='utf-8') as outfile:
        json.dump(best_hyperparameters, outfile, ensure_ascii=False, indent=2)
