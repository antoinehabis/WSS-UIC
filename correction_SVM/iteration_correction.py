import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parent.parent)
from config import *
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import numpy as np

def metrics(predictions, trues):
    tp = np.sum((predictions) * trues)
    tn = np.sum((1 - predictions) * (1 - trues))
    fp = np.sum((predictions) * (1 - trues))
    fn = np.sum((1 - predictions) * trues)
    eps = 1e-6
    accuracy = np.sum(trues == (predictions)) / trues.shape[0]
    precision_healthy = tn / (tn + fn + eps)
    precision_tumor = tp / (tp + fp + eps)
    m_precision = (precision_healthy + precision_tumor) / 2
    recall_tumor = tp / (tp + fn + eps)
    recall_healthy = tn / (tn + fp + eps)
    m_recall = (recall_tumor + recall_healthy) / 2
    f1 = 2 * (m_precision * m_recall) / (m_precision + m_recall + eps)
    balanced_accuracy = (m_precision + m_recall) / 2

    return balanced_accuracy, m_precision, m_recall, f1


def find_indexes(x, nb_scribble):
    y = np.argwhere(x).flatten()
    np.random.shuffle(y)
    return y[:nb_scribble]


def compute_new_dataset(features, predictions, trues, initialization, nb_scribble=10):
    if initialization:
        thresh = 0.33
        n_limit = 1000
        tp = np.sum((predictions > thresh) * trues)
        tn = np.sum((predictions < thresh) * (1 - trues))
        fp = np.sum((predictions > thresh) * (1 - trues))
        fn = np.sum((predictions < thresh) * trues)

        tn_indexes = np.argwhere(predictions < thresh).flatten()[:n_limit]
        tp_indexes = np.argwhere(predictions > thresh).flatten()[:n_limit]

        positive_index = tp_indexes
        negative_index = tn_indexes

        nb_scribble = len(tp_indexes)

        pos = features[tp_indexes]
        neg = features[tn_indexes]

        data = np.concatenate((pos, neg), axis=0)

        y = np.concatenate(
            (np.ones(tp_indexes.shape[0]), np.zeros(tn_indexes.shape[0]))
        )
        return data, y

    else:
        tp = predictions * trues
        tn = (1 - predictions) * (1 - trues)

        fn = trues * (1 - predictions)
        fp = (1 - trues) * predictions

        if np.sum(fn) == 0:
            ### Instead of taking patches from FN we take patches from TP

            indexes_fn = find_indexes(tp, nb_scribble)
        ### Usually what happens
        else:
            indexes_fn = find_indexes(fn, nb_scribble)

        ### If there are no False Positives:

        if np.sum(fp) == 0:
            ### Instead of taking patches from FP we take patches from TN
            indexes_fp = find_indexes(tn, nb_scribble)

        ### Usually what happens
        else:
            indexes_fp = find_indexes(fp, nb_scribble)

        positive_index = indexes_fn
        negative_index = indexes_fp
        pos = features[positive_index]
        neg = features[negative_index]

        data = np.concatenate((pos, neg), axis=0)
        y = np.concatenate(
            (np.ones(indexes_fn.shape[0]), np.zeros(indexes_fp.shape[0]))
        )

        return data, y, indexes_fn, indexes_fp


def generate_progression_table(image, init_epochs=800, inc_epochs=40):
    current_image_path = os.path.join(path_prediction_features, image)
    mc_predictions = np.load(os.path.join(current_image_path, "predictions.npy"))
    predictions = np.mean(np.squeeze(mc_predictions), axis=0)
    trues = np.load(os.path.join(current_image_path, "trues.npy"))
    features = np.load(os.path.join(current_image_path, "features.npy"))
    features = PCA(1000).fit_transform(features)
    row1 = metrics(predictions > optimal_threshold, trues)

    data, y = compute_new_dataset(features, predictions, trues, initialization=True)

    svm = SGDClassifier(shuffle=True, learning_rate="constant")
    svm.eta0 = 1e-3

    # Initialize SVM
    for i in range(init_epochs):
        svm.partial_fit(data, y, classes=np.unique(y))
    # SVM pass 1
    data, y, indexes_fn1, indexes_fp1 = compute_new_dataset(
        features, predictions, trues, initialization=False
    )

    for i in range(inc_epochs):
        svm.partial_fit(data, y)

    a_predictions = svm.predict(features)
    a_predictions[indexes_fn1] = 1
    a_predictions[indexes_fp1] = 0

    row2 = metrics(a_predictions, trues)

    if np.around(row2[0], 3) == 1:
        row3, row4, row5 = [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    # SVM pass 2

    data, y, indexes_fn2, indexes_fp2 = compute_new_dataset(
        features, a_predictions, trues, initialization=False
    )

    for i in range(inc_epochs):
        svm.partial_fit(data, y, classes=[0, 1])
    b_predictions = svm.predict(features)
    b_predictions[indexes_fn1] = 1
    b_predictions[indexes_fp1] = 0
    b_predictions[indexes_fn2] = 1
    b_predictions[indexes_fp2] = 0
    row3 = metrics(b_predictions, trues)

    if np.around(row3[0], 3) == 1:
        row4, row5 = [1, 1, 1, 1], [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    # SVM pass 3
    data, y, indexes_fn3, indexes_fp3 = compute_new_dataset(
        features, b_predictions, trues, initialization=False
    )
    for i in range(inc_epochs):
        svm.partial_fit(data, y, classes=[0, 1])
    c_predictions = svm.predict(features)
    c_predictions[indexes_fn1] = 1
    c_predictions[indexes_fp1] = 0
    c_predictions[indexes_fn2] = 1
    c_predictions[indexes_fp2] = 0
    c_predictions[indexes_fn3] = 1
    c_predictions[indexes_fp3] = 0

    row4 = metrics(c_predictions, trues)

    if np.around(row4[0], 3) == 1:
        row5 = [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    # SVM pass 3
    data, y, indexes_fn4, indexes_fp4 = compute_new_dataset(
        features, c_predictions, trues, initialization=False
    )

    for i in range(inc_epochs):
        svm.partial_fit(data, y, classes=[0, 1])

    d_predictions = svm.predict(features)
    d_predictions[indexes_fn1] = 1
    d_predictions[indexes_fp1] = 0
    d_predictions[indexes_fn2] = 1
    d_predictions[indexes_fp2] = 0
    d_predictions[indexes_fn3] = 1
    d_predictions[indexes_fp3] = 0
    d_predictions[indexes_fn4] = 1
    d_predictions[indexes_fp4] = 0

    row5 = metrics(d_predictions, trues)

    return np.array([row1, row2, row3, row4, row5])


# def generate_progression_table_dynamic_lr(image,
#                                           init_epochs=800,
#                                           corr_epochs=40,
#                                           image_uncertainty = 1.):
#     '''
#     svm = SGDClassifier(shuffle= True, learning_rate='constant')
#     svm_lr =
#     svm.eta0 = 1e-3 '''

#     current_image_path = os.path.join(path_prediction_features, image)
#     mc_predictions = np.load(os.path.join(current_image_path,'predictions.npy'))
#     predictions = np.mean(np.squeeze(mc_predictions),axis =0)
#     trues = np.load(os.path.join(current_image_path,'trues.npy'))
#     features = np.load(os.path.join(current_image_path,'features.npy'))
#     features = PCA(1000).fit_transform(features)

#     # VGG 16 metrics
#     row1 = metrics(predictions,trues)

#     data,y = compute_new_dataset(features,
#                                 predictions,
#                                 trues,
#                                 initialization=True)

#     svm = SGDClassifier(shuffle= True,
#                         learning_rate='constant')
#     svm.eta0 = (1e-4)

#     # Initialize SVM
#     for i in range (init_epochs):
#         svm.partial_fit(data,
#                         y,
#                         classes=np.unique(y))

#     svm.eta0 = (1e-3)*image_uncertainty

#     # SVM pass 1
#     data, y, indexes_fn1, indexes_fp1  = compute_new_dataset(features,
#                                                          predictions,
#                                                          trues,
#                                                          initialization=False)

#     for i in range (corr_epochs):
#         svm.partial_fit(data,y)

#     a_predictions = svm.predict(features)
#     a_predictions[indexes_fn1] = 1
#     a_predictions[indexes_fp1] = 0
#     row2 = metrics(a_predictions,trues)

#     # SVM pass 2
#     data, y, indexes_fn2, indexes_fp2  = compute_new_dataset(features,
#                                                          a_predictions,
#                                                          trues,
#                                                          initialization=False)

#     if metrics(a_predictions,trues)[0] == 1:
#         b_predictions = a_predictions
#         b_predictions[indexes_fn2] = 1
#         b_predictions[indexes_fp2] = 0
#         row3=[1,1,1,1]
#         row4=row3
#         return np.array([row1,
#                      row2,
#                      row3,
#                      row4])


#     for i in range (corr_epochs):
#         svm.partial_fit(data,
#                         y,
#                         classes=[0,1])
#     b_predictions = svm.predict(features)
#     b_predictions[indexes_fn1] = 1
#     b_predictions[indexes_fp1] = 0
#     b_predictions[indexes_fn2] = 1
#     b_predictions[indexes_fp2] = 0
#     row3 = metrics(b_predictions,trues)


#     #SVM pass 3
#     data, y, indexes_fn3, indexes_fp3  = compute_new_dataset(features,
#                                                          b_predictions,
#                                                          trues,
#                                                          initialization=False)
#     if metrics(b_predictions,trues)[0] == 1:
#         row4 = [1,1,1,1]
#         return np.array([row1,
#                      row2,
#                      row3,
#                      row4])


#     for i in range (corr_epochs):
#         svm.partial_fit(data,
#                         y,
#                         classes=[0,1])
#     c_predictions = svm.predict(features)
#     c_predictions[indexes_fn1] = 1
#     c_predictions[indexes_fp1] = 0
#     c_predictions[indexes_fn2] = 1
#     c_predictions[indexes_fp2] = 0
#     c_predictions[indexes_fn3] = 1
#     c_predictions[indexes_fp3] = 0
#     row4 = metrics(c_predictions,trues)

#     return np.array([row1,
#                      row2,
#                      row3,
#                      row4])
