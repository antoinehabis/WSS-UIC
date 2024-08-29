import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
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
    accuracy = np.sum(trues == predictions) / trues.shape[0]
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


def compute_new_dataset(features, predictions, trues, initialization, nb_scribble=60):

    if initialization:
        binary_predictions = (predictions >= optimal_threshold).astype(int)
        n_limit = 5000
        argsort = np.argsort(predictions)
        tp_indexes = argsort[-np.min([np.sum(binary_predictions), n_limit]) :]
        tn_indexes = argsort[: np.min([np.sum((1 - binary_predictions)), n_limit])]

        pos = features[tp_indexes]
        neg = features[tn_indexes]

        data = np.concatenate((pos, neg), axis=0)

        y = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))
        return data, y

    else:
        binary_predictions = (predictions >= 0.5).astype(int)

        tp = binary_predictions * trues
        tn = (1 - binary_predictions) * (1 - trues)
        fn = trues * (1 - binary_predictions)
        fp = (1 - trues) * binary_predictions

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


def generate_progression_table(
    image, model_name, init_epochs=1000, inc_epochs=30, save_patches_preds_corr="y"
):
    
    if save_patches_preds_corr == "y":
        path_corrections_save = os.path.join(path_prediction_features, image)
        if not os.path.exists(path_corrections_save):
            os.makedirs(path_corrections_save)

    #################### LOAD THE DATA #####################
    current_image_path = os.path.join(path_prediction_features, image)
    mc_predictions = np.load(
        os.path.join(current_image_path, "predictions"+model_name+".npy")
    )
    predictions = np.mean(np.squeeze(mc_predictions), axis=0)
    trues = np.load(os.path.join(current_image_path, "trues.npy"))
    features = np.load(os.path.join(current_image_path, "features"+model_name+".npy"))
    features = PCA(1000).fit_transform(features)  ### normalize the features for the svm

    #################### INITIALIZE SVM ####################

    row1 = metrics(predictions >= optimal_threshold, trues)
    data0, y0 = compute_new_dataset(features, predictions, trues, initialization=True)
    svm = SGDClassifier(shuffle=True,loss='modified_huber')
    svm.eta0 = 1e-3
    for i in range(init_epochs):
        svm.partial_fit(data0, y0, classes=[0, 1])
    #################### PASS 1 SVM ########################
        
    data1, y1, indexes_fn1, indexes_fp1 = compute_new_dataset(
        features, predictions, trues, initialization=False
    )
    data1, y1 = np.concatenate((data1,data0),axis = 0), np.concatenate((y1,y0),axis = 0)

    for i in range(inc_epochs):
        svm.partial_fit(data1, y1)

    a_predictions = svm.predict_proba(features)[:,1]
    a_predictions[indexes_fn1] = 1
    a_predictions[indexes_fp1] = 0

    if save_patches_preds_corr == "y":
        np.save(
            os.path.join(path_corrections_save, "predictions_correction_1_heatmap.npy"),
            a_predictions.reshape(1, -1),
        )

    row2 = metrics(a_predictions >= 0.5, trues)

    if np.around(row2[0], 3) == 1:
        row3, row4, row5 = [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    #################### PASS 2 SVM ########################
    
    data2, y2, indexes_fn2, indexes_fp2 = compute_new_dataset(
        features, a_predictions, trues, initialization=False
    )
    data2, y2 = np.concatenate((data2,data1),axis = 0), np.concatenate((y2,y1),axis = 0)

    for i in range(inc_epochs):
        svm.partial_fit(data2, y2, classes=[0, 1])
    b_predictions = svm.predict_proba(features)[:,1]

    b_predictions[indexes_fn1] = 1
    b_predictions[indexes_fp1] = 0
    b_predictions[indexes_fn2] = 1
    b_predictions[indexes_fp2] = 0

    if save_patches_preds_corr == "y":
        np.save(
            os.path.join(path_corrections_save, "predictions_correction_2_heatmap.npy"),
            b_predictions.reshape(1, -1),
        )

    row3 = metrics(b_predictions >= 0.5, trues)

    if np.around(row3[0], 3) == 1:
        row4, row5 = [1, 1, 1, 1], [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    #################### PASS 3 SVM ########################
    
    data3, y3, indexes_fn3, indexes_fp3 = compute_new_dataset(
        features, b_predictions, trues, initialization=False
    )
    data3, y3 = np.concatenate((data3,data2),axis = 0), np.concatenate((y3,y2),axis = 0)

    for i in range(inc_epochs):
        svm.partial_fit(data3, y3, classes=[0, 1])

    c_predictions = svm.predict_proba(features)[:,1]

    c_predictions[indexes_fn1] = 1
    c_predictions[indexes_fp1] = 0
    c_predictions[indexes_fn2] = 1
    c_predictions[indexes_fp2] = 0
    c_predictions[indexes_fn3] = 1
    c_predictions[indexes_fp3] = 0

    if save_patches_preds_corr == "y":
        np.save(
            os.path.join(path_corrections_save, "predictions_correction_3_heatmap.npy"),
            c_predictions.reshape(1, -1),
        )

    row4 = metrics(c_predictions >= 0.5, trues)

    if np.around(row4[0], 3) == 1:
        row5 = [1, 1, 1, 1]
        return np.array([row1, row2, row3, row4, row5])

    #################### PASS 4 SVM ########################
    data4, y4, indexes_fn4, indexes_fp4 = compute_new_dataset(
        features, c_predictions, trues, initialization=False
    )
    data4, y4 = np.concatenate((data4,data3),axis = 0), np.concatenate((y4,y3),axis = 0)

    for i in range(inc_epochs):
        svm.partial_fit(data4, y4, classes=[0, 1])

    d_predictions = svm.predict_proba(features)[:,1]
    d_predictions[indexes_fn1] = 1
    d_predictions[indexes_fp1] = 0
    d_predictions[indexes_fn2] = 1
    d_predictions[indexes_fp2] = 0
    d_predictions[indexes_fn3] = 1
    d_predictions[indexes_fp3] = 0
    d_predictions[indexes_fn4] = 1
    d_predictions[indexes_fp4] = 0

    row5 = metrics(d_predictions >= 0.5, trues)

    if save_patches_preds_corr == "y":
        np.save(
            os.path.join(path_corrections_save, "predictions_correction_4_heatmap.npy"),
            d_predictions.reshape(1, -1),
        )

    return np.array([row1, row2, row3, row4, row5])
