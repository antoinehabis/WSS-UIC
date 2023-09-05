from config import *
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import logistic
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from math import ceil
from uncertainty_metrics import *
from iteration_correction import *
from utils import *

# def compute_new_dataset(features,
#                         predictions,
#                         trues,
#                         initialization,
#                         nb_scribble=10):

    
#     if initialization:

#         thresh = 0.33
#         n_limit = 1000
#         tp = np.sum((predictions>thresh) * trues)
#         tn = np.sum((predictions<thresh) * (1 - trues))
#         fp = np.sum((predictions>thresh) * (1 - trues))
#         fn = np.sum((predictions<thresh) * trues)
    
        
#         tn_indexes = np.argwhere(predictions < thresh).flatten()[:n_limit]
#         tp_indexes = np.argwhere(predictions > thresh).flatten()[:n_limit]

#         positive_index = tp_indexes
#         negative_index = tn_indexes

#         nb_scribble = len(tp_indexes)

       
#         pos = features[tp_indexes]
#         neg = features[tn_indexes]

#         data = np.concatenate((pos,neg), axis = 0)

#         # y = np.concatenate((np.ones(nb_scribble + min(nb_scribble,max_fn)), np.zeros(nb_tn + min(nb_scribble,max_fp))))
#         y = np.concatenate((np.ones(tp_indexes.shape[0]), np.zeros(tn_indexes.shape[0])))
#         return data, y

        
#     else:    
        
#         tp = predictions * trues
#         tn = (1-predictions) * (1 - trues)
        
# def compute_new_dataset(features,
#                         predictions,
#                         trues,
#                         initialization,
#                         nb_scribble=10):

    
#     if initialization:

#         thresh = 0.33
#         n_limit = 1000
#         tp = np.sum((predictions>thresh) * trues)
#         tn = np.sum((predictions<thresh) * (1 - trues))
#         fp = np.sum((predictions>thresh) * (1 - trues))
#         fn = np.sum((predictions<thresh) * trues)
    
        
#         tn_indexes = np.argwhere(predictions < thresh).flatten()[:n_limit]
#         tp_indexes = np.argwhere(predictions > thresh).flatten()[:n_limit]

#         positive_index = tp_indexes
#         negative_index = tn_indexes

#         nb_scribble = len(tp_indexes)

       
#         pos = features[tp_indexes]
#         neg = features[tn_indexes]

#         data = np.concatenate((pos,neg), axis = 0)

#         # y = np.concatenate((np.ones(nb_scribble + min(nb_scribble,max_fn)), np.zeros(nb_tn + min(nb_scribble,max_fp))))
#         y = np.concatenate((np.ones(tp_indexes.shape[0]), np.zeros(tn_indexes.shape[0])))
#         return data, y

        
#     else:    
        
#         tp = predictions * trues
#         tn = (1-predictions) * (1 - trues)
        
       
#         fn = trues * (1 - predictions)
#         fp = (1-trues) * predictions
        
#         if np.sum(fn) == 0:
#             ### Instead of taking patches from FN we take patches from TP
#             max_fn = np.sum(trues).astype(int)

#             indexes_fn = np.argwhere(tp).flatten()[:nb_scribble,]
#         ### Usually what happens
#         else:
#             max_fn = len(np.argwhere(fn).flatten())
#             indexes_fn = np.argwhere(fn).flatten()[:nb_scribble]

#         ### If there are no False Positives:

#         if np.sum(fp) == 0:
#             ### Instead of taking patches from FP we take patches from TN

#             max_fp = np.sum((1 - trues)).astype(int)
#             indexes_fp = np.argwhere(tn).flatten()[:nb_scribble]

#         ### Usually what happens
#         else:
#             max_fp = len(np.argwhere(fp).flatten())
#             indexes_fp = np.argwhere(fp).flatten()[:nb_scribble]

        
#         positive_index = indexes_fn
#         negative_index = indexes_fp
#         pos = features[positive_index]
#         neg = features[negative_index]   
        
#         data = np.concatenate((pos,neg), axis = 0)
#         # y = np.concatenate((np.ones(min(nb_scribble,max_fn)), np.zeros(min(nb_scribble,max_fp))))
#         y = np.concatenate((np.ones(indexes_fn.shape[0]), np.zeros(indexes_fp.shape[0])))
#         return data, y, indexes_fn , indexes_fp
        

       
#         fn = trues * (1 - predictions)
#         fp = (1-trues) * predictions
        
#         if np.sum(fn) == 0:
#             ### Instead of taking patches from FN we take patches from TP
#             max_fn = np.sum(trues).astype(int)

#             indexes_fn = np.argwhere(tp).flatten()[:nb_scribble,]
#         ### Usually what happens
#         else:
#             max_fn = len(np.argwhere(fn).flatten())
#             indexes_fn = np.argwhere(fn).flatten()[:nb_scribble]

#         ### If there are no False Positives:

#         if np.sum(fp) == 0:
#             ### Instead of taking patches from FP we take patches from TN

#             max_fp = np.sum((1 - trues)).astype(int)
#             indexes_fp = np.argwhere(tn).flatten()[:nb_scribble]

#         ### Usually what happens
#         else:
#             max_fp = len(np.argwhere(fp).flatten())
#             indexes_fp = np.argwhere(fp).flatten()[:nb_scribble]

        
#         positive_index = indexes_fn
#         negative_index = indexes_fp
#         pos = features[positive_index]
#         neg = features[negative_index]   
        
#         data = np.concatenate((pos,neg), axis = 0)
#         # y = np.concatenate((np.ones(min(nb_scribble,max_fn)), np.zeros(min(nb_scribble,max_fp))))
#         y = np.concatenate((np.ones(indexes_fn.shape[0]), np.zeros(indexes_fp.shape[0])))
#         return data, y, indexes_fn , indexes_fp
        



def generate_progression_table(image,
                               init_epochs=800,
                               inc_epochs=40):

                               
    current_image_path = os.path.join(path_prediction_features, image)
    mc_predictions = np.load(os.path.join(current_image_path,'predictions.npy'))
    predictions = np.mean(np.squeeze(mc_predictions),axis =0)
    trues = np.load(os.path.join(current_image_path,'trues.npy'))
    features = np.load(os.path.join(current_image_path,'features.npy'))
    features = PCA(1000).fit_transform(features)

    row1 = metrics(predictions>optimal_threshold,
                   trues)

    data,y = compute_new_dataset(features,
                                predictions,
                                trues,
                                initialization=True)

    svm = SGDClassifier(shuffle= True)

    # Initialize SVM
    for i in range (init_epochs):
        svm.partial_fit(data,
                        y,
                        classes=np.unique(y))
    # SVM pass 1
    data, y, indexes_fn1, indexes_fp1  = compute_new_dataset(features,
                                                         predictions,
                                                         trues,
                                                         initialization=False)
    
    for i in range (inc_epochs):
        svm.partial_fit(data,y)

    a_predictions = svm.predict(features)
    a_predictions[indexes_fn1] = 1
    a_predictions[indexes_fp1] = 0
    
    row2 = metrics(a_predictions,trues)
    
    if metrics(a_predictions,trues)[0] == 1:
        row3=[1,1,1,1]
        row4=row3
        return np.array([row1,
                     row2,
                     row3,
                     row4])
    # SVM pass 2

    data, y, indexes_fn2, indexes_fp2  = compute_new_dataset(features,
                                                             a_predictions,
                                                             trues,
                                                             initialization=False)
    

   
    
    for i in range (inc_epochs):
        svm.partial_fit(data,
                        y,
                        classes=[0,1])
    b_predictions = svm.predict(features)
    b_predictions[indexes_fn1] = 1
    b_predictions[indexes_fp1] = 0
    b_predictions[indexes_fn2] = 1
    b_predictions[indexes_fp2] = 0
    row3 = metrics(b_predictions,trues)

    if metrics(b_predictions,trues)[0] == 1:
            row4 = [1,1,1,1]
            return np.array([row1,
                        row2,
                        row3,
                        row4])

    #SVM pass 3
    data, y, indexes_fn3, indexes_fp3  = compute_new_dataset(features,
                                                         b_predictions,
                                                         trues,
                                                         initialization=False)
   

    
    for i in range (inc_epochs):
        svm.partial_fit(data,
                        y,
                        classes=[0,1])
    c_predictions = svm.predict(features)
    c_predictions[indexes_fn1] = 1
    c_predictions[indexes_fp1] = 0
    c_predictions[indexes_fn2] = 1
    c_predictions[indexes_fp2] = 0
    c_predictions[indexes_fn3] = 1
    c_predictions[indexes_fp3] = 0
    row4 = metrics(c_predictions,trues)

    return np.array([row1,
                     row2,
                     row3,
                     row4])


def main(training_condition):
    # image_list = os.listdir(path_prediction_features)
    val_set =  ['test_016', 'test_068', 'test_021', 'test_075', 'test_113', 'test_094', 'test_082', 'test_105', 'test_073', 'test_071', 'test_121', 'test_064', 'test_065', 'test_069', 'test_051', 'test_008', 'test_033', 'test_030', 'test_048', 'test_052', 'test_046', 'test_099', 'test_038', 'test_013'] 
    image_list = val_set
    epochs_range = range(700, 701, 100)
    pass_percentage = 0.20
    mean_tables = [i for i in epochs_range]
    n_tables = 10

    for s in range(n_tables):
        for m, i in enumerate(epochs_range): # Iterate over epochs
            corr_epochs = int(pass_percentage*i)

            print(i, corr_epochs)
            score_tables = [l for l in range(len(image_list))] # Initialize the metric table list of every image 
            for n, image in tqdm(enumerate(image_list)): # Iterate over images
                print(image, i)
                image_table = generate_progression_table(image, i, corr_epochs)
                score_tables[n] = image_table

            score_table_mean = np.mean(score_tables, 0)
            df = pd.DataFrame(score_table_mean,
                            index=['VGG-16', 'SVM Pass 1', 'SVM Pass 2', 'SVM Pass 3'],
                            columns=['Accuracy','Precision', 'Recall', 'F1 Score'])

            mean_tables[m] = score_table_mean
            tables_directory = os.path.join(path_metric_tables, training_condition, str(i)+'epochs')
            
            if not os.path.exists(tables_directory):
                os.makedirs(tables_directory)
                

            table_name = 'metric_table_{}epochs_5percent_{}.csv'.format(i,s)
            df.to_csv(os.path.join(tables_directory, table_name))
            print(df)
    return mean_tables

if __name__ == '__main__':
    main('SVM__validation_initialization_no_MC')

