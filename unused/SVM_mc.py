from config import *
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import logistic
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from math import floor, sqrt
from iteration_correction import *
from uncertainty_metrics import *

def main(training_condition):
    test_set =  ['test_040', 'test_090', 'test_026', 'test_104', 'test_027', 'test_002', 'test_029', 'test_001', 'test_061', 'test_108', 'test_110', 'test_004', 'test_092', 'test_066', 'test_122', 'test_074', 'test_079', 'test_084', 'test_117', 'test_010', 'test_011', 'test_102', 'test_097', 'test_116']
    # val_set =  ['test_016', 'test_068', 'test_021', 'test_075', 'test_113', 'test_094', 'test_082', 'test_105', 'test_073', 'test_071', 'test_121', 'test_064', 'test_065', 'test_069', 'test_051', 'test_008', 'test_033', 'test_030', 'test_048', 'test_052', 'test_046', 'test_099', 'test_038', 'test_013'] 
    image_list = test_set

    epochs = 1000
    n_tables = 1
    pass_percentage = 0.08
    
    for s in range(n_tables):
        
        score_tables = np.zeros((len(image_list),4,4))
        for n, image in tqdm(enumerate(image_list)):
            current_image_path = os.path.join(path_prediction_features, image)
            mc_predictions = np.load(os.path.join(current_image_path,'predictions.npy'))
            predictions = np.squeeze(mc_predictions)
            MAX_ENTROPY = 0.4
            MIN_ENTROPY = 0.09
            corr_effect = (compute_entropy(predictions,patch_level=False) - MIN_ENTROPY)/(MAX_ENTROPY- MIN_ENTROPY)
            corr_epochs = np.clip((corr_effect*epochs*pass_percentage),1,None).astype(int)
            image_table = generate_progression_table(image, epochs, corr_epochs)
            score_tables[n] = image_table 
            
        score_table_mean = np.mean(score_tables, 0)
        df = pd.DataFrame(score_table_mean,
                        index=['VGG-16', 'SVM Pass 1', 'SVM Pass 2', 'SVM Pass 3'],
                        columns=['Accuracy','Precision', 'Recall', 'F1 Score'])

        mean_tables = score_table_mean
        tables_directory = os.path.join(path_metric_tables, training_condition, str(epochs)+'epochs')        
        if not os.path.exists(tables_directory):
            os.makedirs(tables_directory)        
        table_name = 'metric_table_validation_{}epochs_{}.csv'.format(epochs,s)
        df.to_csv(os.path.join(tables_directory, table_name))
        
    return mean_tables

if __name__ == '__main__':
    main('700validation_MC_epochs_nosplit')
