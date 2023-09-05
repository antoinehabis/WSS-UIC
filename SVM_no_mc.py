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

def main(training_condition):
    image_list = os.listdir(path_prediction_features)
    # image_list = ['test_040', 'test_090', 'test_026', 'test_104', 'test_027', 'test_002', 'test_029', 'test_001', 'test_061', 'test_108', 'test_110', 'test_004', 'test_092', 'test_066', 'test_122', 'test_074', 'test_079', 'test_084', 'test_117', 'test_010', 'test_011', 'test_102', 'test_097', 'test_116']
    epochs_range = range(700, 701, 100)
    pass_percentage = 0.025
    n_tables = 1
    mean_tables = np.zeros((n_tables, 4, 4))
    for i, epochs in enumerate(epochs_range): # Iterate over epochs

        score_tables = [l for l in range(len(image_list))] # Initialize the metric table list of every image 
        for n, image in tqdm(enumerate(image_list)): # Iterate over images
            print(n)
            image_table = generate_progression_table(image, epochs, int(pass_percentage*epochs))
            score_tables[n] = image_table
            
        score_table_mean = np.mean(score_tables, 0)

        df = pd.DataFrame(score_table_mean,
                        index=['VGG-16', 'SVM Pass 1', 'SVM Pass 2', 'SVM Pass 3'],
                        columns=['Accuracy','Precision', 'Recall', 'F1 Score'])

        mean_tables[i] = score_table_mean
        tables_directory = os.path.join(path_metric_tables, training_condition)
        if not os.path.exists(tables_directory):
            os.makedirs(tables_directory)
        
        table_name = 'metric_table_{}_epochs_test24_20percent.csv'.format(epochs)
        df.to_csv(os.path.join(tables_directory, table_name))
    return mean_tables

if __name__ == '__main__':
    main('SVM_initialization_no_MC')

