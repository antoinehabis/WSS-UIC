from config import *
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import logistic
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from math import ceil
from iteration_correction import *
from uncertainty_metrics import *

def main(training_condition):

    image_list = os.listdir(path_prediction_features)
    epochs_range = range(800, 801, 100)
    percentage_corrections = 0.20
    
    mean_tables = [epochs for epochs in epochs_range]
    for i, epochs in tqdm(enumerate(epochs_range)):
        score_tables = np.zeros((len(image_list),4,4))
        for n, image in tqdm(enumerate(image_list)):

            current_image_path = os.path.join(path_prediction_features, image)
            mc_predictions = np.load(os.path.join(current_image_path,'predictions.npy'))
            predictions = np.squeeze(mc_predictions)
            image_uncertainty = compute_entropy(predictions,patch_level=False)
            print(image_uncertainty)
            image_table = generate_progression_table_dynamic_lr(image,
                                                                epochs,
                                                                int(epochs* percentage_corrections),
                                                                image_uncertainty)
            score_tables[n] = image_table
        score_table_mean = np.mean(score_tables, 0)

        df = pd.DataFrame(score_table_mean,
                        index=['VGG-16', 'SVM Pass 1', 'SVM Pass 2', 'SVM Pass 3'],
                        columns=['Accuracy','Precision', 'Recall', 'F1 Score'])

        mean_tables[i] = score_table_mean

        tables_directory = os.path.join(path_metric_tables, training_condition)
        if not os.path.exists(tables_directory):
            os.makedirs(tables_directory)

        table_name = 'metric_table_{}lr_160max_linear_higherLR.csv'.format(epochs)
        df.to_csv(os.path.join(tables_directory, table_name))
    return mean_tables

if __name__ == '__main__':
    main('800init_MC_lr')
