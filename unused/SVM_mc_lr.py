from config import *
from iteration_correction import *
from uncertainty_metrics import *


val_set =  ['test_016', 'test_068', 'test_021', 'test_075', 'test_113', 'test_094', 'test_082', 'test_105', 'test_073', 'test_071', 'test_121', 'test_064', 'test_065', 'test_069', 'test_051', 'test_008', 'test_033', 'test_030', 'test_048', 'test_052', 'test_046', 'test_099', 'test_038', 'test_013'] 
image_list = val_set
epochs = 800
percentage_corrections = 0.50

score_tables = np.zeros((len(image_list),4,4))
for n, image in tqdm(enumerate(image_list)):

    current_image_path = os.path.join(path_prediction_features, image)
    mc_predictions = np.load(os.path.join(current_image_path,'predictions.npy'))
    predictions = np.squeeze(mc_predictions)
    image_uncertainty = compute_entropy(predictions,patch_level=False)
    image_table = generate_progression_table_dynamic_lr(image,
                                                        epochs,
                                                        int(epochs* percentage_corrections),
                                                        image_uncertainty)
    score_tables[n] = image_table
score_table_mean = np.mean(score_tables, 0)

df = pd.DataFrame(score_table_mean,
                index=['VGG-16', 'SVM Pass 1', 'SVM Pass 2', 'SVM Pass 3'],
                columns=['Accuracy','Precision', 'Recall', 'F1 Score'])

mean_tables = score_table_mean

tables_directory = os.path.join(path_metric_tables, '800init_MC_lr')
if not os.path.exists(tables_directory):
    os.makedirs(tables_directory)

table_name = 'metric_table_{}lr_160max_linear_higherLR.csv'.format(epochs)
df.to_csv(os.path.join(tables_directory, table_name))