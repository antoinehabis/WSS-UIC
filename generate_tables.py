from config import *
from uncertainty_metrics import *
from iteration_correction import *


def main(folder, use_mc=True):
    test_val_set = os.listdir(path_prediction_features)

    if folder == "test":
        image_list = test_set
    if folder == "val":
        image_list img= val_set
    ###
    epochs_range = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_tables = 1
    tables = np.zeros((n_tables, len(epochs_range), 5, 4))

    for i in range(n_tables):
        for j, epochs in enumerate(epochs_range):
            score_tables = np.zeros((len(image_list), 5, 4))

            for n, image in tqdm(enumerate(image_list)):
                if use_mc:
                    current_image_path = os.path.join(path_prediction_features, image)
                    mc_predictions = np.load(
                        os.path.join(current_image_path, "predictions.npy")
                    )
                    predictions = np.squeeze(mc_predictions)
                    MAX_ENTROPY = 0.4
                    # MIN_ENTROPY = 0.09
                    corr_effect = (
                        compute_entropy(predictions, patch_level=False) / MAX_ENTROPY
                    )
                    corr_epochs = np.clip((corr_effect * epochs), 1, None).astype(int)
                else:
                    corr_epochs = epochs
                image_table = generate_progression_table(image, 1000, corr_epochs)
                score_tables[n] = image_table
            tables[i, j] = np.mean(score_tables, 0)
    table_means = np.mean(tables, axis=0)
    table_stds = np.std(tables, axis=0)

    for j, (table_mean, table_std) in enumerate(zip(table_means, table_stds)):
        if use_mc:
            sufix = "_mc"
        else:
            sufix = "_no_mc"

        df_mean = pd.DataFrame(
            table_mean,
            index=["VGG-16", "SVM Pass 1", "SVM Pass 2", "SVM Pass 3", "SVM Pass 4"],
            columns=["Accuracy", "Precision", "Recall", "F1 Score"],
        )

        df_std = pd.DataFrame(
            table_std,
            index=["VGG-16", "SVM Pass 1", "SVM Pass 2", "SVM Pass 3", "SVM Pass 4"],
            columns=["Accuracy", "Precision", "Recall", "F1 Score"],
        )

        mean_table_directory = os.path.join(path_metric_tables, folder + sufix, "mean")
        std_table_directory = os.path.join(path_metric_tables, folder + sufix, "std")

        if not os.path.exists(mean_table_directory):
            os.makedirs(mean_table_directory)
        if not os.path.exists(std_table_directory):
            os.makedirs(std_table_directory)

        table_name = "metric_table_{}.csv".format(str(epochs_range[j]))
        df_mean.to_csv(os.path.join(mean_table_directory, table_name))
        df_std.to_csv(os.path.join(std_table_directory, table_name))

    return 0


if __name__ == "__main__":
    main("val", False)
