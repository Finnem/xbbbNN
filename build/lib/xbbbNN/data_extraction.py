import numpy as np

def get_features(used_data, filter):
    return used_data[[col for col in used_data if filter(col)]]
    
def get_labels_with_weight_and_ideal_approx(data, primary_column, secondary_column, approximation, approximation_weighting):
    labels = []
    positive_counts = np.sum(data[primary_column].astype(bool) & (~data[secondary_column].astype(bool)))
    negative_counts = np.sum(~data[primary_column].astype(bool))

    positive_weight = (positive_counts + negative_counts) / positive_counts
    negative_weight = (positive_counts + negative_counts) / negative_counts
    def set_output_vector(row):
        energy_to_predict = 0.0
        if row[primary_column] == 1:
            weight = positive_weight
            if row[secondary_column] == 1:
                energy_to_predict = row[approximation]
                weight *= approximation_weighting
            else:
                energy_to_predict = row["label"]
        else:
            weight = negative_weight

        labels.append([energy_to_predict, weight])
    data.apply(set_output_vector, axis = 1)
    labels = np.array(labels)
    return labels