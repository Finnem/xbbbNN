import numpy as np
from xbpy import rdutil

def get_features(used_data, filter):
    return used_data[[col for col in used_data if filter(col)]]
    
def get_labels_with_weight_and_ideal_approx(data, primary_column, secondary_column, approximation, approximation_weighting):
    labels = []
    positive_counts = np.sum(data[primary_column].astype(bool) & (~data[secondary_column].astype(bool)))
    negative_counts = np.sum(~data[primary_column].astype(bool))

    positive_weight = (positive_counts + negative_counts) / positive_counts
    negative_weight = (positive_counts + negative_counts) / negative_counts
    to_predict = np.zeros_like(data[primary_column]).astype(float)
    weight = np.full_like(data[primary_column], negative_weight).astype(float)
    primary_filter = (data[primary_column] == 1) & (data[secondary_column] == 0)
    to_predict[primary_filter] = data["label"][primary_filter]
    weight[primary_filter] = positive_weight
    secondary_filter = (data[primary_column] == 1) & (data[secondary_column] == 1)
    weight[secondary_filter] = positive_weight * approximation_weighting
    to_predict[secondary_filter] = data[approximation][secondary_filter]
    """
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
    """
    return np.array([to_predict, weight]).T
    return labels




