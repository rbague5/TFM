import pandas as pd
from skrebate import ReliefF

# The output of the Relief algorithm is a weight between −1 and 1 for each attribute, with more positive weights
# indicating more predictive attributes. ... This procedure of updating the weight of the attribute is performed
# for a random set of samples in the data or for every sample in the data.

def reliefF(datadict, data, target):
    # print(data)
    # print(target)
    # print(datadict.columns)

    clf = ReliefF()
    clf.fit(data, target)

    for feature_score, feature_name in sorted(zip(clf.feature_importances_, datadict.columns), reverse=True):
        print(feature_score, feature_name)


if __name__ == "__main__":
    CITY = 'gijon'

    # Procesar los pickles de cada ciudad para ver informacion de particiones
    path = './train_test/D_TRAIN/D_TRAIN_' + CITY + '.csv'

    with open(path, 'rb') as f:
        datadict = pd.read_csv(f)

    features = datadict.loc[:, datadict.columns != 'like']
    features = features.loc[:, features.columns != 'rating']

    # features = features.fillna(0)
    # print(features)
    # print(type(features))
    # sirve para tener las columnas para luego mostrarlas
    features_df = features

    for i, row in features.iterrows():
        if isinstance(row['min_range'], str):
            string_num = row['min_range']
            float_num = string_num.replace(',', '')
            features.at[i, 'min_range'] = float_num

        if isinstance(row['max_range'], str):
            string_num = row['max_range']
            float_num = string_num.replace(',', '')
            features.at[i, 'max_range'] = float_num

    features["min_range"] = features["min_range"].astype(float)
    features["max_range"] = features["max_range"].astype(float)

    features = features.to_numpy()
    target = datadict[datadict.columns[-1]]

    reliefF(features_df, features, target)
