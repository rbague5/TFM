import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer


def mutual_info_classifier(features_df, features, target):

    print(features)
    features_to_print = []
    import time
    start_time = time.time()
    feature_scores = mutual_info_classif(features, target, discrete_features=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    for score, f_name in sorted(zip(feature_scores, features_df.columns), reverse=True):
        print(f_name, score)
        features_to_print.append(f_name)

        # print("\hline")
        # print(f"{f_name} & {score:.4f} xxx" )

    print("# First 10 details")
    print(f"fs_10_mim={features_to_print[2:12]}")

    print("# First 20 details")
    print(f"fs_20_mim={features_to_print[2:22]}")

    print("# First 30 details")
    print(f"fs_30_mim={features_to_print[2:32]}")

    print("# First 40 details")
    print(f"fs_40_mim={features_to_print[2:42]}")

    print("# First 50 details")
    print(f"fs_50_mim={features_to_print[2:52]}")

    print("# First 60 details")
    print(f"fs_60_mim={features_to_print[2:62]}")

    print("# First 70 details")
    print(f"fs_70_mim={features_to_print[2:72]}")

    print("# First 80 details")
    print(f"fs_80_mim={features_to_print[2:82]}")

    print("# First 90 details")
    print(f"fs_90_mim={features_to_print[2:92]}")


if __name__ == "__main__":

    CITY = 'gijon'

    # Procesar los pickles de cada ciudad para ver informacion de particiones
    path = './train_test/D_TRAIN/D_TRAIN_' + CITY + '.csv'

    with open(path, 'rb') as f:
        datadict = pd.read_csv(f)

    features = datadict.loc[:, datadict.columns != 'like']

    features = features.fillna(0)
    print(features)
    print(type(features))
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
    #
    features["min_range"] = features["min_range"].astype(float)
    features["max_range"] = features["max_range"].astype(float)

    # features["min_range"].to_numeric("min_range")
    # features["min_range"].to_numeric("max_range")
    print(features)

    # discretizamos los rangos de precios
    # min_range_dis = KBinsDiscretizer(n_bins=3, encode='ordinal',
    #                           strategy="kmeans").fit_transform(features[['min_range']])
    # min_range_dis = pd.DataFrame(min_range_dis)
    # min_range_dis = min_range_dis.rename(columns={0: 'min_range'})
    # features[['min_range']] = min_range_dis
    #
    # max_range_dis = KBinsDiscretizer(n_bins=3, encode='ordinal',
    #                                  strategy="kmeans").fit_transform(features[['max_range']])
    # max_range_dis = pd.DataFrame(max_range_dis)
    # max_range_dis = max_range_dis.rename(columns={0: 'max_range'})
    # features[['max_range']] = max_range_dis

    print(features)

    features = features.to_numpy()

    target = datadict[datadict.columns[-1]]

    mutual_info_classifier(features_df, features, target)