from __future__ import print_function
import pandas as pd
import time
from collections import Counter

def analizar_datos(city, path):
    col_names = ['num_users', 'num_restaurants', 'avg_reviews_user', 'avg_reviews_restaurant', 'num_likes', 'num_dislikes']
    df_info_city = pd.DataFrame(columns=col_names, index=[0])

    print(f"Checking city: {city}")

    with open(path, 'rb') as f:
        datadict = pd.read_csv(f)
        print('Type datadict = ', type(datadict))
        print('Reviews Keys:')

        del datadict['rating']
        print(datadict.keys())

        # cambiamos el valor de las columnas para q no sea tipo string
        for i, row in datadict.iterrows():
            if isinstance(row['min_range'], str):
                string_num = row['min_range']
                float_num = string_num.replace(',', '')
                datadict.at[i, 'min_range'] = float_num

            if isinstance(row['max_range'], str):
                string_num = row['max_range']
                float_num = string_num.replace(',', '')
                datadict.at[i, 'max_range'] = float_num
        #
        datadict["min_range"] = datadict["min_range"].astype(float)
        datadict["max_range"] = datadict["max_range"].astype(float)

        restaurantId = datadict['restaurantId'].tolist()
        userId = datadict['userId'].tolist()

        # reviewId = datadict['reviewId']
        num_userId = datadict['userId'].nunique()
        num_reviews = len(datadict.index)
        user_ids = datadict['userId'].unique()
        num_restaurantId = datadict['restaurantId'].nunique()

        print(f"Num of different user id: {num_userId}")
        print(f"Num of different restaurant id: {num_restaurantId}")
        print(f"Num of reviews: {num_reviews}")

        # count id of users toknow how many times they write a comment
        counter_users = Counter(userId)
        total_users = sum(counter_users.values())

        # total reviews / total users nos da la relacion de reviews x usuario
        average_review_for_user = total_users / num_userId
        print(f"Average of reviews / user = {average_review_for_user}")

        # count id of restaurants to know how many reviews each restaurant has
        counter_restaurants = Counter(restaurantId)
        total_restaurants = sum(counter_restaurants.values())
        # total reviews / total users nos da la relacion de reviews x usuario
        average_review_for_restaurant = total_restaurants / num_restaurantId
        print(f"Average of reviews / restaurant = {average_review_for_restaurant}")
        print(user_ids)

        df_likes = datadict[datadict['like'] == 1]
        num_likes = len(df_likes.index)

        df_dislikes = datadict[datadict['like'] == 0]
        num_dislikes = len(df_dislikes.index)

        df_info_city['num_users'] = num_userId
        df_info_city['num_restaurants'] = num_restaurantId
        df_info_city['avg_reviews_user'] = average_review_for_user
        df_info_city['avg_reviews_restaurant'] = average_review_for_restaurant
        df_info_city['num_likes'] = num_likes
        df_info_city['num_dislikes'] = num_dislikes
        print(df_global_info_city)

        time.sleep(500)
        return datadict, user_ids, df_info_city


def check_restaurant_in_test_and_not_in_train(d_train, d_test):
    rest_id_train = d_train['restaurantId'].tolist()
    rest_id_test = d_test['restaurantId'].tolist()

    for indx, row in d_test.iterrows():
        if row['restaurantId'] not in rest_id_train:
            print(f"{row['restaurantId']} --> This should go to train dataframe!")
            d_train = d_train.append(row)
            d_test = d_test.drop(index=indx)

    return d_train, d_test


def check_df(df):

    column_names = ['num_users', 'num_restaurants', 'num_reviews', 'num_likes', 'num_dislikes']
    info_df = pd.DataFrame(columns=column_names, index=[0])

    num_users = df['userId'].nunique()
    num_restaurants = df['restaurantId'].nunique()
    num_reviews = len(df.index)

    df_likes = df[df['like'] == 1]
    num_likes = len(df_likes.index)

    df_dislikes = df[df['like'] == 0]
    num_dislikes = len(df_dislikes.index)

    info_df['num_users'] = num_users
    info_df['num_restaurants'] = num_restaurants
    info_df['num_reviews'] = num_reviews
    info_df['num_likes'] = num_likes
    info_df['num_dislikes'] = num_dislikes

    return info_df


if __name__ == "__main__":

    train_big_path = 'train_test/D_TRAIN/'
    train_small_path = 'train_test/D_TRAIN/D_train/'
    val_path = 'train_test/D_TRAIN/D_val/'
    test_path = 'train_test/D_TEST/'

    CITY = 'gijon'

    # Procesar los pickles de cada ciudad para ver informacion de particiones
    path = './train_test/' + CITY + '_details_union_user_review.csv'

    df_city, user_ids, df_global_info_city = analizar_datos(CITY, path)


    col_names = df_city.columns

    df_train_full = pd.DataFrame(columns=col_names)
    df_TEST = pd.DataFrame(columns=col_names)

    # iteramos por cada usuario del dataset
    for user in user_ids:
        real_user_id = user - 1

        # el usuario comienza en 1 pero tendria q ser 0
        print(f"Checking user: {real_user_id}")

        reviews_of_user = df_city[df_city['userId'] == real_user_id]
        # print(reviews_of_user)

        # print("#### ORIGINAL DF ####")
        # print(reviews_of_user)

        # split dataframe of user in two different dataframes of like/dislike
        df_likes_user = reviews_of_user[reviews_of_user['like'] == 1]
        df_dislikes_user = reviews_of_user[reviews_of_user['like'] == 0]

        # # print(df_likes_user)
        # print(len(df_likes_user.index))
        # # print(df_dislikes_user)
        # print(len(df_dislikes_user.index))

        # Si solamente hay una review positiva para ese usuario, va a train.
        if len(df_likes_user.index) == 1:
            df_train_full = df_train_full.append(df_likes_user)
            # print("#### PART 1 ####")
            # print(df_likes_user)

        # Si hay mas de una review positiva para ese usuario, todas a train y 1 a test
        if len(df_likes_user.index) > 1:
            first_likes = df_likes_user.iloc[:-1]
            df_train_full = df_train_full.append(first_likes)
            last_like = df_likes_user.iloc[-1]
            df_TEST = df_TEST.append(last_like)
            # print("#### PART 2 ####")
            # print(first_likes)
            # print("#### PART LAST ####")
            # print(last_like)

        # Si solamente hay una review negativa para ese usuario, va a train
        if len(df_dislikes_user.index) == 1:
            df_train_full = df_train_full.append(df_dislikes_user)
            # print("#### PART 3 ####")
            # print(df_dislikes_user)
            # Si hay mas de una review negativa para ese usuario, todas a train y 1 a test
        if len(df_dislikes_user.index) > 1:
            first_dislikes = df_dislikes_user.iloc[:-1]
            df_train_full = df_train_full.append(first_dislikes)
            last_dislike = df_dislikes_user.iloc[-1]
            df_TEST = df_TEST.append(last_dislike)
            # print("#### PART 4 ####")
            # print(first_dislikes)
            # print("#### PART LAST ####")
            # print(last_dislike)

    print(df_train_full)
    print(df_TEST)

    df_train_full, df_TEST = check_restaurant_in_test_and_not_in_train(d_train=df_train_full, d_test=df_TEST)

    print(df_train_full)
    print(df_TEST)

    # iteramos sobre los ids de d_TRAIN para hacer las particiones train/val
    user_ids = df_train_full['userId'].unique()

    df_train_small = pd.DataFrame(columns=col_names)
    df_val = pd.DataFrame(columns=col_names)

    # Dividir conjunto D_TRAIN en D_train y D_val
    # iteramos por cada usuario del dataset de train
    for user in user_ids:

        print(f"Checking user: {user}")

        reviews_of_user = df_train_full[df_train_full['userId'] == user]
        # print(reviews_of_user)

        # print("#### ORIGINAL DF ####")
        # print(reviews_of_user)

        # split dataframe of user in two different dataframes of like/dislike
        df_likes_user = reviews_of_user[reviews_of_user['like'] == 1]
        df_dislikes_user = reviews_of_user[reviews_of_user['like'] == 0]

        # print(df_likes_user)
        # print(len(df_likes_user.index))
        # print(df_dislikes_user)
        # print(len(df_dislikes_user.index))

        # Si solamente hay una review positiva para ese usuario, va a train.
        if len(df_likes_user.index) == 1:
            df_train_small = df_train_small.append(df_likes_user)
            # print("#### PART 1 ####")
            # print(df_likes_user)

        # Si hay mas de una review positiva para ese usuario, todas a train y 1 a test
        if len(df_likes_user.index) > 1:
            first_likes = df_likes_user.iloc[:-1]
            df_train_small = df_train_small.append(first_likes)
            last_like = df_likes_user.iloc[-1]
            df_val = df_val.append(last_like)
            # print("#### PART 2 ####")
            # print(first_likes)
            # print("#### PART LAST ####")
            # print(last_like)

        # Si solamente hay una review negativa para ese usuario, va a train
        if len(df_dislikes_user.index) == 1:
            df_train_small = df_train_small.append(df_dislikes_user)
            # print("#### PART 3 ####")
            # print(df_dislikes_user)
            # Si hay mas de una review negativa para ese usuario, todas a train y 1 a test
        if len(df_dislikes_user.index) > 1:
            first_dislikes = df_dislikes_user.iloc[:-1]
            df_train_small = df_train_small.append(first_dislikes)
            last_dislike = df_dislikes_user.iloc[-1]
            df_val = df_val.append(last_dislike)
            # print("#### PART 4 ####")
            # print(first_dislikes)
            # print("#### PART LAST ####")
            # print(last_dislike)
    #
    # print(df_train_small)
    df_train_full.to_csv(train_big_path + 'D_TRAIN_' + CITY + '.csv', index=False)
    df_train_small.to_csv(train_small_path + 'D_train_' + CITY + '.csv', index=False)
    df_val.to_csv(val_path + 'D_val_' + CITY + '.csv', index=False)
    df_TEST.to_csv(test_path + 'D_TEST_' + CITY + '.csv', index=False)


    # df_global_info_city.to_csv('train_test/D_INFO/D_INFO_CITY_' + CITY + '.csv', index=False)

    # número de usuarios, restaurantes, y reviews en cada partición D_TRAIN y D_TEST (+ D_train y D_val).
    info_df_TRAIN = check_df(df_train_full)
    print("info_df_TRAIN")
    print(info_df_TRAIN)
    #info_df_TRAIN.to_csv('train_test/D_INFO/D_TRAIN_info_' + CITY + '.csv', index=False)

    info_df_TEST = check_df(df_TEST)
    print("info_df_TEST")
    print(info_df_TEST)
    # info_df_TEST.to_csv('train_test/D_INFO/D_TEST_info_' + CITY + '.csv', index=False)

    info_df_train = check_df(df_train_small)
    print("info_df_train")
    print(info_df_train)
    # info_df_train.to_csv('train_test/D_INFO/D_train_small_info_' + CITY + '.csv', index=False)

    info_df_val = check_df(df_val)
    print("info_df_val")
    print(info_df_val)
    # info_df_val.to_csv('train_test/D_INFO/D_val_info_' + CITY + '.csv', index=False)









