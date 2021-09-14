import random
import pandas as pd


CITY = 'gijon'
train_big_path = './' + CITY + '/D_TRAIN/D_TRAIN_' + CITY + '.csv'
train_small_path = './' + CITY + '/D_TRAIN/D_train/D_train_' + CITY + '.csv'
val_path = './' + CITY + '/D_TRAIN/D_val/D_val_' + CITY + '.csv'
test_path = './' + CITY + '/D_TEST/D_TEST_' + CITY + '.csv'
details_path = './' + CITY + '/' + CITY + '_details_binary.csv'

# Leemos el fichero CSV con las etiquetas
D_TRAIN = pd.read_csv(train_big_path)
D_train = pd.read_csv(train_small_path)
D_test = pd.read_csv(test_path)
d_val = pd.read_csv(val_path)

d_details = pd.read_csv(details_path)
d_details = D_TRAIN.drop(columns=['userId', 'like'])

# Actualizamos los indices de cada particion
TRAIN_data = D_TRAIN.reset_index(drop=True)
train_data = D_train.reset_index(drop=True)
val_data = d_val.reset_index(drop=True)
test_data = D_test.reset_index(drop=True)
d_details = d_details.reset_index(drop=True)
#borramos los id de restaurant q sean iguales pq son todos la misma info
d_details = d_details.drop_duplicates(subset=['restaurantId'], keep='first')

list_users = train_data['userId'].unique().tolist()
list_restaurants = train_data['restaurantId'].unique().tolist()

dislike_value = 0

print(f"Number of total reviews in train data: {len(train_data.index)}")
df_likes_user = train_data[train_data['like'] == 1]
print(f"Number of POSITIVE reviews in train data: {len(df_likes_user.index)}")
df_dislikes_user = train_data[train_data['like'] == 0]
print(f"Number of NEGATIVE reviews in train data: {len(df_dislikes_user.index)}")

for user_id in list_users:
    print("########################################################################")
    print(f"USER: {user_id}")
    df_user = train_data[train_data['userId'] == user_id]

    # split dataframe of user in two different dataframes of like/dislike
    df_likes_user = df_user[df_user['like'] == 1]
    num_likes_user = len(df_likes_user.index)
    # print(f"LIKES: {num_likes_user}")
    list_user_restaurants_that_likes = df_likes_user['restaurantId'].unique().tolist()

    df_dislikes_user = df_user[df_user['like'] == 0]
    num_dislikes_user = len(df_dislikes_user.index)
    # print(f"DISLIKES: {num_dislikes_user}")

    if num_dislikes_user < num_likes_user:
        negative_samples_to_generate = num_likes_user - num_dislikes_user
        restaurants_to_select = list(set(list_restaurants) - set(list_user_restaurants_that_likes))
        selected_restaurants_negative_sampling = random.sample(restaurants_to_select, negative_samples_to_generate)
        df_of_selected_genative_samples = df_user[df_user['restaurantId'].isin(d_details)]

        details_restaurants_neg_samples = d_details[
            d_details['restaurantId'].isin(selected_restaurants_negative_sampling)]
        details_restaurants_neg_samples[['userId', 'like']] = [user_id, dislike_value]

        train_data = train_data.append(details_restaurants_neg_samples)
        df_user = train_data[train_data['userId'] == user_id]
        # print(df_user)
    else:
        continue

    # comentar hasta avajo para que no esté balanceado
    # cuando el numero de exemplos negativos sea igual o superior al de positivos paramos para q este all balanceado
    # train_df_likes = train_data[train_data['like'] == 1]
    # num_likes_train = len(train_df_likes.index)
    #
    # train_df_dislikes = train_data[train_data['like'] == 0]
    # num_dislikes_train = len(train_df_dislikes.index)
    #
    # if num_dislikes_train >= num_likes_train:
    #     break

train_data.to_csv('full_balanced_d_train_gijon.csv', index=False)