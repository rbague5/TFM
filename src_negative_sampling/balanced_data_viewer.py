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
print(f"Number of POSITIVE reviews in train data: {len(df_likes_user.index)} ({len(df_likes_user.index)/len(train_data.index)*100:.2f}\%)")
df_dislikes_user = train_data[train_data['like'] == 0]
print(f"Number of NEGATIVE reviews in train data: {len(df_dislikes_user.index)} ({len(df_dislikes_user.index)/len(train_data.index)*100:.2f}\%)")
print()
print(f"Number of total reviews in val data: {len(val_data.index)}")
df_likes_user = val_data[val_data['like'] == 1]
print(f"Number of POSITIVE reviews in val data: {len(df_likes_user.index)} ({len(df_likes_user.index)/len(val_data.index)*100:.2f}\%)")
df_dislikes_user = val_data[val_data['like'] == 0]
print(f"Number of NEGATIVE reviews in val data: {len(df_dislikes_user.index)} ({len(df_dislikes_user.index)/len(val_data.index)*100:.2f}\%)")
print()
print(f"Number of total reviews in test data: {len(test_data.index)}")
df_likes_user = test_data[test_data['like'] == 1]
print(f"Number of POSITIVE reviews in test data: {len(df_likes_user.index)} ({len(df_likes_user.index)/len(test_data.index)*100:.2f}\%)")
df_dislikes_user = test_data[test_data['like'] == 0]
print(f"Number of NEGATIVE reviews in test data: {len(df_dislikes_user.index)} ({len(df_dislikes_user.index)/len(test_data.index)*100:.2f}\%)")
print()
print("############# NOW WITH NEGATIVE SAMPLING #############")
full_balanced = pd.read_csv("full_balanced_d_train_gijon.csv")
print(f"Number of total reviews in train data: {len(full_balanced.index)}")
df_likes_user = full_balanced[full_balanced['like'] == 1]
print(f"Number of POSITIVE reviews in train data: {len(df_likes_user.index)} ({len(df_likes_user.index)/len(full_balanced.index)*100:.2f}\%)")
df_dislikes_user = full_balanced[full_balanced['like'] == 0]
print(f"Number of NEGATIVE reviews in train data: {len(df_dislikes_user.index)} ({len(df_dislikes_user.index)/len(full_balanced.index)*100:.2f}\%)")
