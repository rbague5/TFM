import time, datetime
import pandas as pd
import numpy as np
import tensorflow
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Concatenate, Dropout, Input, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

def prepare_train_data(d_TRAIN_path, d_train_path, selected_features):
    # Leemos el fichero CSV con las etiquetas
    D_TRAIN = pd.read_csv(d_TRAIN_path)
    D_train = pd.read_csv(d_train_path)
    # Actualizamos los indices de cada particion
    TRAIN_data = D_TRAIN.reset_index(drop=True)
    train_data = D_train.reset_index(drop=True)
    # Convertimos NaN a 0
    TRAIN_data = TRAIN_data.fillna(0)
    train_data = train_data.fillna(0)
    # La descripción de los restaurantes se almacena en 'train_X' y la clase en 'train_Y'
    x_train = train_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_train = train_data.iloc[:, -1:]  # la última columa es la clase
    x_train_details = x_train.drop(['userId', 'restaurantId'], axis=1)
    x_train_details = x_train_details[selected_features]

    return TRAIN_data, x_train, y_train, x_train_details


def prepare_val_data(d_val_path, selected_features):
    d_val = pd.read_csv(d_val_path)
    # Actualizamos los indices de cada particion
    val_data = d_val.reset_index(drop=True)
    # Convertimos NaN a 0
    val_data = val_data.fillna(0)
    x_val = val_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_val = val_data.iloc[:, -1:]  # la última columa es la clase
    x_val_details = x_val.drop(['userId', 'restaurantId'], axis=1)
    x_val_details = x_val_details[selected_features]

    return val_data, x_val, y_val, x_val_details


def prepare_test_data(d_test_path, selected_features):
    D_TEST = pd.read_csv(d_test_path)
    # Actualizamos los indices de cada particion
    test_data = D_TEST.reset_index(drop=True)
    test_data = test_data.fillna(0)
    x_test = test_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_test = test_data.iloc[:, -1:]  # la última columa es la clase
    x_test_details = x_test.drop(['userId', 'restaurantId'], axis=1)
    x_test_details = x_test_details[selected_features]

    return test_data, x_test, y_test, x_test_details


def create_model(num_users, num_restaurants, num_columns_details, learning_rate, emb_size, dropout):
    # Flattens the input. Does not affect the batch size.
    # Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).
    model_embeddding_restaurants = Sequential(name="Embedding restaurants")
    model_embeddding_restaurants.add(Embedding(input_dim=num_restaurants, output_dim=emb_size, input_shape=(1,), name="input_num_restaurants"))
    model_embeddding_restaurants.add(Flatten())

    model_details = Sequential(name="Details")    #  Si modificas el tamaño de embedding tienes que modificar también el tamaño de la capa Dense que utilizas para el input de las features, de lo contrario los inputs no tienen el mismo peso que era el problema que comentamos en su inicio.
    model_details.add(Dense(emb_size, input_shape=(num_columns_details,), activation="relu", name="dense_details"))
    model_details.add(Flatten())

    concatenate = Concatenate(axis=1)([model_details.output, model_embeddding_restaurants.output])
    dense = Dense(emb_size)(concatenate)
    dense = Activation("relu")(dense)
    dense = BatchNormalization()(dense)

    model_embeddding_users = Sequential(name="Embedding users")
    model_embeddding_users.add(Embedding(input_dim=num_users, output_dim=emb_size, input_shape=(1,), name="input_num_users"))
    model_embeddding_users.add(Flatten())

    concatenate = Concatenate(axis=1)([dense, model_embeddding_users.output])

    concatenate = Dense(128, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(32, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(16, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(8, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(4, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(2, kernel_regularizer=keras.regularizers.l1(0.01))(concatenate)
    concatenate = Activation("relu")(concatenate)
    concatenate = BatchNormalization()(concatenate)

    concatenate = Dense(1)(concatenate)
    concatenate_take_out = Activation("sigmoid")(concatenate)

    opt = adam_v2.Adam(learning_rate=learning_rate)

    final_model = Model(inputs=[model_details.input, model_embeddding_restaurants.input, model_embeddding_users.input],
                        outputs=concatenate_take_out)

    final_model.compile(loss="binary_crossentropy",
                        optimizer=opt,
                        metrics=[
                            tensorflow.metrics.AUC(),
                            'accuracy'
                        ])

    """ SE VISUALIZA EL MODELO """
    # Imprimir en modo texto finalmente el resumen/arquitectura de nuestro modelo
    # Esta información permite conocer el número de parámetros que se han de aprender
    # print(final_model.summary())
    # plot_model(final_model, to_file="baseline_model.png")

    return final_model


def train_model(final_model, checkpoint_path, x_train_data, y_train_data, x_train_details, x_val_data, y_val_data, x_val_details,
                batch_size, num_epochs, patience):
    # This callback will stop the training when there is no improvement in  the loss for three consecutive epochs.
    # early_stopping_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

    # checkpoint
    filepath = "./model_weights/weights.best." + model_name + ".details.hdf5"
    # Create a callback that saves the model's weights
    checkpoint_best_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='min', period=1)
    early_stopping_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    history = final_model.fit(x=[x_train_details, x_train_data['restaurantId'], x_train_data['userId']],
                              y=y_train_data,
                              validation_data=([x_val_details, x_val_data['restaurantId'], x_val_data['userId']], y_val_data),
                              # validation_steps=len(x_val_data),
                              epochs=num_epochs,
                              # steps_per_epoch=len(x_train) / batch_size,
                              batch_size=batch_size,
                              shuffle=True,
                              use_multiprocessing=True,
                              callbacks=[early_stopping_val_loss, checkpoint_best_model],
                              workers=6,
                              verbose=1)
    return final_model, history


def plot_history(history, attribute, message, model_name):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.epoch, np.array(history.history[attribute]),
             label=message)
    plt.legend()
    plt.ylim([0, max(1, max(np.array(history.history[attribute])))])
    # print("Máximo error:", max(np.array(history.history[attribute])))
    # print("Mínimo error:", min(np.array(history.history[attribute])))
    # plt.savefig('./plots/Feature Selection/' + model_name + '/' + message)
    plt.show()


def plot_dual_history(history, model_name):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    # plt.savefig('./plots/Feature Selection/' + model_name + '/' + 'Model_Losses')
    plt.show()


def evaluate_model(final_model, full_data, x_data_details, x_data, y_data):
    test_loss, test_auc, test_acc = final_model.evaluate([x_data_details, x_data['restaurantId'], x_data['userId']],
                         y_data,
                         steps=len(full_data),
                         verbose=1)
    print("test_loss: %.4f, test_auc: %.4f, test_acc: %.4f" % (test_loss, test_auc, test_acc))
    # print()
    # print(f"{test_loss:.4f}")
    # print(f"{test_auc:.4f}")


def predict_model(final_model, x_test_details, x_test_data):
    # Obtenemos las predicciones para todos los ejemplos del conjunto de test
    predictions = final_model.predict([x_test_details, x_test_data['restaurantId'], x_test_data['userId']],
                                      steps=len(x_test_data),
                                      verbose=1)
    return predictions


def save_model(path):
    # Guardar el Modelo
    final_model.save(path)


def check_predictions(test_data, pred):
    real_values = list()
    predicted_values = list()
    correct_predictions = 0
    incorrect_predictions = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for prediction, index_test_data in zip(range(0, len(pred)), range(0, len(test_data))):
        real_value = test_data.loc[index_test_data, 'like']
        # print("#################################################")
        # print(f"REAL value : Ejemplo {prediction} ==> {test_data.loc[real_value,'like']}")
        if pred[prediction] > 0.5:
            value_asigned = 1
            # print(f"PREDICTED value : Ejemplo {predictions[prediction]} ==> {value_asigned}")
        else:
            value_asigned = 0
            # print(f"PREDICTED value : Ejemplo {predictions[prediction]} ==> {value_asigned}")

        # guardamos en una lista los real values y los predicted para despues calcular el balanced accuracy
        real_values.append(real_value)
        predicted_values.append(value_asigned)

        if real_value == value_asigned and value_asigned == 1:
            TP += 1
            # print("CORRECT PREDICTION!")
            correct_predictions += 1
        elif real_value == value_asigned and value_asigned == 0:
            TN += 1
            correct_predictions += 1
        elif real_value == 1 and value_asigned == 0:
            FN += 1
            incorrect_predictions += 1
        elif real_value == 0 and value_asigned == 1:
            FP += 1
            incorrect_predictions += 1
        else:
            print("IDK")
            break

    balanced_accuracy = balanced_accuracy_score(real_values, predicted_values)

    print(f"TP = {TP}")
    print(f"TN = {TN}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"True Positive Rate (TPR) = TP/(TP+FN) = {(TP / (TP + FN)):.4f}")
    print(f"True Negative Rate (TNR) = TN/(TN+FP) = {(TN / (TN + FP)):.4f}")
    print(f"Balanced accuracy = {balanced_accuracy:.4f}")
    print(f"False Positive Rate (FPR) = FP/(FP+TN) = {(FP / (FP + TN)):.4f}")
    print(f"False Negative Rate (FNR) = FN/(FN+TP) = {(FN / (FN + TP)):.4f}")

    print(
        f"Correct predictions: {correct_predictions} / {len(pred)} = {((correct_predictions / len(pred)) * 100):.4f}")
    print(
        f"Incorrect predictions: {incorrect_predictions} / {len(pred)} = {((incorrect_predictions / len(pred)) * 100):.4f}")

    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)
    # print(f"{(TP / (TP + FN)):.4f}")
    # print(f"{(TN / (TN + FP)):.4f}")
    # print(f"{balanced_accuracy:.4f}")
    # print(f"{(FP / (FP + TN)):.4f}")
    # print(f"{(FN / (FN + TP)):.4f}")
    # 
    # print(
    #     f"{correct_predictions}/{len(pred)}={((correct_predictions / len(pred)) * 100):.4f}")
    # print(
    #     f"{incorrect_predictions}/{len(pred)}={((incorrect_predictions / len(pred)) * 100):.4f}")


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Especificamos los paths al directorio que contiene las imagenes y al fichero con las etiquetas
    data_path = 'GIJON/'
    d_info_path = data_path + 'D_INFO/D_train_small_info_gijon.csv'  # esto se tendria que cambiar para el modelo final que etrenaremos TRAIN
    d_TEST_path = data_path + 'D_TEST/D_TEST_gijon.csv'
    d_TRAIN_path = data_path + 'D_TRAIN/D_TRAIN_gijon.csv'

    d_train_path = data_path + 'D_TRAIN/D_train/D_train_gijon.csv'
    d_train_balanced_path = '../src_negative_sampling/full_balanced_d_train_gijon.csv'
    d_val_path = data_path + 'D_TRAIN/D_val/D_val_gijon.csv'


    # MIM details
    # First 10 details
    fs_10_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European']
    # First 20 details
    fs_20_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar']
    # First 30 details
    fs_30_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments']
    # First 40 details
    fs_40_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks']
    # First 50 details
    fs_50_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks', 'Brew Pub', 'Gift Cards Available', 'Television', 'Parking Available',
                 'Bar', 'Central American', 'Dining bars', 'Wine Bar', 'Thai', 'Malaysian']
    # First 60 details
    fs_60_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks', 'Brew Pub', 'Gift Cards Available', 'Television', 'Parking Available',
                 'Bar', 'Central American', 'Dining bars', 'Wine Bar', 'Thai', 'Malaysian', 'Filipino', 'Dog Friendly',
                 'Diner', 'Asian', 'Seating', 'Southern-Italian', 'Neapolitan', 'Campania', 'Seafood', 'Sicilian']
    # First 70 details
    fs_70_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks', 'Brew Pub', 'Gift Cards Available', 'Television', 'Parking Available',
                 'Bar', 'Central American', 'Dining bars', 'Wine Bar', 'Thai', 'Malaysian', 'Filipino', 'Dog Friendly',
                 'Diner', 'Asian', 'Seating', 'Southern-Italian', 'Neapolitan', 'Campania', 'Seafood', 'Sicilian',
                 'Indian', 'French', 'Chinese', 'Playgrounds', 'Spanish', 'South American', 'Family style',
                 'Street Food', 'Fast Food', 'Peruvian']
    # First 80 details
    fs_80_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks', 'Brew Pub', 'Gift Cards Available', 'Television', 'Parking Available',
                 'Bar', 'Central American', 'Dining bars', 'Wine Bar', 'Thai', 'Malaysian', 'Filipino', 'Dog Friendly',
                 'Diner', 'Asian', 'Seating', 'Southern-Italian', 'Neapolitan', 'Campania', 'Seafood', 'Sicilian',
                 'Indian', 'French', 'Chinese', 'Playgrounds', 'Spanish', 'South American', 'Family style',
                 'Street Food', 'Fast Food', 'Peruvian', 'Accepts Discover', 'Table Service', 'German',
                 'Non-smoking restaurants', 'International', 'Validated Parking', 'Japanese', 'Delivery',
                 'Middle Eastern', 'Breakfast']
    # First 90 details
    fs_90_mim = ['Gluten Free Options', 'Lunch', 'Dinner', 'Vegetarian Friendly', 'max_range', 'Highchairs Available',
                 'Vegan Options', 'min_range', 'Free Wifi', 'European', 'Private Dining', 'Mediterranean',
                 'Reservations', 'Accepts Mastercard', 'Accepts Visa', 'Takeout', 'Wheelchair Accessible',
                 'Accepts Credit Cards', 'Wine and Beer', 'Full Bar', 'Late Night', 'Barbecue', 'Outdoor Seating',
                 'Accepts American Express', 'Central European', 'Healthy', 'Fusion', 'Italian', 'Pub',
                 'Digital Payments', 'Argentinean', 'Contemporary', 'Soups', 'Pizza', 'Serves Alcohol', 'Cafe', 'Latin',
                 'Gastropub', 'Grill', 'Drinks', 'Brew Pub', 'Gift Cards Available', 'Television', 'Parking Available',
                 'Bar', 'Central American', 'Dining bars', 'Wine Bar', 'Thai', 'Malaysian', 'Filipino', 'Dog Friendly',
                 'Diner', 'Asian', 'Seating', 'Southern-Italian', 'Neapolitan', 'Campania', 'Seafood', 'Sicilian',
                 'Indian', 'French', 'Chinese', 'Playgrounds', 'Spanish', 'South American', 'Family style',
                 'Street Food', 'Fast Food', 'Peruvian', 'Accepts Discover', 'Table Service', 'German',
                 'Non-smoking restaurants', 'International', 'Validated Parking', 'Japanese', 'Delivery',
                 'Middle Eastern', 'Breakfast', 'Mexican', 'Free off-street parking', 'Street Parking', 'Waterfront',
                 'American', 'Buffet', 'Steakhouse', 'Sushi', 'Brunch', 'Deli']


    fs_10_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night']
    # First 20 details
    fs_20_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa']
    # First 30 details
    fs_30_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking']
    # First 40 details
    fs_40_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly']
    # First 50 details
    fs_50_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                     'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue']
    # First 60 details
    fs_60_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                     'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue',
                     'Free off-street parking', 'Pub', 'Argentinean', 'Grill', 'Family style', 'Soups', 'Latin',
                     'Central American', 'Fast Food', 'Indian']
    # First 70 details
    fs_70_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                     'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue',
                     'Free off-street parking', 'Pub', 'Argentinean', 'Grill', 'Family style', 'Soups', 'Latin',
                     'Central American', 'Fast Food', 'Indian', 'Wine Bar', 'Japanese', 'Sushi', 'Diner',
                     'Middle Eastern', 'South American', 'Southern-Italian', 'Neapolitan', 'Campania', 'Sicilian']
    # First 80 details
    fs_80_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                     'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue',
                     'Free off-street parking', 'Pub', 'Argentinean', 'Grill', 'Family style', 'Soups', 'Latin',
                     'Central American', 'Fast Food', 'Indian', 'Wine Bar', 'Japanese', 'Sushi', 'Diner',
                     'Middle Eastern', 'South American', 'Southern-Italian', 'Neapolitan', 'Campania', 'Sicilian',
                     'Thai', 'Malaysian', 'Filipino', 'Dog Friendly', 'International', 'Waterfront',
                     'Non-smoking restaurants', 'Beach', 'Dining bars', 'Jazz Bar']
    # First 90 details
    fs_90_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                     'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                     'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                     'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                     'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                     'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                     'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                     'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue',
                     'Free off-street parking', 'Pub', 'Argentinean', 'Grill', 'Family style', 'Soups', 'Latin',
                     'Central American', 'Fast Food', 'Indian', 'Wine Bar', 'Japanese', 'Sushi', 'Diner',
                     'Middle Eastern', 'South American', 'Southern-Italian', 'Neapolitan', 'Campania', 'Sicilian',
                     'Thai', 'Malaysian', 'Filipino', 'Dog Friendly', 'International', 'Waterfront',
                     'Non-smoking restaurants', 'Beach', 'Dining bars', 'Jazz Bar', 'Accepts Discover', 'Brew Pub',
                     'French', 'Buffet', 'German', 'Peruvian', 'Street Food', 'max_range', 'Playgrounds', 'Chinese']
    # All reliefF details (93)
    fs_93_reliefF = ['Wheelchair Accessible', 'Mediterranean', 'Takeout', 'Wine and Beer', 'Outdoor Seating',
                      'Television', 'European', 'Drinks', 'Free Wifi', 'Late Night', 'Highchairs Available', 'Spanish',
                      'Accepts Credit Cards', 'Seafood', 'Full Bar', 'Vegan Options', 'Accepts American Express',
                      'Digital Payments', 'Reservations', 'Accepts Visa', 'Breakfast', 'Healthy', 'Table Service',
                      'Steakhouse', 'Contemporary', 'Street Parking', 'min_range', 'Fusion', 'Accepts Mastercard',
                      'Validated Parking', 'Pizza', 'Gift Cards Available', 'Parking Available', 'Delivery', 'Italian',
                      'Brunch', 'Gluten Free Options', 'Seating', 'Mexican', 'Vegetarian Friendly', 'Asian', 'Cafe',
                      'Private Dining', 'Lunch', 'Serves Alcohol', 'Dinner', 'Bar', 'American', 'Cash Only', 'Barbecue',
                      'Free off-street parking', 'Pub', 'Argentinean', 'Grill', 'Family style', 'Soups', 'Latin',
                      'Central American', 'Fast Food', 'Indian', 'Wine Bar', 'Japanese', 'Sushi', 'Diner',
                      'Middle Eastern', 'South American', 'Southern-Italian', 'Neapolitan', 'Campania', 'Sicilian',
                      'Thai', 'Malaysian', 'Filipino', 'Dog Friendly', 'International', 'Waterfront',
                      'Non-smoking restaurants', 'Beach', 'Dining bars', 'Jazz Bar', 'Accepts Discover', 'Brew Pub',
                      'French', 'Buffet', 'German', 'Peruvian', 'Street Food', 'max_range', 'Playgrounds', 'Chinese',
                      'Deli', 'Central European', 'Gastropub']

    ### Hyperparameters for training ###
    # Batch_zie
    # Define EMBEDDING SIZE
    EMB_SIZE = 64
    # BATCH SIZE
    BATCH_SIZE = 16
    # Define learning rate.
    LEARNING_RATE = 0.0001
    # This is the number of epochs (passes over the full training data)
    NUM_EPOCHS = 100
    # DROPOUT
    DROPOUT = 0.8
    #PATIENCE
    PATIENCE = 5

    tf.random.set_seed(100)
    auc_val = 0

    fs_mim_list = [fs_10_mim, fs_20_mim, fs_30_mim, fs_40_mim, fs_50_mim, fs_60_mim, fs_70_mim, fs_80_mim, fs_90_mim]
    for lis in fs_mim_list:
        features_to_select = lis
        # NUM FEATURES
        NUM_FEATURES = len(features_to_select)

        model_name = "fs_" + str(NUM_FEATURES) + "_mim"

        checkpoint_path = "./training_feature_selection/" + str(NUM_FEATURES) + "-{epoch:04d}.ckpt"

        print(f"Selected details: {len(features_to_select)} -> {features_to_select}")

        D_TRAIN, x_train, y_train, x_train_details = prepare_train_data(d_TRAIN_path=d_TRAIN_path,
                                                                        d_train_path=d_train_balanced_path,
                                                                        selected_features=features_to_select)
        # print("x_train_details")
        # print(x_train_details)
        val_data, x_val, y_val, x_val_details = prepare_val_data(d_val_path=d_val_path,
                                                                        selected_features=features_to_select)
        # print("x_val_details")
        # print(x_val_details)
        test_data, x_test, y_test, x_test_details = prepare_test_data(d_test_path=d_TEST_path,
                                                                        selected_features=features_to_select)
        # print("x_test_details")
        # print(x_test_details)

        d_info = pd.read_csv(d_info_path)

        # Size of the vocabulary, i.e. maximum integer index + 1.
        # https://keras.io/api/layers/core_layers/embedding/
        num_users = int(sorted(D_TRAIN['userId'].unique())[-1]) + 1
        num_restaurants = int(sorted(D_TRAIN['restaurantId'].unique())[-1]) + 1
        number_columns_details = len(x_train_details.columns)

        final_model = create_model(num_users=num_users, num_restaurants=num_restaurants,
                                   num_columns_details=number_columns_details, learning_rate=LEARNING_RATE,
                                   emb_size=EMB_SIZE, dropout=DROPOUT)

        # starting time
        start = time.time()

        final_model, history = train_model(final_model=final_model, checkpoint_path=checkpoint_path,x_train_data=x_train,
                                           y_train_data=y_train, x_train_details=x_train_details, x_val_data=x_val,
                                           y_val_data=y_val, x_val_details=x_val_details, batch_size=BATCH_SIZE,
                                           num_epochs=NUM_EPOCHS, patience=PATIENCE)

        # end time
        end = time.time()

        time_train_model = datetime.timedelta(seconds=(end - start))
        # total time taken


        plot_history(history, attribute='loss', message='Loss in training', model_name=model_name)
        plot_history(history, attribute='val_loss', message='Validation Loss in training', model_name=model_name)
        if auc_val == 0:
            plot_history(history, attribute='auc', message='AUC', model_name=model_name)
        elif auc_val != 0:
            plot_history(history, attribute='auc_'+ str(auc_val), message='AUC', model_name=model_name)

        plot_dual_history(history, model_name=model_name)


        final_model.load_weights("./model_weights/weights.best." + model_name + ".details.hdf5")

        print(f"EMB_SIZE: {EMB_SIZE}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        print(f"LEARNING_RATE: {LEARNING_RATE}")

        print("Predecimos datos con  mejor modelo en conjunto de test")
        predictions = predict_model(final_model=final_model, x_test_details=x_test_details, x_test_data=x_test)

        print(f"Results: {len(lis)}--> {lis}")
        check_predictions(test_data=test_data, pred=predictions)

        print("Evaluamos mejor modelo en conjunto de test")
        evaluate_model(final_model=final_model, full_data=test_data, x_data_details=x_test_details, x_data=x_test,
                       y_data=y_test)
        # Evaluar el modelo en el conjunto de val
        print("Evaluamos mejor modelo en conjunto de validación")
        evaluate_model(final_model=final_model, full_data=val_data, x_data_details=x_val_details, x_data=x_val,
                       y_data=y_val)

        print(f"Runtime of training neural network is {time_train_model}")
        auc_val += 1

        import winsound
        duration = 500  # milliseconds
        freq = 220  # Hz
        winsound.Beep(freq, duration)

    fs_reliefF_list = [fs_10_reliefF, fs_20_reliefF, fs_30_reliefF, fs_40_reliefF, fs_50_reliefF, fs_60_reliefF,
                       fs_70_reliefF, fs_80_reliefF, fs_90_reliefF, fs_93_reliefF]
    for lis in fs_reliefF_list:
        features_to_select = lis
        # NUM FEATURES
        NUM_FEATURES = len(features_to_select)

        model_name = "fs_" + str(NUM_FEATURES) + "_reliefF"

        checkpoint_path = "./training_feature_selection/" + str(NUM_FEATURES) + "-{epoch:04d}.ckpt"

        print(f"Selected details: {len(features_to_select)} -> {features_to_select}")

        D_TRAIN, x_train, y_train, x_train_details = prepare_train_data(d_TRAIN_path=d_TRAIN_path,
                                                                        d_train_path=d_train_balanced_path,
                                                                        selected_features=features_to_select)

        val_data, x_val, y_val, x_val_details = prepare_val_data(d_val_path=d_val_path,
                                                                 selected_features=features_to_select)

        test_data, x_test, y_test, x_test_details = prepare_test_data(d_test_path=d_TEST_path,
                                                                      selected_features=features_to_select)

        d_info = pd.read_csv(d_info_path)

        # Size of the vocabulary, i.e. maximum integer index + 1.
        # https://keras.io/api/layers/core_layers/embedding/
        num_users = int(sorted(D_TRAIN['userId'].unique())[-1]) + 1
        num_restaurants = int(sorted(D_TRAIN['restaurantId'].unique())[-1]) + 1
        number_columns_details = len(x_train_details.columns)

        final_model = create_model(num_users=num_users, num_restaurants=num_restaurants,
                                   num_columns_details=number_columns_details, learning_rate=LEARNING_RATE,
                                   emb_size=EMB_SIZE, dropout=DROPOUT)

        # starting time
        start = time.time()

        final_model, history = train_model(final_model=final_model, checkpoint_path=checkpoint_path,
                                           x_train_data=x_train,
                                           y_train_data=y_train, x_train_details=x_train_details, x_val_data=x_val,
                                           y_val_data=y_val, x_val_details=x_val_details, batch_size=BATCH_SIZE,
                                           num_epochs=NUM_EPOCHS, patience=PATIENCE)

        # end time
        end = time.time()

        time_train_model = datetime.timedelta(seconds=(end - start))
        # total time taken

        plot_history(history, attribute='loss', message='Loss in training', model_name=model_name)
        plot_history(history, attribute='val_loss', message='Validation Loss in training', model_name=model_name)
        if auc_val == 0:
            plot_history(history, attribute='auc', message='AUC', model_name=model_name)
        elif auc_val != 0:
            plot_history(history, attribute='auc_' + str(auc_val), message='AUC', model_name=model_name)

        plot_dual_history(history, model_name=model_name)


        final_model.load_weights("./model_weights/weights.best." + model_name + ".details.hdf5")

        print(f"EMB_SIZE: {EMB_SIZE}")
        print(f"BATCH_SIZE: {BATCH_SIZE}")
        print(f"LEARNING_RATE: {LEARNING_RATE}")

        print("Predecimos datos con  mejor modelo en conjunto de test")
        predictions = predict_model(final_model=final_model, x_test_details=x_test_details, x_test_data=x_test)

        print(f"Results: {len(lis)}--> {lis}")
        check_predictions(test_data=test_data, pred=predictions)

        print("Evaluamos mejor modelo en conjunto de test")
        evaluate_model(final_model=final_model, full_data=test_data, x_data_details=x_test_details, x_data=x_test,
                       y_data=y_test)
        # Evaluar el modelo en el conjunto de val
        print("Evaluamos mejor modelo en conjunto de validación")
        evaluate_model(final_model=final_model, full_data=val_data, x_data_details=x_val_details, x_data=x_val,
                       y_data=y_val)

        print(f"Runtime of training neural network is {time_train_model}")
        auc_val += 1




