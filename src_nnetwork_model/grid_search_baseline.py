from kerashypetune import KerasGridSearch
import pandas as pd
import tensorflow
from keras.optimizers import adam_v2
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Concatenate,BatchNormalization
from tensorflow import keras
import tensorflow as tf


def prepare_train_data(d_TRAIN_path, d_train_path):
    # Leemos el fichero CSV con las etiquetas
    D_TRAIN = pd.read_csv(d_TRAIN_path)
    D_train = pd.read_csv(d_train_path)
    # Actualizamos los indices de cada particion
    TRAIN_data = D_TRAIN.reset_index(drop=True)
    train_data = D_train.reset_index(drop=True)
    # La descripción de los restaurantes se almacena en 'train_X' y la clase en 'train_Y'
    x_train = train_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_train = train_data.iloc[:, -1:]  # la última columa es la clase
    x_train_details = x_train.drop(['userId', 'restaurantId'], axis=1)
    print(f"train data: {train_data}")
    return TRAIN_data, x_train, y_train, x_train_details


def prepare_val_data(d_val_path):
    d_val = pd.read_csv(d_val_path)
    # Actualizamos los indices de cada particion
    val_data = d_val.reset_index(drop=True)
    x_val = val_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_val = val_data.iloc[:, -1:]  # la última columa es la clase
    x_val_details = x_val.drop(['userId', 'restaurantId'], axis=1)
    return val_data, x_val, y_val, x_val_details


def prepare_test_data(d_test_path):
    D_TEST = pd.read_csv(d_test_path)
    # Actualizamos los indices de cada particion
    test_data = D_TEST.reset_index(drop=True)
    x_test = test_data.iloc[:, :-1]  # todas las columnas salvo las última
    y_test = test_data.iloc[:, -1:]  # la última columa es la clase
    x_test_details = x_test.drop(['userId', 'restaurantId'], axis=1)
    return test_data, x_test, y_test, x_test_details


def create_model(param):
    model_embeddding_users = Sequential(name="Embedding users")
    model_embeddding_users.add(Embedding(input_dim=param['num_users'], output_dim=param['emb_size'], input_shape=(1,),
                                             name="input_num_users"))
    model_embeddding_users.add(Flatten())
    # Flattens the input. Does not affect the batch size.
    # Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).
    model_embeddding_restaurants = Sequential(name="Embedding restaurants")
    model_embeddding_restaurants.add(Embedding(input_dim=param['num_restaurants'], output_dim=param['emb_size'], input_shape=(1,),
                                               name="input_num_restaurants"))
    model_embeddding_restaurants.add(Flatten())

    concatenate = Concatenate(axis=1)([model_embeddding_users.output, model_embeddding_restaurants.output])

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

    opt = adam_v2.Adam(learning_rate=param['lr'])

    final_model = Model(inputs=[model_embeddding_users.input, model_embeddding_restaurants.input],
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


if __name__ == '__main__':

    global num_users
    global num_restaurants
    global number_columns_details

    # Especificamos los paths al directorio que contiene las imagenes y al fichero con las etiquetas
    data_path = 'GIJON/'
    d_info_path = data_path + 'D_INFO/D_train_small_info_gijon.csv'  # esto se tendria que cambiar para el modelo final que etrenaremos TRAIN
    d_TEST_path = data_path + 'D_TEST/D_TEST_gijon.csv'
    d_TRAIN_path = data_path + 'D_TRAIN/D_TRAIN_gijon.csv'

    d_train_path = data_path + 'D_TRAIN/D_train/D_train_gijon.csv'
    d_train_balanced_path = '../src_negative_sampling/full_balanced_d_train_gijon.csv'
    d_val_path = data_path + 'D_TRAIN/D_val/D_val_gijon.csv'

    D_TRAIN, x_train, y_train, x_train_details = prepare_train_data(d_TRAIN_path=d_TRAIN_path,
                                                                    d_train_path=d_train_balanced_path)
    val_data, x_val, y_val, x_val_details = prepare_val_data(d_val_path=d_val_path)
    test_data, x_test, y_test, x_test_details = prepare_test_data(d_test_path=d_TEST_path)

    d_info = pd.read_csv(d_info_path)

    # Size of the vocabulary, i.e. maximum integer index + 1.
    # https://keras.io/api/layers/core_layers/embedding/
    num_users = int(sorted(D_TRAIN['userId'].unique())[-1]) + 1
    num_restaurants = int(sorted(D_TRAIN['restaurantId'].unique())[-1]) + 1
    number_columns_details = len(x_train_details.columns)

    tf.random.set_seed(100)

    # define the grid search parameters
    param_grid = {
        'num_users': num_users,
        'num_restaurants': num_restaurants,
        'lr': [.001, .0001, .00001],
        'batch_size': [16, 32, 64],
        'epochs': 100,
        'emb_size': [64, 128, 256]
    }
    early_stopping_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    kgs = KerasGridSearch(create_model, param_grid,
                          monitor='val_loss', greater_is_better=False, tuner_verbose=1)
    kgs.search(x=[x_train['userId'], x_train['restaurantId']],
               y=y_train,
               validation_data=([x_val['userId'], x_val['restaurantId']], y_val),
               callbacks=[early_stopping_val_loss])

    print(kgs.scores)
    print(kgs.best_score)
    print(kgs.best_params)
    print(kgs.best_model)
