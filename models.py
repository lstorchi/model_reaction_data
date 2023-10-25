import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, train_test_split)


from tensorflow import keras
import tensorflow as tf

import tensorflow.keras.optimizers as tko
import tensorflow.keras.activations as tka
import tensorflow.keras.losses as tkl
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from ann_visualizer.visualize import ann_viz


####################################################################################################

def nndmodel(perc_split, X, Y, nepochs, modelshapes, batch_size=-1, inputshape=-1):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=42)

    if inputshape == -1:
        inputshape = X_train.shape[1]

    if batch_size == -1:
        batch_size = int(X_train.shape[0])

    mses_test = []
    mses_train = []
    r2s_test = []
    r2s_train = []
    modeidxs = []

    for midx, modelshape in enumerate(modelshapes):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(inputshape)))

        for n in modelshape:
            model.add(keras.layers.Dense(units = n, activation = 'relu'))

        model.add(keras.layers.Dense(units = 1, activation = 'linear'))
        model.compile(loss='mse', optimizer="adam", metrics='mse')
        #ann_viz(model, title="Discriminator Model",\
        #         view=True)
        
        history = model.fit(X_train, y_train, epochs=nepochs,  batch_size=batch_size, \
            verbose=0)

        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)

        r2s_train.append(r2_train)
        mses_train.append(mse_train)
        r2s_test.append(r2_test)
        mses_test.append(mse_test)

        modeidxs.append(midx)


    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(modeidxs, mses_test, '-o', color='black')
    plt.plot(modeidxs, mses_train, '-o', color='red')
    plt.xlabel('Model Index')
    plt.ylabel('MSE')
    plt.xticks(modeidxs)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(modeidxs, r2s_test, '-o', color='black')
    plt.plot(modeidxs, r2s_train, '-o', color='red')
    plt.xlabel('Model Index')
    plt.ylabel('R2')
    plt.xticks(modeidxs)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

    return

####################################################################################################

def plsemodel (perc_split, X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=42)

    mses_test = []
    mses_train = []
    r2s_test = []
    r2s_train = []
    ncomps = []

    for ncomp in range(1,15):
        pls = PLSRegression(ncomp)
        pls.fit(X_train, y_train)

        y_pred = pls.predict(X_train)
        y_pred_test = pls.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)

        r2s_train.append(r2_train)
        mses_train.append(mse_train)
        r2s_test.append(r2_test)
        mses_test.append(mse_test)
        ncomps.append(ncomp)

    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(ncomps, mses_test, '-o', color='black')
    plt.plot(ncomps, mses_train, '-o', color='red')
    plt.xlabel('Number of Components')
    plt.ylabel('MSE')
    plt.xticks(ncomps)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

    plt.clf()
    plt.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    plt.plot(ncomps, r2s_test, '-o', color='black')
    plt.plot(ncomps, r2s_train, '-o', color='red')
    plt.xlabel('Number of Components')
    plt.ylabel('R2')
    plt.xticks(ncomps)
    #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    plt.show()

    return

####################################################################################################
