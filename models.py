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

def nn_model(perc_split, X, Y, nepochs, modelshapes, batch_size=-1, inputshape=-1):

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

def pls_model (perc_split, X, Y, search = True, ncomp_start = 1, ncomp_max = 15):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=42)
    
    if search == True:

        rmses_test = []
        rmses_train = []
        r2s_test = []
        r2s_train = []
        ncomps = []
    
        for ncomp in range(ncomp_start, ncomp_max):
            pls = PLSRegression(ncomp)
            pls.fit(X_train, y_train)
    
            y_pred = pls.predict(X_train)
            y_pred_test = pls.predict(X_test)
    
            rmse_train = mean_squared_error(y_train, y_pred, squared=False)
            rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    
            r2_train = r2_score(y_train, y_pred)
            r2_test = r2_score(y_test, y_pred_test)
    
            r2s_train.append(r2_train)
            rmses_train.append(rmse_train)
            r2s_test.append(r2_test)
            rmses_test.append(rmse_test)
            ncomps.append(ncomp)
    
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        #pyplot.plot(ncomps, r2s, '-o', color='black')
        plt.plot(ncomps, rmses_test, '-o', color='black')
        plt.plot(ncomps, rmses_train, '-o', color='red')
        plt.xlabel('Number of Components')
        plt.ylabel('RMS')
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

    else:

        pls = PLSRegression(ncomp_start)
        pls.fit(X_train, y_train)
    
        y_pred = pls.predict(X_train)
        y_pred_test = pls.predict(X_test)

        plt.clf()
        plt.rcParams.update({'font.size': 15})
        plt.plot(y_pred, y_train, 'o', color='red')
        plt.xlabel('PREDICTED')
        plt.ylabel('TRUE')
        plt.show()
    
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        plt.plot(y_pred_test, y_test, 'o', color='black')
        plt.xlabel('PREDICTED')
        plt.ylabel('TRUE')
        plt.show()

    return

####################################################################################################

def rf_model (perc_split, X, Y, search = True, in_n_estimators = [50, 100, 300, 500, 800, 1200],
              in_max_depth = [None, 5, 8, 15, 25, 30], 
              in_min_samples_split = [2, 5, 10, 15, 100], 
              in_min_samples_leaf = [10, 20, 50, 100, 200], 
              in_random_state = [42], 
              in_max_features = [1, 3, 5, 8, 9, 10, 100], 
              in_bootstrap = [True] ):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=42)
    
    if search == True:

        rmses_test = []
        rmses_train = []
        r2s_test = []
        r2s_train = []
        idxs = []

        hyperF = {"n_estimators" : in_n_estimators, 
            "max_depth" : in_max_depth, 
            "min_samples_split" : in_min_samples_split, 
            "min_samples_leaf" : in_min_samples_leaf, 
            "random_state" : in_random_state, 
            "bootstrap" : in_bootstrap,
            "max_features" : in_max_features}

        idx = 1
        min_train_rmse = 10000000000
        min_test_rmse = 10000000000
        max_train_r2 = -10000000000
        max_test_r2 = -10000000000
        min_train_rmse_hyper = {}
        min_test_rmse_hyper = {}
        max_train_r2_hyper = {}
        max_test_r2_hyper = {}
        for a in hyperF["n_estimators"]:
          for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                model = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )
                                model.fit(X_train, y_train)
    
                                y_pred = model.predict(X_train)
                                y_pred_test = model.predict(X_test)

                                train_rmse = mean_squared_error(y_train, y_pred, \
                                                               squared=False)
                                test_rmse = mean_squared_error(y_test, y_pred_test, \
                                                              squared=False)
                                rmses_train.append(train_rmse)
                                rmses_test.append(test_rmse)

                                if train_rmse < min_train_rmse:
                                    min_train_rmse = train_rmse
                                    min_train_rmse_hyper = {
                                        "n_estimators" : a, 
                                        "max_depth" : b, 
                                        "min_samples_split" : c, 
                                        "min_samples_leaf" : d, 
                                        "random_state" : e, 
                                        "bootstrap" : f,
                                        "max_features" : g}

                                if test_rmse < min_test_rmse:
                                    min_test_rmse = test_rmse
                                    min_test_rmse_hyper = {
                                        "n_estimators" : a, 
                                        "max_depth" : b, 
                                        "min_samples_split" : c, 
                                        "min_samples_leaf" : d, 
                                        "random_state" : e, 
                                        "bootstrap" : f,
                                        "max_features" : g
                                    }

                                r2_train = r2_score(y_train, y_pred)
                                r2_test = r2_score(y_test, y_pred_test)
                                r2s_train.append(r2_train)
                                r2s_test.append(r2_test)

                                if r2_train > max_train_r2:
                                    max_train_r2 = r2_train
                                    max_train_r2_hyper = {
                                        "n_estimators" : a, 
                                        "max_depth" : b, 
                                        "min_samples_split" : c, 
                                        "min_samples_leaf" : d, 
                                        "random_state" : e, 
                                        "bootstrap" : f,
                                        "max_features" : g
                                    }

                                if r2_test > max_test_r2:
                                    max_test_r2 = r2_test
                                    max_test_r2_hyper = {
                                        "n_estimators" : a, 
                                        "max_depth" : b, 
                                        "min_samples_split" : c, 
                                        "min_samples_leaf" : d, 
                                        "random_state" : e, 
                                        "bootstrap" : f,
                                        "max_features" : g
                                    }                                        

                                idxs.append(idx)
                                idx += 1

        print("min_train_rmse_hyper: ", min_train_rmse_hyper)
        print("min_test_rmse_hyper: ", min_test_rmse_hyper)
        print("max_train_r2_hyper: ", max_train_r2_hyper)
        print("max_test_r2_hyper: ", max_test_r2_hyper)

    
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        #pyplot.plot(ncomps, r2s, '-o', color='black')
        plt.plot(idxs, rmses_test, 'o', color='black')
        plt.plot(idxs, rmses_train, 'o', color='red')
        plt.xlabel('Index')
        plt.ylabel('RMSE')
        plt.xticks(idxs)
        #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
        plt.show()
    
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        #pyplot.plot(ncomps, r2s, '-o', color='black')
        plt.plot(idxs, r2s_test, 'o', color='black')
        plt.plot(idxs, r2s_train, 'o', color='red')
        plt.xlabel('Index')
        plt.ylabel('R2')
        plt.xticks(idxs)
        #plt.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
        plt.show()

        return min_train_rmse_hyper, min_test_rmse_hyper, max_train_r2_hyper, max_test_r2_hyper

    else:
        if len(in_n_estimators) > 1 or \
           len(in_max_depth) > 1 or \
           len(in_min_samples_split) > 1 or \
           len(in_min_samples_leaf) > 1 or \
           len(in_random_state) > 1 or \
           len(in_bootstrap) > 1 or \
           len(in_max_features) > 1:
          print("ERROR: Only one hyperparameter can be used for prediction.")
          return
      
        model = RandomForestRegressor(
                      n_estimators=in_n_estimators[0],
                      max_depth=in_max_depth[0],
                      min_samples_split=in_min_samples_split[0],
                      min_samples_leaf=in_min_samples_leaf[0],
                      random_state=in_random_state[0],
                      bootstrap=in_bootstrap[0],
                      max_features=in_max_features[0]
                  )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = mean_squared_error(y_train, y_pred, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        r2_train = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)
      
        print("train_rmse: ", train_rmse)
        print("test_rmse: ", test_rmse)
        print("r2_train: ", r2_train)
        print("r2_test: ", r2_test)
      
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        plt.plot(y_pred, y_train, 'o', color='red')
        plt.xlabel('PREDICTED')
        plt.ylabel('TRUE')
        plt.show()
      
        plt.clf()
        plt.rcParams.update({'font.size': 15})
        plt.plot(y_pred_test, y_test, 'o', color='black')
        plt.xlabel('PREDICTED')
        plt.ylabel('TRUE')
        plt.show()

    return 

####################################################################################################

