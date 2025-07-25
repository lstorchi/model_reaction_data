import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
#from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, train_test_split)

# if needed TF
#from tensorflow import keras
#import tensorflow as tf
#import keras.optimizers as tko
#import keras.activations as tka
#import keras.losses as tkl
#from keras.layers import Input, Dense
#from keras.models import Model

from sklearn import preprocessing

import numpy as np

#from ann_visualizer.visualize import ann_viz

import commonutils

SPLIT_RANDOM_STATE = 42
SHOWPLOTS = False
DEBUG = False

####################################################################################################

def pls_model (Xin, Yin, supersetlist, setlist, \
            ncomp_start = 1, ncomp_max = 15, \
            normlize = False, split = True, perc_split = 0.2, \
            plot = False, loo = True):

    X = None
    Y = None

    if normlize:
        scalerX = preprocessing.StandardScaler().fit(Xin)
        X = scalerX.transform(Xin)

        Y = Yin
    else:
        X = Xin
        Y = Yin

    X_train = X
    X_test = X
    y_train = Y
    y_test = Y
        
    if split:
        X_train, X_test, y_train, y_test, supersetlist_train, \
            supersetlist_test, setlist_train, setlist_test \
                = train_test_split(X, Y, supersetlist, setlist, \
                                test_size=perc_split, random_state=SPLIT_RANDOM_STATE)
    
    ncomps = []

    r2s = []
    rmses = []
    mapes = []
    wtmads = []
    loormses = []
    
    for ncomp in range(ncomp_start, ncomp_max+1):
        
        if loo:
            cv = LeaveOneOut()
            model = PLSRegression(ncomp)
            scores = cross_val_score(model, X_train, y_train, \
                scoring='neg_mean_squared_error', \
                cv=cv, n_jobs=-1)
            loormse = np.sqrt(np.mean(np.absolute(scores)))
            loormses.append(loormse)

        pls = PLSRegression(ncomp)
        pls.fit(X_train, y_train)
    
        y_pred = pls.predict(X_train)
        y_pred_test = pls.predict(X_test)
        
        pred = pls.predict(X)
        if len(pred.shape) == 2:
            pred = pred[:,0]
        #print(setlist)
        wtmadf = commonutils.wtmad2(setlist, list(Y), list(pred))
        wtmad_value = None
        if len(set(supersetlist)) == 1:
            ssetname = list(set(supersetlist))[0]
            wtmad_value = wtmadf[ssetname]
        else:
            wtmad_value = wtmadf["Full"]

        rmse = 0.0
        try:
            rmse = root_mean_squared_error(Y, pred)
        except:
            rmse = mean_squared_error(Y, pred, squared=False)
        r2 = r2_score(Y, pred)
        mape = mean_absolute_percentage_error(Y, pred)
    
        r2s.append(r2)
        rmses.append(rmse)
        mapes.append(mape)
        ncomps.append(ncomp)
        wtmads.append(wtmad_value)

    if plot:
        # plot the rmses
        plt.clf()
        #plt.rcParams.update({'font.size': 15})
        plt.plot(ncomps, rmses, '-o', color='black', label='RMSE')
        if loo:
            plt.plot(ncomps, loormses, '-o', color='red', label='LOORMSE')
        plt.plot(ncomps, wtmads, '-o', color='blue', label='WTMAD')
        plt.xticks(ncomps)
        plt.xlabel('Number of Components')
        plt.ylabel('RMSE')
        plt.legend()
        #plt.savefig("PLS_components_RMSE.png", bbox_inches="tight", dpi=600)
        plt.show()

    return ncomps, rmses, r2s, wtmads, loormses, mapes

####################################################################################################

# needed TF keras
#def nn_model(perc_split, X, scalex, Y, scaley, supersetlist, setlist, \
#             nepochs, modelshapes, batch_sizes, inputshape=-1,\
#             search=True, split=True):
#
#    loss_metric = "mse"
#    X_train = X
#    X_test = X
#    y_train = Y
#    y_test = Y
#
#    if split:
#        X_train, X_test, y_train, y_test = train_test_split(X, Y, \
#            test_size=perc_split, random_state=SPLIT_RANDOM_STATE)
#    
#    if inputshape == -1:
#        inputshape = X_train.shape[1]
#
#    mapes = []
#    rmses = []
#    wtmads = []
#    models = []
#    modeidxs = []
#
#    if search:
#        midx = 0
#        maxidx = len(modelshapes)*len(nepochs)*len(batch_sizes)
#
#        for modelshape in modelshapes:
#            for nepoch in nepochs:
#                for nbatch_size in batch_sizes:
#                    model = keras.Sequential()
#                
#                    model.add(keras.layers.Input(shape=(inputshape,)))
#                
#                    for n in modelshape:
#                        model.add(keras.layers.Dense(units = n, \
#                                                     activation = 'relu'))
#                
#                    model.add(keras.layers.Dense(units = 1, \
#                                                 activation = 'linear'))
#                    model.compile(loss=loss_metric, optimizer="adam", \
#                                  metrics=['mse', 'mape'])
#                    
#                    model.fit(X_train, y_train, epochs=nepoch,  \
#                            batch_size=nbatch_size, \
#                            verbose=0)
#                    
#                    Y_rescaled = scaley.inverse_transform(Y)
#                    y_pred = model.predict(X, verbose=0)
#                    y_pred_rescaled = scaley.inverse_transform(y_pred)
#                    if len(y_pred_rescaled.shape) == 2:
#                        y_pred_rescaled = y_pred_rescaled[:,0]
#                    if len(Y_rescaled.shape) == 2:
#                        Y_rescaled = Y_rescaled[:,0]
#                    wtmad = commonutils.wtmad2(setlist, Y_rescaled, y_pred_rescaled)
#                    wtmad = wtmad["Full"]
#                    mape = mean_absolute_percentage_error(Y, y_pred_rescaled)
#                    rmse = root_mean_squared_error(Y_rescaled        , y_pred_rescaled)
#
#                    mapes.append(mape)
#                    wtmads.append(wtmad)
#                    rmses.append(rmse)
#
#                    models.append((modelshape, nepoch, nbatch_size))
#                    modeidxs.append(midx)
#                    midx += 1
#
#                    commonutils.printProgressBar(midx, maxidx, \
#                                    prefix = 'Progress:', \
#                                    suffix = 'Complete', length = 50)
#                    
#        index_min_mape = np.argmin(mapes)
#        index_min_wtmad = np.argmin(wtmads)
#        index_min_rmse = np.argmin(rmses)
#        
#        avg = np.average(mapes)
#        print("Average MAPE: ", avg)
#        std = np.std(mapes)
#        print("    STD MAPE: ", std)
#        avg = np.average(rmses)
#        print("Average RMSE: ", avg)
#        std = np.std(rmses)
#        print("    STD RMSE: ", std)
#        #plot hitogram
#        plt.clf()
#        plt.hist(mapes, bins=50)
#        plt.show()
#        plt.clf()
#        plt.hist(rmses, bins=50)
#        plt.show()
#        plt.clf()
#        plt.hist(wtmads, bins=50)
#        plt.show()
#
#
#        return models[index_min_mape], models[index_min_wtmad], models[index_min_rmse]
#    else:
#        modelshape = modelshapes[0]
#        nepoch = nepochs[0]
#        batch_size = batch_sizes[0]
# 
#        model = keras.Sequential()
#        model.add(keras.layers.Input(shape=(inputshape,)))
#        
#        for n in modelshape:
#            model.add(keras.layers.Dense(units = n, activation = 'relu'))
#        
#        model.add(keras.layers.Dense(units = 1, activation = 'linear'))
#        model.compile(loss=loss_metric, optimizer="adam", metrics=['mse', 'mape'])
#        """
#        cannot easily define this as yje training loss at the end of 
#        each epoch is the mean of the batch losses.
#        lossmetric == "wtmad":
#            def wtmad_loss(scaley, setlist, supersetlist):
#                def loss(y_true, y_pred):
#                    y_pred_rescaled = scaley.inverse_transform(y_pred)
#                    y_true_rescaled = scaley.inverse_transform(y_true)
#                    wtamad = commonutils.wtmad_calc(supersetlist, setlist, \
#                                                y_pred_rescaled, y_true_rescaled, \
#                                                includeFull = True)
#                    wtmad = wtamad["Full"]
#
#                    return 
#                return loss
#            model.compile(loss=wtmad_loss(scaley, setlist, supersetlist), \
#                          optimizer="adam", metrics=['mse', 'mape'])
#        """
#        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
#                                    test_size=perc_split, random_state=SPLIT_RANDOM_STATE)
#    
#        history = model.fit(X_train, y_train, \
#                            epochs=nepoch,  \
#                            batch_size=batch_size, \
#                            validation_data=(X_val, y_val),
#                            verbose=0)
#        
#        y_pred_train = model.predict(X_train)
#        y_pred_valid = model.predict(X_val)
#        y_pred_test = model.predict(X_test)
#        y_pred = model.predict(X)
#
#        results = {
#            "model" : model,
#            "history" : history,
#            "y_pred_train" : y_pred_train,
#            "y_train": y_train,
#            "y_pred_valid" : y_pred_valid,
#            "y_valid" : y_val,
#            "y_pred_test" : y_pred_test,
#            "y_test" : y_test,
#            "y_pred_full" : y_pred,
#            "y_full" : Y
#        }
#
#        return results
#            
#    return None

####################################################################################################

def rf_model (perc_split, X, Y, search = True, in_n_estimators = [50, 100, 300, 400],
              in_max_depth = [None, 5, 8, 15, 25, 30], 
              in_min_samples_split = [2, 5, 10, 15], 
              in_min_samples_leaf = [10, 20, 50], 
              in_random_state = [42], 
              in_max_features = [1, 3, 5, 8, 9, 10], 
              in_bootstrap = [True] ):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                    test_size=perc_split, random_state=SPLIT_RANDOM_STATE)
    
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
        maxidx = len(hyperF["n_estimators"])* \
                    len(hyperF["max_depth"])* \
                    len(hyperF["min_samples_split"])* \
                    len(hyperF["min_samples_leaf"])* \
                    len(hyperF["random_state"])* \
                    len(hyperF["bootstrap"])* \
                    len(hyperF["max_features"])
        
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

                                train_rmse = root_mean_squared_error(y_train, y_pred)
                                test_rmse = root_mean_squared_error(y_test, y_pred_test)
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

                                commonutils.printProgressBar(idx, maxidx, \
                                                 prefix = 'Progress:', \
                                                    suffix = 'Complete', length = 50)

                                #print(idx / maxidx * 100, "%")

        if DEBUG:
            print("min_train_rmse_hyper: ", min_train_rmse_hyper)
            print("min_test_rmse_hyper: ", min_test_rmse_hyper)
            print("max_train_r2_hyper: ", max_train_r2_hyper)
            print("max_test_r2_hyper: ", max_test_r2_hyper)


        if SHOWPLOTS:    
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
        
        train_rmse = root_mean_squared_error(y_train, y_pred)
        test_rmse = root_mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)
      
        if DEBUG:
            print("train_rmse: ", train_rmse)
            print("test_rmse: ", test_rmse)
            print("r2_train: ", r2_train)
            print("r2_test: ", r2_test)

        if SHOWPLOTS:    
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

        return train_rmse, test_rmse, r2_train, r2_test, model, \
              X_train, X_test, y_train, y_test

    return 

####################################################################################################

