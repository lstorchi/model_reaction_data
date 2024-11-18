import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import commonutils
import models

from commonutils import ModelResults

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataclasses import dataclass
import prettyprinter as pp

from sklearn.cross_decomposition import PLSRegression
import warnings
import sys

from sklearn import preprocessing

from copy import deepcopy
import pickle

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import sys
sys.path.append("./CLossLr")
import customlosslr as clr

from commonutils import ModelsStore

#warnings.simplefilter("ignore")

from commonfuncsforcli import *

if __name__ == "__main__":

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print("Usage: python3 runmodels.py " + \
              "<selected_functional> <selected_basisset> " + \
                "[<functionals> <basis_sets>]")
        sys.exit(1)

    featuresvalues_perset,\
        fullsetnames, \
        models_results, \
        supersetnames = readdata()

    #["PBE", "PBE0"]
    #["MINIX", "SVP", "TZVP", "QZVP"]

    selected_basisset = ""
    selected_functional = ""
    functionals = []
    basis_sets = []

    selected_basisset = sys.argv[2].strip()
    selected_functional = sys.argv[1].strip()
    if len(sys.argv) == 5:
        functionals = sys.argv[3].split(",")
        basis_sets = sys.argv[4].split(",")

    print("Selected functional: ", selected_functional)
    print("Selected basis set: ", selected_basisset)
    print("Functionals: ", functionals)
    print("Basis sets: ", basis_sets)
    
    sep = "_"
    for setname in fullsetnames:
        desciptors = {}
        k = selected_functional + sep +\
                    selected_basisset 
        for features in featuresvalues_perset[setname]:
            for val in features:
                if val.find(k) != -1:
                    if val not in desciptors:
                        desciptors[val] = [features[val]]
                    else:
                        desciptors[val].append(features[val])
    
        for features in featuresvalues_perset[setname]:
            for val in features:
                for func in functionals:
                    for basis in basis_sets:
                        if not(basis == selected_basisset and \
                                func == selected_functional):
                            if val.find(func + sep + basis) != -1:
                                actualk = val 
                                refk  = selected_functional + sep  + \
                                        selected_basisset + \
                                        val.replace(func + sep + basis, "")
                                newk = actualk + "_difftoref"
                                if newk not in desciptors:
                                    desciptors[newk] = [features[actualk]-features[refk]]
                                else:
                                    desciptors[newk].append(features[actualk]-features[refk])
        
        models_results[setname].features = desciptors
    
    # feastures selection
    setname = "Full"
    numoffeat = len(models_results[setname].features)
    print("Number of features for ", numoffeat)
    for setname in fullsetnames:
        if len(models_results[setname].features) != numoffeat:
            print("Number of features for ", setname, " is different")
            sys.exit(1)
    
    toremove = []
    setname = "Full"
    for k in models_results[setname].features:
        if len(set(models_results[setname].features[k])) == 1:
            toremove.append(k)
            print("Constant fatures to remove: ", k)
    
    # remove constant values
    for setname in fullsetnames:
        for k in toremove:
            del models_results[setname].features[k]
    
    # force removing features Nuclear Repulsion difference
    tormfeatname = set()
    print("Removing Nuclear Repulsion differences")
    for setname in fullsetnames: 
        toremove = []
        for k in models_results[setname].features:
            if k.find("NR_difftoref") != -1:
                toremove.append(k)
        for k in toremove:
            #print("Removing feature ", k)
            tormfeatname.add(k)
            del models_results[setname].features[k]
    
    print("Removed features: ", tormfeatname)
    setname = "Full"
    numoffeat = len(models_results[setname].features)
    print("Number of features ", numoffeat)
    for setname in fullsetnames:
        if len(models_results[setname].features) != numoffeat:
            print("Number of features for ", setname, " is different")
            sys.exit(1)
    
    
    models_store = {}
    fp = open("modelsgeneral.csv", "w")
    for setname in list(supersetnames)+["Full"]:
        models_store[setname] = ModelsStore()
    
        print("Running models for dataset: ", setname)
        X, Y, features_names = \
                commonutils.build_XY_matrix \
                (models_results[setname].features, \
                models_results[setname].labels)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                          test_size=0.20, random_state=42)
        setlist = models_results[setname].setnames  
        supersetlist = models_results[setname].supersetnames
        maxcomp = X.shape[1]
    
        # Linear Regression
        models_store[setname].lr_model = \
                clr.custom_loss_lr (loss=clr.mean_average_error)
        models_store[setname].lr_model.fit(X, Y)
        y_pred_lr = models_store[setname].lr_model.predict(X)
        lrrmse = root_mean_squared_error(Y, y_pred_lr)
        lrrmape = mean_absolute_percentage_error(Y, y_pred_lr)
        models_store[setname].lr_model_splitted = \
                clr.custom_loss_lr (loss=clr.mean_average_error)
        models_store[setname].lr_model_splitted.fit(X_train, y_train)
        y_pred_lr = models_store[setname].lr_model_splitted.predict(X_test)
        lrrmsetest = root_mean_squared_error(y_test, y_pred_lr)
        lrrmaoetest = mean_absolute_percentage_error(y_test, y_pred_lr)
        y_pred_lr = models_store[setname].lr_model_splitted.predict(X_train)
        lrrmsetrain = root_mean_squared_error(y_train, y_pred_lr)
        lrrmaopetrain = mean_absolute_percentage_error(y_train, y_pred_lr)
        if PRINTALSOINSTDOUT:
            print("%40s ,             LR RMSE, %10.2f"%(setname,lrrmse))
            print("%40s ,       LR Train RMSE, %10.2f"%(setname,lrrmsetrain))
            print("%40s ,        LR Test RMSE, %10.2f"%(setname,lrrmsetest))
            print("%40s ,             LR MAPE, %10.2f"%(setname,lrrmape))
            print("%40s ,       LR Train MAPE, %10.2f"%(setname,lrrmaopetrain))
            print("%40s ,        LR Test MAPE, %10.2f"%(setname,lrrmaoetest))
    
        print("%40s ,             LR RMSE, %10.2f"%(setname,lrrmse), file=fp)
        print("%40s ,       LR Train RMSE, %10.2f"%(setname,lrrmsetrain), file=fp)
        print("%40s ,        LR Test RMSE, %10.2f"%(setname,lrrmsetest), file=fp)
        print("%40s ,             LR MAPE, %10.2f"%(setname,lrrmape), file=fp)
        print("%40s ,       LR Train MAPE, %10.2f"%(setname,lrrmaopetrain), file=fp)
        print("%40s ,        LR Test MAPE, %10.2f"%(setname,lrrmaoetest), file=fp)
        #print()
    
        # Custom Loss Linear Regression
        models_store[setname].lr_custom_model =\
                clr.custom_loss_lr (loss=clr.mean_absolute_percentage_error)
        models_store[setname].lr_custom_model.fit(X, Y, \
             beta_init_values=models_store[setname].lr_model.get_beta())
        y_pred_custom_lr = models_store[setname].lr_custom_model.predict(X)
        custom_lrrmse = root_mean_squared_error(Y, y_pred_custom_lr)
        custom_lrrmape = mean_absolute_percentage_error(Y, y_pred_custom_lr)
        models_store[setname].lr_custom_model_splitted  = \
                clr.custom_loss_lr (loss=clr.mean_absolute_percentage_error) 
        models_store[setname].lr_custom_model_splitted.fit(X_train, y_train,\
                    beta_init_values=models_store[setname].lr_model_splitted.get_beta())
        y_pred_custom_lr = models_store[setname].lr_custom_model_splitted.predict(X_test)
        custom_lrrmsetest = root_mean_squared_error(y_test, y_pred_custom_lr)
        custom_lrrmapetest = mean_absolute_percentage_error(y_test, y_pred_custom_lr)
        y_pred_custom_lr = models_store[setname].lr_custom_model_splitted.predict(X_train)
        custom_lrrmsetrain = root_mean_squared_error(y_train, y_pred_custom_lr)
        custom_lrrmapetrain = mean_absolute_percentage_error(y_train, y_pred_custom_lr)
        if PRINTALSOINSTDOUT:
            print("%40s ,      Custom LR RMSE, %10.2f"%(setname,custom_lrrmse))
            print("%40s ,Custom LR Train RMSE, %10.2f"%(setname,custom_lrrmsetrain))
            print("%40s , Custom LR Test RMSE, %10.2f"%(setname,custom_lrrmsetest))
            print("%40s ,      Custom LR MAPE, %10.2f"%(setname,custom_lrrmape))
            print("%40s ,Custom LR Train MAPE, %10.2f"%(setname,custom_lrrmapetrain))
            print("%40s , Custom LR Test MAPE: %10.2f"%(setname,custom_lrrmapetest))
    
        print("%40s ,      Custom LR RMSE, %10.2f"%(setname,custom_lrrmse), file=fp)
        print("%40s ,Custom LR Train RMSE, %10.2f"%(setname,custom_lrrmsetrain), file=fp)
        print("%40s , Custom LR Test RMSE, %10.2f"%(setname,custom_lrrmsetest), file=fp)
        print("%40s ,      Custom LR MAPE, %10.2f"%(setname,custom_lrrmape), file=fp)
        print("%40s ,Custom LR Train MAPE, %10.2f"%(setname,custom_lrrmapetrain), file=fp)
        print("%40s , Custom LR Test MAPE: %10.2f"%(setname,custom_lrrmapetest), file=fp)
    
    fp.close()

    setname = None
    ssetname = "Full"
    lr_model_full = models_store[ssetname].lr_model
    lr_model_full_splitted = models_store[ssetname].lr_model_splitted
    lr_custom_model_full = models_store[ssetname].lr_custom_model
    lr_custom_model_full_splitted = models_store[ssetname].lr_custom_model_splitted
    
    ypredFull_lr = []
    ypredFull_lr_split = []
    ypredFull_lr_custom = []
    ypredFull_lr_custom_split = []
    ypredFull_allbasissets = {}
    ypredFull_d3bj = []
    
    for method in models_results[ssetname].funcional_basisset_ypred:
        if method.find(selected_functional) != -1:
            ypredFull_allbasissets[method] = []
    
    setnamesFull = []
    
    fp = open("modelsresults.csv", "w")
    
    for ssetname in supersetnames:
        lr_model_ssetname = models_store[ssetname].lr_model
        lr_model_ssetname_splitted = models_store[ssetname].lr_model_splitted
        lr_custom_model_ssetname = models_store[ssetname].lr_custom_model
        lr_custom_model_ssetname_splitted = models_store[ssetname].lr_custom_model_splitted
    
        X, Y, features_names = \
                commonutils.build_XY_matrix (\
                models_results[ssetname].features,\
                models_results[ssetname].labels)
        setlist = models_results[ssetname].setnames
        setnamesFull.extend(setlist)
    
        # SuperSet LR
        y_pred = lr_model_ssetname.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        ypredFull_lr.extend(y_pred)
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , SS LR", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , SS LR", mape), file=fp)
        y_pred = lr_model_ssetname_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        ypredFull_lr_split.extend(y_pred)
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , SS LR split", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , SS LR split", mape), file=fp)
        
        # SuperSet Custom LR
        y_pred = lr_custom_model_ssetname.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        ypredFull_lr_custom.extend(y_pred)
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , SS Custom LR", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , SS Custom LR", mape), file=fp)
        y_pred = lr_custom_model_ssetname_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        ypredFull_lr_custom_split.extend(y_pred)
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , SS Custom LR split", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , SS Custom LR split", mape), file=fp)
    
        # Full LR
        y_pred = lr_model_full.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , Full LR", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , Full LR", mape), file=fp)
        ypred = lr_model_full_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , Full LR split", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , Full LR split", mape), file=fp)
    
        # Full Custom LR
        y_pred = lr_custom_model_full.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , Full Custom LR", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , Full Custom LR", mape), file=fp)
        y_pred = lr_custom_model_full_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        mape = mean_absolute_percentage_error(Y, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , Full Custom LR split", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , Full Custom LR split", mape), file=fp)
    
        mape = models_results[ssetname].insidemethods_mape["D3(BJ)"] 
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %7.3f"%(ssetname+" , D3(BJ)", mape))
        print(" %60s MAPE , %7.3f"%(ssetname+" , D3(BJ)", mape), file=fp)
        ypredFull_d3bj.extend(models_results[ssetname].insidemethods_ypred["D3(BJ)"])
    
        for method in models_results[ssetname].funcional_basisset_ypred:
            if method.find(selected_functional) != -1:
                y_pred = models_results[ssetname].funcional_basisset_ypred[method]
                ypredFull_allbasissets[method].extend(y_pred)
                if PRINTALSOINSTDOUT:
                    print(" %60s MAPE , %7.3f"%(ssetname+' , ' \
                        +method,\
                        mean_absolute_percentage_error(Y, y_pred)))
                print(" %60s MAPE , %7.3f"%(ssetname+' , '+  \
                        method,\
                        mean_absolute_percentage_error(Y, y_pred)), file=fp)
    fp.close()
     
    basissets_touse = set(basis_sets + [selected_basisset])
    functional_to_use = set(functionals + [selected_functional])
    
    classes = []
    features = {}
    supersetnameslist = list(supersetnames.keys())
    for setname in featuresvalues_perset:
        if setname in supersetnameslist:
            print("Setname: ", setname)
            for entry in featuresvalues_perset[setname]:
                classes.append(supersetnameslist.index(setname))
    
    X, Y, features_names =\
            commonutils.build_XY_matrix (\
            models_results['Full'].features,\
            models_results['Full'].labels)
    X_train, X_test, y_train, y_test = train_test_split(\
            X, classes, test_size=0.20, random_state=41)
    accuracys = []
    numoftrees = []
    for ntrees in range(10, 200, 10):
        rf = RandomForestClassifier(n_estimators=ntrees, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracys.append(accuracy)
        numoftrees.append(ntrees)
    
    bestaccuracy = max(accuracys)   
    bestntrees = numoftrees[accuracys.index(bestaccuracy)]
    print("Best accuracy: ", max(accuracys), " with ", bestntrees, " trees")
    
    rf = RandomForestClassifier(n_estimators=bestntrees, random_state=42)
    rf.fit(X_train, y_train)
    testaccuracy = rf.score(X_test, y_test)
    trainaccuracy = rf.score(X_train, y_train)
    overallaccuracy = rf.score(X, classes)
    print("  Train accuracy: %5.2f"%(trainaccuracy))
    print("   Test accuracy: %5.2f"%(testaccuracy))
    print("Overall accuracy: %5.2f"%(overallaccuracy))
    
    assert(len(ypredFull_lr) == len(ypredFull_lr_split)) 
    assert(len(ypredFull_lr) == len(ypredFull_lr_custom))
    assert(len(ypredFull_lr) == len(ypredFull_lr_custom_split))
    assert(len(ypredFull_lr) == len(models_results["Full"].labels))
    assert(len(ypredFull_lr) == len(setnamesFull))
    assert(len(ypredFull_lr) == len(ypredFull_d3bj))
    for method in ypredFull_allbasissets:
        assert(len(ypredFull_lr) == len(ypredFull_allbasissets[method]))
    
    X, Y, features_names =\
            commonutils.build_XY_matrix (models_results['Full'].\
            features,\
            models_results['Full'].labels)
    setlist = models_results['Full'].setnames
    
    y_pred_RF_LR = []
    y_pred_RF_LR_split = []
    y_pred_RF_LR_CUSTOM = []
    y_pred_RF_LR_CUSTOM_split = []
    for i in range(len(X)):
        c = rf.predict([X[i]])
        supersetrname= supersetnameslist[c[0]]
        #print("X: ", i, " Y: ", Y[i], " C: ", c, " ==> ", supersetnameslist[c[0]])
    
        y = models_store[supersetrname].lr_model.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR.append(y[0])
        y = models_store[supersetrname].lr_model_splitted.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR_split.append(y[0])
    
        y = models_store[supersetrname].lr_custom_model.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR_CUSTOM.append(y[0])
        y = models_store[supersetrname].lr_custom_model_splitted.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR_CUSTOM_split.append(y[0])
    
    fp = open("modelsresults.csv", "a")
    
    predictred = {}
    
    predictred["Full , using LR Full"] = \
                  models_store["Full"].lr_model.predict(X)
    if len(predictred["Full , using LR Full"].shape) == 2:
        predictred["Full , using LR Full"] = \
                            predictred["Full , using LR Full"][:,0]
    predictred["Full , using LR Full split"] = \
                models_store["Full"].lr_model_splitted.predict(X)
    if len(predictred["Full , using LR Full split"].shape) == 2:
        predictred["Full , using LR Full split"] =         \
            predictred["Full , using LR Full split"][:,0]
    predictred["Full , using LR SS"] = ypredFull_lr
    predictred["Full , using LR SS split"] = ypredFull_lr_split
    
    predictred["Full , using Custom LR Full"] = \
                models_store["Full"].lr_custom_model.predict(X)
    if len(predictred["Full , using Custom LR Full"].shape) == 2:
        predictred["Full , using Custom LR Full"] =  \
                          predictred["Full , using Custom LR Full"][:,0]
    predictred["Full , using Custom LR Full split"] =  \
              models_store["Full"].lr_custom_model_splitted.predict(X)
    if len(predictred["Full , using Custom LR Full split"].shape) == 2:
        predictred["Full , using Custom LR Full split"] = \
                            predictred["Full , using Custom LR Full split"][:,0]
    predictred["Full , using Custom LR SS"] = ypredFull_lr_custom
    predictred["Full , using Custom LR SS split"] = ypredFull_lr_custom_split
    
    predictred["Full , using LRRF"] = y_pred_RF_LR
    predictred["Full , using LRRF split"] = y_pred_RF_LR_split
    predictred["Full , using Custom LRRF"] = y_pred_RF_LR_CUSTOM
    predictred["Full , using Custom LRRF split"] = y_pred_RF_LR_CUSTOM_split

    mapes_to_collect = {}

    for m in predictred:
        ypred = predictred[m]
        wtamd2_full_usingss = \
                commonutils.wtmad2(setnamesFull,\
                ypred,\
                models_results["Full"].labels)
        wtamd2 = wtamd2_full_usingss["Full"]
        if PRINTALSOINSTDOUT:
            print("%44s %7.3f"%(m + " WTMAD2, ", wtamd2))
        print("%44s %7.3f"%(m + " WTMAD2, ", wtamd2), file=fp)
    
    for method in ypredFull_allbasissets:
        wtamd2_full_allbasissets = \
                commonutils.wtmad2(setnamesFull, \
                ypredFull_allbasissets[method], \
                models_results["Full"].labels)
        wtamd2 = wtamd2_full_allbasissets["Full"]
        if PRINTALSOINSTDOUT:
            print("%44s %7.3f"%("Full , "+ method + " WTMAD2, ", wtamd2))
        print("%44s %7.3f"%("Full , "+ method + " WTMAD2, ", wtamd2), file=fp)
    
    wtamd2_full_d3bj = \
            commonutils.wtmad2(setnamesFull, \
            ypredFull_d3bj,  \
            models_results["Full"].labels)
    wtamd2 = wtamd2_full_d3bj["Full"]
    if PRINTALSOINSTDOUT:
        print("%44s %7.3f"%("Full , D3(BJ) WTMAD2, ", wtamd2))
    print("%44s %7.3f"%("Full , D3(BJ) WTMAD2, ", wtamd2), file=fp)
    
    for m in predictred:
        ypred = predictred[m]
        mape_full_usingss = mean_absolute_percentage_error( \
                models_results["Full"].labels, ypred)
        if PRINTALSOINSTDOUT:
            print("%44s %7.3f"%(m + " MAPE, ", mape_full_usingss))
        print("%44s %7.3f"%(m + " MAPE, ", mape_full_usingss), file=fp)
        mapes_to_collect[m] = mape_full_usingss

    for method in ypredFull_allbasissets:
        mape_full_allbasissets = mean_absolute_percentage_error(  \
                models_results["Full"].labels, ypredFull_allbasissets[method])
        if PRINTALSOINSTDOUT:
            print("%44s %7.3f"%("Full , " + method + " MAPE, ", mape_full_allbasissets))
        print("%44s %7.3f"%("Full , " + method + " MAPE, ", mape_full_allbasissets), file=fp)
        mapes_to_collect[method] = mape_full_allbasissets

    mape_full_d3bj = mean_absolute_percentage_error( \
            models_results["Full"].labels, ypredFull_d3bj)
    if PRINTALSOINSTDOUT:
        print("%44s %7.3f"%("Full , D3(BJ) MAPE, ", mape_full_d3bj))
    print("%44s %7.3f"%("Full , D3(BJ) MAPE, ", mape_full_d3bj), file=fp)
    mapes_to_collect["D3(BJ)"] = mape_full_d3bj

    fp.close()

    for m in mapes_to_collect:
        mtop = m.replace("Full , using ", "")
        print(" %40s , %7.3f"%(mtop, mapes_to_collect[m]))
    
    fp = open("modelscoefficients.csv", "w")    
    for setname in list(supersetnames)+["Full"]:   
        lr_model = models_store[setname].lr_model
        lr_model_splitted = models_store[setname].lr_model_splitted
        lr_custom_model = models_store[setname].lr_custom_model
        lr_custom_model_splitted = models_store[setname].lr_custom_model_splitted
    
        X, Y, features_names = \
                commonutils.build_XY_matrix (models_results[setname].\
                features, \
                models_results[setname].labels)
        # LR model
        lr_test_and_rpint (lr_model, X, Y, setname + " LR ", features_names, fp)
        lr_test_and_rpint (lr_model_splitted, X, Y, setname + " LR split ", \
                           features_names, fp)
    
        # Custom LR model
        lr_test_and_rpint (lr_custom_model, X, Y, setname + " Custom LR ", \
                           features_names, fp)
        lr_test_and_rpint (lr_custom_model_splitted, X, Y, \
                        setname + " Custom LR split ", \
                        features_names, fp)
    
    fp.close()
