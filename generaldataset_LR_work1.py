import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import commonutils
import models

from commonutils import ModelResults

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
#from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
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

warnings.simplefilter("ignore")

from commonfuncsforcli import *

CHECKANDTESTSINGLEMODEL = False
EXTRACTFLPS = True

REMOVEFLPS = False
STARTBETAONE = False

if __name__ == "__main__":

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print("Usage: python3 runmodels.py " + \
              "<selected_functional> <selected_basisset> " + \
                "[<functionals> <basis_sets>]")
        sys.exit(1) 

    # all features 
    # equations = {"EC" :"EC" ,\
    #             "EX" : "EX",\
    #             "FSPE" : "FINAL_SINGLE_POINT_ENERGY",\
    #             "DC" : "Dispersion_correction",\
    #             "PE" : "Potential_Energy",\
    #             "KE" : "Kinetic_Energy",\
    #             "OEE" : "One_Electron_Energy",\
    #             "TEE" : "Two_Electron_Energy",\
    #             "NR" : "Nuclear_Repulsion"}
    # SHIFTFT = ""

    # First Reduced Form:
    equations = {"Te": "Kinetic_Energy", \
             "V_NN": "Nuclear_Repulsion",\
             "V_eN": "One_Electron_Energy - Kinetic_Energy",\
             "EX": "EX",\
             "E_J": "Two_Electron_Energy - EX - EC",\
             "DC": "Dispersion_correction",\
             "EC": "EC"}
    SHIFTFT = "DC"

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
    print(" Selected basis set: ", selected_basisset)
    print("        Functionals: ", functionals)
    print("         Basis sets: ", basis_sets)
    
    featuresvalues_perset,\
    fullsetnames, \
    models_results, \
    supersetnames = readdata(shiftusingFT=SHIFTFT, \
                    selected_functionalin=selected_functional, \
                    selected_basisin=selected_basisset, \
                    equations=equations)
    
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
    
    # features selection
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

    # test separately need to split in two sets the data 
    # specific for FLPs thata is at the end
    flpsdim = 14
    Xflps = []
    yflps = None
    ftsflps = []
    Xtherest = []
    ytherest = None
    ftsrest = []
    if EXTRACTFLPS:
        flpsname = "FLPs"
        flpssupername  = "LARGE_SYSTEMS"
        flpsfullname  = "LARGE_SYSTEMS_FLPs"
 
        for k in models_results[flpsfullname].features:
            Xflps.append(models_results[flpsfullname].features[k])
        Xflps = np.array(Xflps).T
        yflps = np.array(models_results[flpsfullname].labels)
        ftsflps = np.array(models_results[flpsfullname].fts)

        fulldim = len(models_results["Full"].labels)
        for k in models_results["Full"].features:
            Xtherest.append(models_results["Full"].features[k])
        Xtherest = np.array(Xtherest).T
        ytherest = np.array(models_results["Full"].labels) 
        Xtherest = Xtherest[:-flpsdim]
        ytherest = ytherest[:-flpsdim]
        ftsrest = models_results["Full"].fts[:-flpsdim] 

        print("Xflps shape: ", Xflps.shape)
        print("yflps shape: ", yflps.shape)
        print("Xtherest shape: ", Xtherest.shape)
        print("ytherest shape: ", ytherest.shape)
        print("Full dim ", len(models_results["Full"].labels))

    if REMOVEFLPS:
        print("Removing FLPS from training")
        flpsname = "FLPs"
        flpssupername  = "LARGE_SYSTEMS"
        flpsfullname = "LARGE_SYSTEMS_FLPs"

        del models_results[flpsfullname]
        # remove last 14 elements each list
        for setname in [flpssupername, "Full"]:
            models_results[setname].setnames = \
                models_results[setname].setnames[:-flpsdim]   
            models_results[setname].supersetnames = \
                models_results[setname].supersetnames[:-flpsdim]
            models_results[setname].chemicals = \
                models_results[setname].chemicals[:-flpsdim]
            models_results[setname].labels = \
                models_results[setname].labels[:-flpsdim]
            models_results[setname].fts = \
                models_results[setname].fts[:-flpsdim]
            
            for featname in models_results[setname].features:
                models_results[setname].features[featname] = \
                    models_results[setname].features[featname][:-flpsdim]
            for funtionalpredname in models_results[setname].funcional_basisset_ypred:
                models_results[setname].funcional_basisset_ypred[funtionalpredname] = \
                    models_results[setname].funcional_basisset_ypred[funtionalpredname][:-flpsdim]
            for predtionname in models_results[setname].insidemethods_ypred:
                models_results[setname].insidemethods_ypred[predtionname] = \
                    models_results[setname].insidemethods_ypred[predtionname][:-flpsdim]

    if CHECKANDTESTSINGLEMODEL:
        for setname in list(supersetnames)+["Full"]:
            print("Running models for dataset: ", setname)
            X, Y, features_names = \
                commonutils.build_XY_matrix \
                (models_results[setname].features, \
                models_results[setname].labels)
            lr_start_model = clr.custom_loss_lr (loss=clr.mean_average_error)
            try:
                lr_start_model.fit(X, Y)
            except Exception as e:
                print("Error: ", e)
                lr_start_model.set_solver("Nelder-Mead")
                lr_start_model.fit(X, Y)
                    
            lrmodel = clr.custom_loss_lr (loss=clr.mean_absolute_percentage_error)
            try:
                lrmodel.fit(X, Y, \
                    beta_init_values = lr_start_model.get_beta())
            except Exception as e:
                print("Error: ", e)
                lrmodel.set_solver("Nelder-Mead")
                lrmodel.fit(X, Y, \
                    beta_init_values = lr_start_model.get_beta())

            y_pred = lrmodel.predict(X)

            fp = open("modelresults_"+ setname +".csv", "w")
            if SHIFTFT != "":
                Yt, y_predt = shiftbackdata (Y, y_pred, models_results[setname].fts)  

                print("Chemicals, True, Predicted, NR, True+NR , Predicted+NR", file=fp)
                for i , yp in enumerate(y_pred):
                    print("%s, %12.6f ,  %12.6f , %12.6f , %12.6f , %12.6f"%(\
                        models_results[setname].chemicals[i],
                        Y[i], yp, models_results[setname].fts[i], \
                        Yt[i], y_predt[i]), \
                        file=fp)
                fp.close()

                Y, y_pred = shiftbackdata (Y, y_pred, models_results[setname].fts)
            else:

                print("True, Predicted", file=fp)
                for i , yp in enumerate(y_pred):
                    print("%12.6f ,  %12.6f "%(\
                        Y[i], yp), \
                        file=fp)
                fp.close()

            mape = mean_absolute_percentage_error(Y, y_pred)
            print("MAPE: %12.6f"%(mape))
            rmse = np.sqrt(mean_squared_error(Y, y_pred))
            mean = np.mean(Y)
            std = np.std(Y)
            min = np.min(Y)
            max = np.max(Y)
            print("RMSE: %12.6f [%3.2f]"%(rmse, rmse/mean))
            print(" AVG %12.6f STD %12.6f MIN %12.6f MAX %12.6f"%( \
                mean, std, min, max))
            # save a scatterplot of the results
            plt.cla()
            plt.scatter(Y, y_pred)
            plt.xlabel("True values")
            plt.ylabel("Predicted values")
            plt.title("True vs Predicted values")
            plt.savefig(setname + "_scatterplot.png")
        exit()
    
    models_store = {}
    #fp = open("modelsgeneral.csv", "w")
    for setname in list(supersetnames)+["Full"]:
        models_store[setname] = ModelsStore()
    
        print("Running models for dataset: ", setname)
        X, y, features_names = \
                commonutils.build_XY_matrix \
                (models_results[setname].features, \
                models_results[setname].labels)
        X_test = None
        X_train = None
        y_test = None
        y_train = None
        fts_test = None
        fts_train = None
        if len(models_results[setname].fts) > 0:
            X_train, X_test, y_train, y_test, fts_train, fts_test \
              = train_test_split(X, y, models_results[setname].fts,
                          test_size=0.20, random_state=42)
        else:
            X_train, X_test, y_train, y_test \
              = train_test_split(X, y, test_size=0.20, random_state=42)
        setlist = models_results[setname].setnames  
        supersetlist = models_results[setname].supersetnames
        y_true = y  
        y_test_true = y_test
        y_train_true = y_train
        fts = models_results[setname].fts

        # Linear regression model to get starting beta values
        lr_start_model = clr.custom_loss_lr (loss=clr.mean_average_error)
        lr_start_split_model = clr.custom_loss_lr (loss=clr.mean_average_error)
        if not STARTBETAONE:
            try:
                lr_start_model.fit(X, y)
            except Exception as e:
                print("Error: ", e)
                lr_start_model.set_solver("Nelder-Mead")
                lr_start_model.fit(X, y)
            try:
                lr_start_split_model.fit(X_train, y_train)
            except Exception as e:
                print("Error: ", e)
                lr_start_split_model.set_solver("Nelder-Mead")
                lr_start_split_model.fit(X_train, y_train)
    
        # Custom Loss Linear Regression
        models_store[setname].lr_custom_model =\
                clr.custom_loss_lr (loss=clr.mean_absolute_percentage_error)
        try:
            if STARTBETAONE:
                models_store[setname].lr_custom_model.fit(X, y) 
            else:
                models_store[setname].lr_custom_model.fit(X, y, \
                    beta_init_values = lr_start_model.get_beta())
        except Exception as e:
            print("Error: ", e)
            models_store[setname].lr_custom_model.set_solver("Nelder-Mead")
            if STARTBETAONE:
                models_store[setname].lr_custom_model.fit(X, y)
            else:
                models_store[setname].lr_custom_model.fit(X, y, \
                    beta_init_values = lr_start_model.get_beta())
        y_pred_custom_lr = models_store[setname].lr_custom_model.predict(X)
        if SHIFTFT != "":
            y_true, y_pred_custom_lr = shiftbackdata (y, y_pred_custom_lr, fts)
        custom_lrrmse = 0.0
        try:
            custom_lrrmse = root_mean_squared_error(y_true, y_pred_custom_lr)
        except:
            custom_lrrmse = np.sqrt(mean_squared_error(y_true, y_pred_custom_lr))    
        custom_lrrmape = mean_absolute_percentage_error(y_true, y_pred_custom_lr)
        models_store[setname].lr_custom_model_splitted  = \
                clr.custom_loss_lr (loss=clr.mean_absolute_percentage_error)
        try:
            if STARTBETAONE:
                models_store[setname].lr_custom_model_splitted.fit(X_train, y_train)
            else:
                models_store[setname].lr_custom_model_splitted.fit(X_train, y_train, \
                    beta_init_values = lr_start_split_model.get_beta())
        except Exception as e:
            print("Error: ", e)
            models_store[setname].lr_custom_model_splitted.set_solver("Nelder-Mead")
            if STARTBETAONE:
                models_store[setname].lr_custom_model_splitted.fit(X_train, y_train)
            else:
                models_store[setname].lr_custom_model_splitted.fit(X_train, y_train, \
                    beta_init_values = lr_start_split_model.get_beta())
        y_pred_custom_lr = models_store[setname].lr_custom_model_splitted.predict(X_test)
        if SHIFTFT != "":
            y_test_true, y_pred_custom_lr = shiftbackdata (y_test, y_pred_custom_lr, fts_test)
        custom_lrrmsetest = 0.0
        try:
            custom_lrrmsetest = root_mean_squared_error(y_test_true, y_pred_custom_lr)
        except:
            custom_lrrmsetest = np.sqrt(mean_squared_error(y_test_true, y_pred_custom_lr))
        custom_lrrmapetest = mean_absolute_percentage_error(y_test_true, y_pred_custom_lr)
        y_pred_custom_lr = models_store[setname].lr_custom_model_splitted.predict(X_train)
        if SHIFTFT != "":
            y_train_true, y_pred_custom_lr = shiftbackdata (y_train, y_pred_custom_lr, fts_train)
        custom_lrrmsetrain = 0.0
        try:
            custom_lrrmsetrain = root_mean_squared_error(y_train_true, y_pred_custom_lr)
        except:
            custom_lrrmsetrain = np.sqrt(mean_squared_error(y_train_true, y_pred_custom_lr))
        custom_lrrmapetrain = mean_absolute_percentage_error(y_train_true, y_pred_custom_lr)
        if PRINTALSOINSTDOUT:
            print("%40s ,      Custom LR RMSE, %12.6f"%(setname,custom_lrrmse))
            print("%40s ,Custom LR Train RMSE, %12.6f"%(setname,custom_lrrmsetrain))
            print("%40s , Custom LR Test RMSE, %12.6f"%(setname,custom_lrrmsetest))
            print("%40s ,      Custom LR MAPE, %12.6f"%(setname,custom_lrrmape))
            print("%40s ,Custom LR Train MAPE, %12.6f"%(setname,custom_lrrmapetrain))
            print("%40s , Custom LR Test MAPE: %12.6f"%(setname,custom_lrrmapetest))
    
        #print("%40s ,      Custom LR RMSE, %12.6f"%(setname,custom_lrrmse), file=fp)
        #print("%40s ,Custom LR Train RMSE, %12.6f"%(setname,custom_lrrmsetrain), file=fp)
        #print("%40s , Custom LR Test RMSE, %12.6f"%(setname,custom_lrrmsetest), file=fp)
        #print("%40s ,      Custom LR MAPE, %12.6f"%(setname,custom_lrrmape), file=fp)
        #print("%40s ,Custom LR Train MAPE, %12.6f"%(setname,custom_lrrmapetrain), file=fp)
        #print("%40s , Custom LR Test MAPE: %12.6f"%(setname,custom_lrrmapetest), file=fp)
    
    #fp.close()
              
    setname = None
    ssetname = "Full"
    lr_custom_model_full = models_store[ssetname].lr_custom_model
    lr_custom_model_full_splitted = models_store[ssetname].lr_custom_model_splitted
    
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
        lr_custom_model_ssetname = models_store[ssetname].lr_custom_model
        lr_custom_model_ssetname_splitted = models_store[ssetname].lr_custom_model_splitted
    
        X, y, features_names = \
                commonutils.build_XY_matrix (\
                models_results[ssetname].features,\
                models_results[ssetname].labels)
        setlist = models_results[ssetname].setnames
        setnamesFull.extend(setlist)
        y_true = y
        fts = models_results[ssetname].fts

        # SuperSet Custom LR
        y_pred = lr_custom_model_ssetname.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        if SHIFTFT != "":
            y_true, y_pred = shiftbackdata (y, y_pred, fts)
        ypredFull_lr_custom.extend(y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %12.6f"%(ssetname+" , SS Custom LR", mape))
        print(" %60s MAPE , %12.6f"%(ssetname+" , SS Custom LR", mape), file=fp)
        y_pred = lr_custom_model_ssetname_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        if SHIFTFT != "":
            y_true, y_pred = shiftbackdata (y, y_pred, fts)
        ypredFull_lr_custom_split.extend(y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %12.6f"%(ssetname+" , SS Custom LR split", mape))
        print(" %60s MAPE , %12.6f"%(ssetname+" , SS Custom LR split", mape), file=fp)
    
        # Full Custom LR
        y_pred = lr_custom_model_full.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        if SHIFTFT != "":
            y_true, y_pred = shiftbackdata (y, y_pred, fts)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %12.6f"%(ssetname+" , Full Custom LR", mape))
        print(" %60s MAPE , %12.6f"%(ssetname+" , Full Custom LR", mape), file=fp)
        y_pred = lr_custom_model_full_splitted.predict(X)
        if len(y_pred.shape) == 2:
            y_pred = y_pred[:,0]
        if SHIFTFT != "":
            y_true, y_pred = shiftbackdata (y, y_pred, fts)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %12.6f"%(ssetname+" , Full Custom LR split", mape))
        print(" %60s MAPE , %12.6f"%(ssetname+" , Full Custom LR split", mape), file=fp)
    
        mape = models_results[ssetname].insidemethods_mape["D3(BJ)"] 
        if PRINTALSOINSTDOUT:
            print(" %60s MAPE , %12.6f"%(ssetname+" , D3(BJ)", mape))
        print(" %60s MAPE , %12.6f"%(ssetname+" , D3(BJ)", mape), file=fp)
        ypredFull_d3bj.extend(models_results[ssetname].insidemethods_ypred["D3(BJ)"])

        if SHIFTFT != "": 
            y_true = y + fts
        for method in models_results[ssetname].funcional_basisset_ypred:
            if method.find(selected_functional) != -1:
                y_pred = models_results[ssetname].funcional_basisset_ypred[method]
                ypredFull_allbasissets[method].extend(y_pred)
                mapecompute = mean_absolute_percentage_error(y_true, y_pred)
                if SHIFTFT != "":
                    if method.find("MINIX") != -1:
                        print("Warning the MINIX NRS is different for heavy atoms")
                        #mapecompute = float('nan')
                        # the MINIX NRS is different for heavy atoms
                if PRINTALSOINSTDOUT:
                    print(" %60s MAPE , %12.6f"%(ssetname+' , ' \
                        +method, mapecompute))
                print(" %60s MAPE , %12.6f"%(ssetname+' , '+  \
                        method, mapecompute), file=fp)
    fp.close()

    basissets_touse = set(basis_sets + [selected_basisset])
    functional_to_use = set(functionals + [selected_functional])

    # build RF model
    classes = []
    features = {}
    supersetnameslist = list(supersetnames.keys())
    for setname in featuresvalues_perset:
        if setname in supersetnameslist:
            print("Setname: ", setname)
            for entry in featuresvalues_perset[setname]:
                classes.append(supersetnameslist.index(setname))
    if REMOVEFLPS:
        classes = classes[:-flpsdim]
        #supersetnameslist = supersetnameslist[:-flpsdim]

    X, _, features_names =\
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
    
    assert(len(ypredFull_lr_custom) == len(ypredFull_lr_custom_split))
    assert(len(ypredFull_lr_custom) == len(models_results["Full"].labels))
    assert(len(ypredFull_lr_custom) == len(setnamesFull))
    assert(len(ypredFull_lr_custom) == len(ypredFull_d3bj))
    for method in ypredFull_allbasissets:
        assert(len(ypredFull_lr_custom) == len(ypredFull_allbasissets[method]))
    
    X, _, features_names =\
            commonutils.build_XY_matrix (models_results['Full'].\
            features,\
            models_results['Full'].labels)
    setlist = models_results['Full'].setnames
    
    y_pred_RF_LR_CUSTOM = []
    y_pred_RF_LR_CUSTOM_split = []
    for i in range(len(X)):
        c = rf.predict([X[i]])
        nr = 0.0
        supersetrname = supersetnameslist[c[0]]
        #print("X: ", i, " Y: ", Y[i], " C: ", c, " ==> ", supersetnameslist[c[0]])
    
        if SHIFTFT != "":
            # this is maybe not properly correct but it is a good approximation
            nr = models_results['Full'].fts[i]

        y = models_store[supersetrname].lr_custom_model.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR_CUSTOM.append(y[0]+nr)
        y = models_store[supersetrname].lr_custom_model_splitted.predict([X[i]])
        if len(y.shape) == 2:
            y = y[:,0]
        y_pred_RF_LR_CUSTOM_split.append(y[0]+nr)

    fp = open("modelsresults.csv", "a")
    
    predictred = {}
    
    X, y, features_names =\
            commonutils.build_XY_matrix (models_results['Full'].\
            features,\
            models_results['Full'].labels) 
    y_true = y
    fts = models_results["Full"].fts

    y_pred = models_store["Full"].lr_custom_model.predict(X)
    if SHIFTFT != "":
        y_true, y_pred = shiftbackdata (y, y_pred, fts)   
    predictred["Full , using Custom LR Full"] = y_pred
    if len(predictred["Full , using Custom LR Full"].shape) == 2:
        predictred["Full , using Custom LR Full"] =  \
                          predictred["Full , using Custom LR Full"][:,0]
    y_pred = models_store["Full"].lr_custom_model_splitted.predict(X)
    if SHIFTFT != "":
        y_true, y_pred = shiftbackdata (y, y_pred, fts)
    predictred["Full , using Custom LR Full split"] =  y_pred
    if len(predictred["Full , using Custom LR Full split"].shape) == 2:
        predictred["Full , using Custom LR Full split"] = \
                            predictred["Full , using Custom LR Full split"][:,0]
    predictred["Full , using Custom LR SS"] = ypredFull_lr_custom
    predictred["Full , using Custom LR SS split"] = ypredFull_lr_custom_split
    
    predictred["Full , using Custom LRRF"] = y_pred_RF_LR_CUSTOM
    predictred["Full , using Custom LRRF split"] = y_pred_RF_LR_CUSTOM_split

    mapes_to_collect = {}

    y_true = models_results["Full"].labels
    if SHIFTFT != "":
        for i in range(len(y_true)):
            y_true[i] = y_true[i] + models_results["Full"].fts[i]

    for m in predictred:
        ypred = predictred[m]
        mape_full_usingss = mean_absolute_percentage_error( \
                y_true, ypred)
        if PRINTALSOINSTDOUT:
            print("%44s %12.6f"%(m + " MAPE, ", mape_full_usingss))
        print("%69s %12.6f"%(m + " MAPE ,", mape_full_usingss), file=fp)
        mapes_to_collect[m] = mape_full_usingss

    for method in ypredFull_allbasissets:
        mape_full_allbasissets = mean_absolute_percentage_error(  \
                models_results["Full"].labels, ypredFull_allbasissets[method])
        if PRINTALSOINSTDOUT:
            print("%44s %12.6f"%("Full , " + method + " MAPE, ", mape_full_allbasissets))
        print("%69s %12.6f"%("Full , " + method + " MAPE ,", mape_full_allbasissets), file=fp)
        mapes_to_collect[method] = mape_full_allbasissets

    mape_full_d3bj = mean_absolute_percentage_error( \
            models_results["Full"].labels, ypredFull_d3bj)
    if PRINTALSOINSTDOUT:
        print("%44s %12.6f"%("Full , D3(BJ) MAPE, ", mape_full_d3bj))
    print("%69s %12.6f"%("Full , D3(BJ) MAPE ,", mape_full_d3bj), file=fp)
    mapes_to_collect["D3(BJ)"] = mape_full_d3bj

    if EXTRACTFLPS:
        y_pred_RF_LR_CUSTOM_FORGMTK = []
        y_pred_RF_LR_CUSTOM_FORGMTK_split = []
        y_pred_RF_LR_CUSTOM_FORFLPS = []
        y_pred_RF_LR_CUSTOM_FORFLPS_split = []
        for i in range(len(Xtherest)):
            c = rf.predict([Xtherest[i]])
            nr = 0.0
            supersetrname = supersetnameslist[c[0]]
            #print("X: ", i, " Y: ", Y[i], " C: ", c, " ==> ", supersetnameslist[c[0]])
           
            if SHIFTFT != "":
                # this is maybe not properly correct but it is a good approximation
                nr = ftsrest[i]
           
            y = models_store[supersetrname].lr_custom_model.predict([X[i]])
            if len(y.shape) == 2:
                y = y[:,0]
            y_pred_RF_LR_CUSTOM_FORGMTK.append(y[0]+nr)
            y = models_store[supersetrname].lr_custom_model_splitted.predict([X[i]])
            if len(y.shape) == 2:
                y = y[:,0]
            y_pred_RF_LR_CUSTOM_FORGMTK_split.append(y[0]+nr)

        fpflps = open("modelsresultsflps.csv", "w") 
        for i in range(len(Xflps)):
            c = rf.predict([Xflps[i]])
            nr = 0.0
            supersetrname = supersetnameslist[c[0]]
           
            if SHIFTFT != "":
                # this is maybe not properly correct but it is a good approximation
                nr = ftsflps[i]
           
            y = models_store[supersetrname].lr_custom_model.predict([Xflps[i]])
            if len(y.shape) == 2:
                y = y[:,0]
            y_pred_RF_LR_CUSTOM_FORFLPS.append(y[0]+nr)
            y = models_store[supersetrname].lr_custom_model_splitted.predict([Xflps[i]])
            if len(y.shape) == 2:
                y = y[:,0]
            y_pred_RF_LR_CUSTOM_FORFLPS_split.append(y[0]+nr)

            ypredrealmodel = models_store["LARGE_SYSTEMS"].lr_custom_model.predict([Xflps[i]]) 
            ypredrfmodel = models_store[supersetrname].lr_custom_model.predict([Xflps[i]])
            ypredfull = models_store["Full"].lr_custom_model.predict([Xflps[i]])
            print("X: ", i, " Ytrue: %10.2f"%(yflps[i]), \
                  "Ypred: %10.2f"%(ypredrealmodel[0]+nr), \
                  "YpredRF: %10.2f"%(ypredrfmodel[0]+nr), \
                  "YpredFull: %10.2f"%(ypredfull[0]+nr), \
                  " ==> ", supersetnameslist[c[0]],file=fpflps)
            #print("X: ", i, " Y: %10.2f"%(yflps[i]), " C: ", c, " ==> ", supersetnameslist[c[0]])
            #print("Ypred: %10.2f"%(ypredrealmodel[0]+nr), " YpredRF: %10.2f"%(ypredrfmodel[0]+nr))
        fpflps.close()

        y_pred_LR_CUSTOM_FULL_FORGMTK = \
                models_store["Full"].lr_custom_model.predict(Xtherest)
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_FULL_FORGMTK)):
                y_pred_LR_CUSTOM_FULL_FORGMTK[i] = \
                        y_pred_LR_CUSTOM_FULL_FORGMTK[i] + ftsrest[i]
        y_pred_LR_CUSTOM_FULL_FORGMTK_split = \
                models_store["Full"].lr_custom_model_splitted.predict(Xtherest)
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_FULL_FORGMTK_split)):
                y_pred_LR_CUSTOM_FULL_FORGMTK_split[i] = \
                        y_pred_LR_CUSTOM_FULL_FORGMTK_split[i] + ftsrest[i]
        y_pred_LR_CUSTOM_FULL_FORFLPS = \
                models_store["Full"].lr_custom_model.predict(Xflps)
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_FULL_FORFLPS)):
                y_pred_LR_CUSTOM_FULL_FORFLPS[i] = \
                        y_pred_LR_CUSTOM_FULL_FORFLPS[i] + ftsflps[i]
        y_pred_LR_CUSTOM_FULL_FORFLPS_split = \
                models_store["Full"].lr_custom_model_splitted.predict(Xflps)
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_FULL_FORFLPS_split)):
                y_pred_LR_CUSTOM_FULL_FORFLPS_split[i] = \
                        y_pred_LR_CUSTOM_FULL_FORFLPS_split[i] + ftsflps[i]
        y_pred_LR_CUSTOM_SS_FORFLPS = \
            models_store["LARGE_SYSTEMS"].lr_custom_model.predict(Xflps)    
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_SS_FORFLPS)):
                y_pred_LR_CUSTOM_SS_FORFLPS[i] = \
                        y_pred_LR_CUSTOM_SS_FORFLPS[i] + ftsflps[i]
        y_pred_LR_CUSTOM_SS_FORFLPS_split = \
            models_store["LARGE_SYSTEMS"].lr_custom_model_splitted.predict(Xflps)
        if SHIFTFT != "":
            for i in range(len(y_pred_LR_CUSTOM_SS_FORFLPS_split)):
                y_pred_LR_CUSTOM_SS_FORFLPS_split[i] = \
                        y_pred_LR_CUSTOM_SS_FORFLPS_split[i] + ftsflps[i]

        # for GMTK
        y_true = ytherest
        if SHIFTFT != "":
            for i in range(len(y_true)):
                y_true[i] = y_true[i] + ftsrest[i]
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_RF_LR_CUSTOM_FORGMTK)
        print(" %60s MAPE , %12.6f"%("GMTK , Custom LR RF", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_RF_LR_CUSTOM_FORGMTK_split)
        print(" %60s MAPE , %12.6f"%("GMTK , Custom LR RF split", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_LR_CUSTOM_FULL_FORGMTK)
        print(" %60s MAPE , %12.6f"%("GMTK , Custom LR Full", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_LR_CUSTOM_FULL_FORGMTK_split)
        print(" %60s MAPE , %12.6f"%("GMTK , Custom LR Full split", mape), file=fp)

        # for FLPs
        y_true = yflps
        if SHIFTFT != "":
            for i in range(len(y_true)):
                y_true[i] = y_true[i] + ftsflps[i]
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_RF_LR_CUSTOM_FORFLPS)
        print(" %60s MAPE , %12.6f"%("FLPs , Custom LR RF", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_RF_LR_CUSTOM_FORFLPS_split)
        print(" %60s MAPE , %12.6f"%("FLPs , Custom LR RF split", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_LR_CUSTOM_FULL_FORFLPS)
        print(" %60s MAPE , %12.6f"%("FLPs , Custom LR Full", mape), file=fp)
        mape = mean_absolute_percentage_error(y_true, \
                            y_pred_LR_CUSTOM_FULL_FORFLPS_split)
        print(" %60s MAPE , %12.6f"%("FLPs , Custom LR Full split", mape), file=fp)

    fp.close()

    for m in mapes_to_collect:
        mtop = m.replace("Full , using ", "")
        print(" %40s , %12.6f"%(mtop, mapes_to_collect[m]))
    
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
    
        # Custom LR model
        lr_test_and_rpint (lr_custom_model, X, Y, setname + " Custom LR ", \
                        features_names, fp, \
                        shiftft=SHIFTFT, fts=models_results[setname].fts)
        lr_test_and_rpint (lr_custom_model_splitted, X, Y, \
                        setname + " Custom LR split ", \
                        features_names, fp, \
                        shiftft=SHIFTFT, fts=models_results[setname].fts)
    
    fp.close()
