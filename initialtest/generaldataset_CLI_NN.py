import commonutils
import models
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from dataclasses import dataclass
import prettyprinter as pp

from sklearn.cross_decomposition import PLSRegression
import warnings
import sys

from sklearn import preprocessing

from copy import deepcopy

###########################################################################

def dump_predictions (fullsetnames, methods, \
                      allvalues_perset, models_results):
    
    for setname in fullsetnames:
        y_pred = models_results[setname].y_pred
        y_pred_rmcorr = models_results[setname].y_pred_rmcorr

        fp = open(setname+"_predictions.csv", "w")

        print("#;" + \
                "chemicals;" + \
                "stechio_ceofs;" + \
                "label;", end="", file=fp)
        mainidx = 0
        for i, d in enumerate(allvalues_perset[setname][mainidx]["difs"]):
            print("dif%d;"%(i+1), end="", file=fp)
        for m in methods:
            print(m + ";", end="", file=fp)
        print("y_pred;", end="", file=fp)
        print("y_pred_rmcorr;", end="", file=fp)
        
        for mainidx in range(len(allvalues_perset[setname])):
            print(mainidx+1, " ; " , end="", file=fp)
            for c in allvalues_perset[setname][mainidx]["chemicals"]:
                print(c, " ", end="", file=fp)
            print(" ; ", end="", file=fp)
            for s in allvalues_perset[setname][mainidx]["stechio_ceofs"]:
                print(s, " ", end="", file=fp)
            print(" ; ", end="", file=fp)
            print(allvalues_perset[setname][mainidx]["label"], " ; ", \
                  end="", file=fp)
            for d in allvalues_perset[setname][mainidx]["difs"]:
                print(d, " ; ", end="", file=fp) 
            for m in methods:
                print(allvalues_perset[setname][mainidx]\
                      [m+"_energydiff"][m+"_FINAL_SINGLE_POINT_ENERGY"], \
                        " ; ", end="", file=fp)
            print(y_pred[mainidx], " ; ", end="", file=fp)
            print(y_pred_rmcorr[mainidx], " ; ", end="", file=fp)
            print("", file=fp)

        fp.close()

###########################################################################

def read_and_init (inrootdir, supersetnames, howmanydifs, methods, \
                   DEBUG=False):
    
    allvalues_perset = {}
    fullsetnames = []
    models_results = {}

    toberemoved = {}
    for super_setname in supersetnames:
        toberemoved[super_setname] = []
        allvalues_perset[super_setname] = []
        fullsetnames.append(super_setname)
        for i, setname in enumerate(supersetnames[super_setname]):
              print("Reading dataset: ", setname)
              rootdir = inrootdir + super_setname + "/" +setname
              labelsfilename = inrootdir + setname +"_labels.txt"
        
              values =\
                    commonutils.read_dataset(rootdir, labelsfilename, \
                                             howmanydifs, methods, \
                                             debug=DEBUG)
              
              if (values is None) or (len(values) <= 2):
                    print(setname + " No data found for this dataset")
                    print("")
                    toberemoved[super_setname].append(i)
              else:
                    fullsetname = super_setname+"_"+setname
                    fullsetnames.append(fullsetname)
                    allvalues_perset[fullsetname] = values  
                    print("Number of samples: ", len(allvalues_perset[fullsetname]))
                    print("Number of basic descriptors: ", len(allvalues_perset[fullsetname]))
              
                    allvalues_perset[super_setname] += allvalues_perset[fullsetname]
                    print("")

    for super_setname in toberemoved:
        for i in sorted(toberemoved[super_setname], reverse=True):
          del supersetnames[super_setname][i]
    
    allvalues_perset["Full"] = []
    for super_setname in supersetnames:
          allvalues_perset["Full"] += allvalues_perset[super_setname]  
    fullsetnames.append("Full")

    for setname in fullsetnames:
        models_results[setname] = ModelResults()

    return allvalues_perset, fullsetnames, models_results

###########################################################################
@dataclass
class ModelResults:
    # predicted values
    y_pred: list = None
    y_pred_tmcorr: list = None
    # data related to full set
    fulldescriptors: list = None
    labels: list = None
    top_correlation: list = None
    fulldescriptors_rmcorr: list = None
    labels_rmcorr: list = None
    # data realated to inside and our methods
    inside_methods_rmse: list = None
    bestinsidemethod_rmse: float = 0.0
    bestinsidemethod: str = None
    inside_methods_r2: list = None
    our_methods_rmse: dict = None
    bestourmethod_rmse: float = 0.0
    bestourmethod: str = None
    our_methods_r2: dict = None
    our_methods_name : list = None

###########################################################################
    
if __name__ == '__main__':
    
    warnings.simplefilter("ignore")
    
    DEBUG = False
    CORRCUT = 0.99

    supersetnames = {"BARRIER_HEIGHTS" : \
                       ["BH76","BHDIV10","BHPERI",\
                        "BHROT27","INV24","PX13","WCPT18"] \
                    ,"INTRAMOLECULAR_INTERACTIONS" : \
                       ["ACONF","ICONF","IDISP","MCONF",\
                        "PCONF21","SCONF","UPU23"] , \
                    "SMALL_MOLECULES" :\
                        ["AL2X6","ALK8","ALKBDE10","BH76",\
                         "DC13","DIPCS10","FH51","G21EA",\
                         "G21IP","G2RC","HEAVYSB11","NBPRC",\
                         "PA26","RC21","SIE4x4","TAUT15",\
                         "W4-11","YBDE18"], \
                    "INTERMOLECULAR_INTERACTIONS" :\
                       ["ADIM6","AHB21","CARBHB12",\
                        "CHB6","HAL59","HEAVY28","IL16",\
                        "PNICO23","RG18","S22","S66","WATER27"] , \
                    "LARGE_SYSTEMS" :\
                        ["BSR36","C60ISO","CDIE20","DARC",\
                         "ISO34","ISOL24","MB16-43","PArel",\
                            "RSE43"]}    
    howmanydifs = 3
    methods = {"PBE" : ["Nuclear Repulsion  :", \
                        "One Electron Energy:", \
                        "Two Electron Energy:", \
                        "Potential Energy   :", \
                        "Kinetic Energy     :", \
                        "E(X)               :"  , \
                        "E(C)               :"  , \
                        "Dispersion correction", \
                        "FINAL SINGLE POINT ENERGY"], 
                "PBE0" : ["Nuclear Repulsion  :", \
                          "One Electron Energy:", \
                          "Two Electron Energy:", \
                          "Potential Energy   :", \
                          "Kinetic Energy     :", \
                          "E(X)               :"  , \
                          "E(C)               :"  , \
                          "Dispersion correction", \
                          "FINAL SINGLE POINT ENERGY"] ,
                "ZORA" : ["Nuclear Repulsion  :", \
                          "One Electron Energy:", \
                          "Two Electron Energy:", \
                          "Potential Energy   :", \
                          "Kinetic Energy     :", \
                          "E(X)               :"  , \
                          "E(C)               :"  , \
                          "Dispersion correction", \
                          "FINAL SINGLE POINT ENERGY"],
                "TPSSh" : ["Nuclear Repulsion  :", \
                          "One Electron Energy:", \
                          "Two Electron Energy:", \
                          "Potential Energy   :", \
                          "Kinetic Energy     :", \
                          "E(X)               :"  , \
                          "E(C)               :"  , \
                          "Dispersion correction", \
                          "FINAL SINGLE POINT ENERGY"]
                }
    
    # read all the data and initialize the data structures
    rootdir = "../datasets/AllData/"   
    allvalues_perset, fullsetnames, models_results = \
        read_and_init (rootdir, supersetnames, howmanydifs, methods, \
                       DEBUG=False)

    # compute and dump summary statistics for each set precomputed methods
    for setname in fullsetnames:
        models_results[setname].inside_methods_rmse = []
        models_results[setname].inside_methods_r2 = []
        models_results[setname].our_methods_rmse = {}
        models_results[setname].our_methods_r2 = {}
        
        models_results[setname].bestinsidemethod_rmse = float("inf")
        models_results[setname].bestinsidemethod = ""
        models_results[setname].bestourmethod_rmse = float("inf")
        models_results[setname].bestourmethod = ""
        models_results[setname].our_methods_name = []

        for methodid in range(howmanydifs):
            y_pred = []
            labels = []
            for val in allvalues_perset[setname]:
                y_pred.append(val["label"] + val["difs"][methodid])
                labels.append(val["label"])
            
            r2 = r2_score(labels, y_pred)
            rmse = mean_squared_error(labels, y_pred, squared=False)
            models_results[setname].inside_methods_rmse.append(rmse)
            models_results[setname].inside_methods_r2.append(r2)
    
            if rmse < models_results[setname].bestinsidemethod_rmse:
                models_results[setname].bestinsidemethod_rmse = rmse
                models_results[setname].bestinsidemethod = str(methodid)

        for j, method in enumerate(methods):
            y_pred = []
            labels = []
            for val in allvalues_perset[setname]:
                y_pred.append(val[method + "_energydiff"][method+"_FINAL_SINGLE_POINT_ENERGY"])
                labels.append(val["label"])
            
            r2 = r2_score(labels, y_pred)
            rmse = mean_squared_error(labels, y_pred, squared=False)

            models_results[setname].our_methods_rmse[method] = rmse
            models_results[setname].our_methods_r2[method] = r2
            models_results[setname].our_methods_name.append(method)

            if rmse < models_results[setname].bestourmethod_rmse:
                models_results[setname].bestourmethod_rmse = rmse
                models_results[setname].bestourmethod = method

    # search top correlations and remove correlated features considering
    # all the sets together and the most important features
    for setname in fullsetnames:
        models_results[setname].fulldescriptors = []
        models_results[setname].labels = []
        for idx, val in enumerate(allvalues_perset[setname]):
            models_results[setname].fulldescriptors.append({})
            for method in methods:
                models_results[setname].fulldescriptors[idx].update(val[method+"_energydiff"])
    
            models_results[setname].labels.append(val["label"])
    
        moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (models_results[setname].fulldescriptors, \
                                         models_results[setname].labels)
    
        df = pd.DataFrame(moldescriptors_featues, columns=features_names)
    
        top_corr = commonutils.get_top_correlations_blog(df, CORRCUT)
    
        models_results[setname].top_correlation = top_corr
    
    if DEBUG:
        setname = "Full"
        print("Top correlations for set: ", setname)
        for tc in top_corr:
            print("  %35s %35s %9.3f"%(tc[0], tc[1], tc[2]))
        print("")

    # remove correlated features
    setname = "Full"
    featurestorm = set()
    for tc in top_corr:
        if tc[0] not in featurestorm:
            featurestorm.add(tc[1])
    if DEBUG:
        print("Features to remove: ")
        for f in featurestorm:
            print("  ", f)
        print("")
    allvalues_perset_orig =  deepcopy(allvalues_perset)
    for setname in fullsetnames:
        commonutils.remove_features_fromset(allvalues_perset[setname], \
                                            list(featurestorm), \
                                            methods)
        
    allvalues_perset_rmcorr = deepcopy(allvalues_perset)
    allvalues_perset = deepcopy(allvalues_perset_orig) 
    if DEBUG:
        for setname in fullsetnames:
            print("Set: ", setname, " Number of samples: ", \
                  len(allvalues_perset[setname]))
            for idx in range(len(allvalues_perset[setname])):
                print("  Sample: ", idx)
                pp.pprint(allvalues_perset_rmcorr[setname][idx])
                pp.pprint(allvalues_perset[setname][idx])
    for setname in fullsetnames:
        models_results[setname].fulldescriptors_rmcorr = []
        models_results[setname].labels_rmcorr = []
        for idx, val in enumerate(allvalues_perset_rmcorr[setname]):
            models_results[setname].fulldescriptors_rmcorr.append({})
            for method in methods:
                if method+"_energydiff" in val:
                    models_results[setname].fulldescriptors_rmcorr[idx].update(\
                        val[method+"_energydiff"])
    
            models_results[setname].labels_rmcorr.append(val["label"])
    if DEBUG:
        for setname in fullsetnames:
            print("Set: ", setname, " Number of samples: ", \
                  len(allvalues_perset[setname]))
            for idx in range(len(models_results[setname].fulldescriptors_rmcorr)):
                print("  Sample: ", idx)
                pp.pprint(models_results[setname].fulldescriptors_rmcorr[idx])
                pp.pprint(models_results[setname].fulldescriptors[idx])

    # compute NN global model using all features and using the less 
    # correlated ones 
    nepochs = [10, 50, 100]
    batch_sizes = [4, 8, 16, 32]
    modelshapes = [[4, 4], [8, 8], [16, 16], \
                [32, 32], [64, 64], [128, 128], \
                [4, 4, 4], [8, 8, 8], [16, 16, 16], \
                [32, 32, 32], [64, 64, 64], \
                [128, 128, 128], [4, 4, 4, 4], \
                [8, 8, 8, 8], [16, 16, 16, 16], \
                [32, 32, 32, 32], [64, 64, 64, 64], \
                [128, 128, 128, 128]]
    setname = "Full"
    moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (models_results[setname].fulldescriptors, \
                                    models_results[setname].labels)
    scalerx = preprocessing.StandardScaler().fit(moldescriptors_featues)
    moldescriptors_featues = scalerx.transform(moldescriptors_featues) 
    scalery = preprocessing.StandardScaler().fit(Y.reshape(-1, 1))
    Y = scalery.transform(Y.reshape(-1, 1))
    modelminrmse, modelsmaxr2 = \
        models.nn_model(0.2, moldescriptors_featues, Y, \
                    nepochs, modelshapes, batch_sizes, inputshape=-1,\
                    search=True)
    
    print("Best NN model for set: ", setname, file=sys.stderr)
    print("  RMSE: ", modelminrmse, file=sys.stderr)
    print("    R2: ", modelsmaxr2, file=sys.stderr)
    results = models.nn_model(0.2, moldescriptors_featues, Y, \
                    [modelminrmse[1]], \
                    [modelminrmse[0]], \
                    [modelminrmse[2]], \
                    inputshape=-1,\
                    search=False)
    
    print("NN model for set ", setname, file=sys.stderr)
    print("       RMSE train: ", results["rmse_train"], file=sys.stderr)
    print("        RMSE test: ", results["rmse_test"], file=sys.stderr)
    print("        RMSE full: ", results["rmse_full"], file=sys.stderr)    
    print("         R2 train: ", results["r2_train"], file=sys.stderr)
    print("          R2 test: ", results["r2_test"], file=sys.stderr)
    print("          R2 full: ", results["r2_full"], file=sys.stderr)

    y_pred = results["y_pred_full"] 
    labels = results["y_full"]
    y_pred = scalery.inverse_transform(y_pred)
    labels = scalery.inverse_transform(labels)

    rmse = mean_squared_error(labels, y_pred, squared=False)
    r2 = r2_score(labels, y_pred)
    print(" denorm RMSE full: ", rmse, file=sys.stderr)
    print("  denorm  R2 full: ", r2, file=sys.stderr)
    plt.clf()
    plt.plot(labels, y_pred, 'o') 
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig("NN_fullset.png")

    for setname in fullsetnames:
        print("Set: ", setname, file=sys.stderr)
        X, Y, features_names = \
            commonutils.build_XY_matrix (\
                models_results[setname].fulldescriptors, \
                                    models_results[setname].labels)
        X_t = scalerx.transform(X)
        Y_t = scalery.transform(Y.reshape(-1, 1))
        Y_pred_t = results["model"].predict(X_t)
        Y_pred = scalery.inverse_transform(Y_pred_t)
        models_results[setname].y_pred = Y_pred
        rmse_full =  mean_squared_error(Y, Y_pred, squared=False)
        r2_full = r2_score(Y, Y_pred)
        print("       Best inside method: ", models_results[setname].bestinsidemethod, file=sys.stderr)
        print("  Best inside method RMSE: ", models_results[setname].bestinsidemethod_rmse, file=sys.stderr)
        print("          Best our method: ", models_results[setname].bestourmethod, file=sys.stderr)
        print("     Best our method RMSE: ", models_results[setname].bestourmethod_rmse, file=sys.stderr)
        print("                RMSE full: ", rmse_full, file=sys.stderr)
        print("                  R2 full: ", r2_full, file=sys.stderr)
        plt.clf()
        plt.plot(Y, Y_pred, 'o') 
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.savefig("NN_"+setname+".png")     

    moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (\
                models_results[setname].fulldescriptors_rmcorr, \
                                    models_results[setname].labels_rmcorr)
    scalerx = preprocessing.StandardScaler().fit(moldescriptors_featues)
    moldescriptors_featues = scalerx.transform(moldescriptors_featues)
    scalery = preprocessing.StandardScaler().fit(Y.reshape(-1, 1))
    Y = scalery.transform(Y.reshape(-1, 1))
    print("Start Grid search ")
    modelminrmse, modelsmaxr2 = \
        models.nn_model(0.2, moldescriptors_featues, Y, \
                    nepochs, modelshapes, batch_sizes, inputshape=-1,\
                    search=True)
    
    print("Best NN model for set: ", setname, file=sys.stderr)
    print("  RMSE: ", modelminrmse, file=sys.stderr)
    print("    R2: ", modelsmaxr2, file=sys.stderr)

    results = models.nn_model(0.2, moldescriptors_featues, Y, \
                    [modelminrmse[1]], \
                    [modelminrmse[0]], \
                    [modelminrmse[2]], \
                    inputshape=-1,\
                    search=False)
    
    print("NN model for set ", setname, file=sys.stderr)
    print("       RMSE train: ", results["rmse_train"], file=sys.stderr)
    print("        RMSE test: ", results["rmse_test"], file=sys.stderr)
    print("        RMSE full: ", results["rmse_full"], file=sys.stderr)
    print("         R2 train: ", results["r2_train"], file=sys.stderr)
    print("          R2 test: ", results["r2_test"], file=sys.stderr)
    print("          R2 full: ", results["r2_full"], file=sys.stderr)

    y_pred = results["y_pred_full"] 
    labels = results["y_full"]
    y_pred = scalery.inverse_transform(y_pred)
    labels = scalery.inverse_transform(labels)

    rmse = mean_squared_error(labels, y_pred, squared=False)
    r2 = r2_score(labels, y_pred)
    print(" denorm RMSE full: ", rmse, file=sys.stderr)
    print("  denorm  R2 full: ", r2, file=sys.stderr)
    plt.clf()
    plt.plot(labels, y_pred, 'o') 
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig("NN_fullset_rmcorr.png")   

    for setname in fullsetnames:
        print("Set: ", setname, file=sys.stderr)
        X, Y, features_names = \
            commonutils.build_XY_matrix (\
                models_results[setname].fulldescriptors_rmcorr, \
                                    models_results[setname].labels_rmcorr)
        X_t = scalerx.transform(X)
        Y_t = scalery.transform(Y.reshape(-1, 1))
        Y_pred_t = results["model"].predict(X_t)
        Y_pred = scalery.inverse_transform(Y_pred_t)
        models_results[setname].y_pred_rmcorr = Y_pred
        rmse_full =  mean_squared_error(Y, Y_pred, squared=False)
        r2_full = r2_score(Y, Y_pred)
        print("       Best inside method: ", models_results[setname].bestinsidemethod, file=sys.stderr)
        print("  Best inside method RMSE: ", models_results[setname].bestinsidemethod_rmse, file=sys.stderr)
        print("          Best our method: ", models_results[setname].bestourmethod, file=sys.stderr)
        print("     Best our method RMSE: ", models_results[setname].bestourmethod_rmse, file=sys.stderr)
        print("                RMSE full: ", rmse_full, file=sys.stderr)
        print("                  R2 full: ", r2_full, file=sys.stderr)
        plt.clf()
        plt.plot(Y, Y_pred, 'o') 
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.savefig("NN_"+setname+"_rmcorr.png")   

    dump_predictions (fullsetnames, methods, \
                      allvalues_perset, models_results)
