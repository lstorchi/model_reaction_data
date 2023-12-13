import commonutils
import models

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

if __name__ == '__main__':

    DEBUG = False
    suprasetnames = {"BARRIER_HEIGHTS" : \
                       ["BH76","BHDIV10","BHPERI",\
                        "BHROT27","INV24","PX13","WCPT18"], \
                    "INTRAMOLECULAR_INTERACTIONS" : \
                       ["ADIM6","AHB21","CARBHB12",\
                        "CHB6","HAL59","HEAVY28","IL16",\
                        "PNICO23","RG18","S22","S66",\
                        "WATER27"] , \
                    "SMALL_MOLECULES" :\
                        ["AL2X6","ALK8","ALKBDE10","BH76",\
                         "DC13","DIPCS10","FH51","G21EA",\
                         "G21IP","G2RC","HEAVYSB11","NBPRC",\
                         "PA26","RC21","SIE4x4","TAUT15",\
                         "W4","11","YBDE18"], \
                    "INTERMOLECULAR_INTERACTIONS" :\
                       ["ADIM6","AHB21","CARBHB12",\
                        "CHB6","HAL59","HEAVY28","IL16",\
                        "PNICO23","RG18","S22","S66","WATER27"] , \
                    "LARGE_SYSTEMS" :\
                        ["BSR36","C60ISO","CDIE20","DARC",\
                         "ISO34","ISOL24","MB16","43","PArel",\
                            "RSE43]}"]}    
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
                          "FINAL SINGLE POINT ENERGY"]
                }
    
    allvalues_perset = {}
    allvalues = []
    
    toberemoved = []
    for i, setname in enumerate(setnames):
          print("Reading dataset: ", setname)
          rootdir = "../datasets/ML_data/" + setname
          labelsfilename = "../datasets/ML_data/"+setname+"/labels.txt"
    
          values =\
                commonutils.read_dataset(rootdir, labelsfilename, howmanydifs, methods,
                                         debug=True)
          
          if (values is None) or (len(values) <= 2):
                print(setname + " No data found for this dataset")
                print("")
                toberemoved.append(i)
          else:
                allvalues_perset[setname] = values  
                print("Number of samples: ", len(allvalues_perset[setname]))
                print("Number of basic PBE descriptors: ", len(allvalues_perset[setname]))
                print("Number of basic  HF descriptors: ", len(allvalues_perset[setname]))
          
                allvalues += allvalues_perset[setname]
                print("")
    
    for i in sorted(toberemoved, reverse=True):
          del setnames[i]
    
    if len(allvalues) > 0:
          allvalues_perset["Full"] = allvalues   
          setnames.append("Full")
    
    print("")
    print("%3s , %10s , "%("#", "SetName"), end="") 
    for methodid in range(howmanydifs):
        print("%6s , %8s , "%("R2 "+str(methodid), "RMSE "+str(methodid)), end="")
    for j, method in enumerate(methods):
        if j< len(methods)-1:
            print("%6s , %8s , "%("R2 "+method, "RMSE "+method), end="")
        else:
            print("%6s , %8s "%("R2 "+method, "RMSE "+method))

    for setname in setnames:
        print("%3d , %10s , "%(len(allvalues_perset[setname]), setname), end="")
        for methodid in range(howmanydifs):
            y_pred = []
            labels = []
            for val in allvalues_perset[setname]:
                y_pred.append(val["label"] + val["difs"][methodid])
                labels.append(val["label"])
            print("%6.3f , %8.3f , "%(r2_score(labels, y_pred), \
                                    mean_squared_error(labels, y_pred, squared=False)), end="")
        for j, method in enumerate(methods):
            y_pred = []
            labels = []
            for val in allvalues_perset[setname]:
                y_pred.append(val[method + "_energydiff"][method+"_FINAL_SINGLE_POINT_ENERGY"])
                labels.append(val["label"])
            if j< len(methods)-1:
                print("%6.3f , %8.3f , "%(r2_score(labels, y_pred), 
                                    mean_squared_error(labels, y_pred, squared=False)), end="")
            else:
                print("%6.3f , %8.3f"%(r2_score(labels, y_pred), 
                                    mean_squared_error(labels, y_pred, squared=False))) 
    
    CORRCUT = 1.0
    
    fulldescriptors = {}
    labels = {}
    top_correlation_perset = {}
    
    for setname in setnames:
        fulldescriptors[setname] = []
        labels[setname] = []
        for idx, val in enumerate(allvalues_perset[setname]):
            fulldescriptors[setname].append({})
            for method in methods:
                fulldescriptors[setname][idx].update(val[method+"_energydiff"])
    
            labels[setname].append(val["label"])
    
        moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (fulldescriptors[setname], labels[setname])
    
        df = pd.DataFrame(moldescriptors_featues, columns=features_names)
    
        top_corr = commonutils.get_top_correlations_blog(df, CORRCUT)
    
        top_correlation_perset[setname] = top_corr
        if DEBUG:
            print("Top correlations for set: ", setname)
            for tc in top_corr:
                print("%35s %35s %9.3f"%(tc[0], tc[1], tc[2]))
            print("")
    
    mostimportantefeatures_persetname = {}
    
    print("%10s , %4s , %9s , %9s , %9s , %9s , %9s , %9s , %9s , %9s"%(\
         "SetName", "Comp", "RMSETrain", "RMSETest", "RMSEFull", \
            "R2Train", "R2Test", "R2Full", "RMSELOO", "R2LOO"))
    for setname in setnames:
        mostimportantefeatures_persetname[setname] = []
        moldescriptors_featues, Y, features_names = \
        commonutils.build_XY_matrix (fulldescriptors[setname], \
                                     labels[setname])
    
        maxcomp = moldescriptors_featues.shape[1]
        # search fo the best number od components and build final model
        perc_split = 0.2
        ncomps, rmses_test, rmses_train, r2s_test, r2s_train = \
            models.pls_model (0.2, moldescriptors_featues, Y, \
                          ncomp_start = 1, ncomp_max = maxcomp)
        r2max_comps = np.argmax(r2s_test)+1
        rmsemin_comps = np.argmin(rmses_test)+1
        compstouse = min(rmsemin_comps, r2max_comps)
    
        perc_split = 0.2
        rmse_train, rmse_test, r2_train, r2_test, rmse_full, r2_full , \
            plsmodel, X_train, X_test, y_train, y_test  = \
                models.pls_model (0.2, moldescriptors_featues, Y, False, compstouse)
        perc_split = 0.0
        rmse, r2 = models.pls_model (perc_split, moldescriptors_featues, Y, False, \
                      compstouse, leaveoneout=True)
        
        print("%10s , %4d , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f"%(\
            setname, compstouse, \
            rmse_train, rmse_test, rmse_full, \
            r2_train, r2_test, r2_full, \
            rmse, r2))
    
        scoring = 'neg_mean_squared_error'
    
        r = permutation_importance(plsmodel, X_test, y_test, n_repeats=30, \
                                random_state=0, scoring=scoring)
        
        for i in r.importances_mean.argsort()[::-1]:
            mostimportantefeatures_persetname[setname].append(features_names[i])
    
        if DEBUG:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
            r_multi = permutation_importance(plsmodel, X_test, y_test, n_repeats=30, \
                                    random_state=0, scoring=scoring)
    
            for metric in r_multi:
                print(f"{metric}"+ " Used")
                r = r_multi[metric]
                for i in r.importances_mean.argsort()[::-1]:
                    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                        print(f"{features_names[i]:<30}"
                            f"{r.importances_mean[i]:.3e}"
                            f" +/- {r.importances_std[i]:.3e}")
                print("")
    
    features_to_remove_perset = {}
    for setname in setnames:
        features_to_remove_perset[setname] = []
        if DEBUG:
            print("Most important features for set: ", setname)
        for mif in mostimportantefeatures_persetname[setname]:
            if DEBUG:
                print("%35s"%(mif))
            for tc in top_correlation_perset[setname]:
                if tc not in mostimportantefeatures_persetname[setname]:
                    if mif == tc[0]:
                        features_to_remove_perset[setname].append(tc[1])
                        if DEBUG:
                            print("Corretlated %35s %9.3f"%(tc[1], tc[2]))
                    elif mif == tc[1]:
                        features_to_remove_perset[setname].append(tc[0])
                        if DEBUG:
                            print("Corretlated %35s %9.3f"%(tc[0], tc[2]))
    #remove some features based on importance and correlation
    for setname in setnames:
        commonutils.remove_features_fromset(allvalues_perset[setname], \
                                            features_to_remove_perset[setname], \
                                            methods)
    fulldescriptors = {}
    labels = {}
    
    for setname in setnames:
        fulldescriptors[setname] = []
        labels[setname] = []
        for idx, val in enumerate(allvalues_perset[setname]):
            fulldescriptors[setname].append({})
            for method in methods:
                fulldescriptors[setname][idx].update(val[method+"_energydiff"])
    
            labels[setname].append(val["label"])
    
        moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (fulldescriptors[setname], labels[setname])
    
    print("%10s , %4s , %9s , %9s , %9s , %9s , %9s , %9s , %9s , %9s"%(\
        "SetName", "Comp", "RMSETrain", "RMSETest", "RMSEFull", \
        "R2Train", "R2Test", "R2Full", "RMSELOO", "R2LOO"))
    for setname in setnames:
        mostimportantefeatures_persetname[setname] = []
        moldescriptors_featues, Y, features_names = \
        commonutils.build_XY_matrix (fulldescriptors[setname], \
                                     labels[setname])
    
        maxcomp = moldescriptors_featues.shape[1]
        # search fo the best number od components and build final model
        perc_split = 0.2
        ncomps, rmses_test, rmses_train, r2s_test, r2s_train = \
            models.pls_model (0.2, moldescriptors_featues, Y, \
                          ncomp_start = 1, ncomp_max = maxcomp)
        r2max_comps = np.argmax(r2s_test)+1
        rmsemin_comps = np.argmin(rmses_test)+1
        compstouse = min(rmsemin_comps, r2max_comps)
    
        perc_split = 0.2
        rmse_train, rmse_test, r2_train, r2_test, rmse_full, r2_full , \
            plsmodel, X_train, X_test, y_train, y_test  = \
                models.pls_model (0.2, moldescriptors_featues, Y, False, compstouse)
        perc_split = 0.0
        rmse, r2 = models.pls_model (perc_split, moldescriptors_featues, Y, False, \
                      compstouse, leaveoneout=True)
        
        print("%10s , %4d , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f , %9.3f"%(\
            setname, compstouse, \
            rmse_train, rmse_test, rmse_full, \
            r2_train, r2_test, r2_full, \
            rmse, r2))   