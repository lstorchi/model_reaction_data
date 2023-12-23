import commonutils
import models

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from dataclasses import dataclass
import prettyprinter as pp

from sklearn.cross_decomposition import PLSRegression
@dataclass
class ModelResults:
    # data considering all features
    pls_model: PLSRegression = None
    rmse_full: float = 0.0
    rmse_train: float = 0.0
    rmse_test: float = 0.0
    rmse_loo: float = 0.0
    r2_full: float = 0.0
    r2_train: float = 0.0
    r2_test: float = 0.0
    r2_loo: float = 0.0
    num_comp: int = 0
    fulldescriptors: list = None
    labels: list = None
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    top_correlation : list = None
    mostimportantefeatures : list = None
    features_to_remove : list = None
    # after removing correlated and 
    # less important features 
    pls_model_rmcorr: PLSRegression = None
    rmse_train_rmcorr : float = 0.0
    rmse_test_rmcorr : float = 0.0
    rmse_full_rmcorr : float = 0.0
    comp_rmcorr : int = 0
    fulldescriptors_rmcorr : list = None
    labels_rmcorr : list = None
    X_train_rmcorr : np.ndarray = None
    X_test_rmcorr : np.ndarray = None
    y_train_rmcorr : np.ndarray = None
    y_test_rmcorr : np.ndarray = None
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


if __name__ == '__main__':

    MODELTYPE = "PLS"
    DEBUG = False
    OUTSUMMARY = True
    CORRCUT = 0.99
    suprasetnames = {"BARRIER_HEIGHTS" : \
                       ["BH76","BHDIV10","BHPERI",\
                        "BHROT27","INV24","PX13","WCPT18"], \
                    "INTRAMOLECULAR_INTERACTIONS" : \
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
    
    allvalues_perset = {}
    fullsetnames = []
    models_results = {}

    toberemoved = {}
    for supersetname in suprasetnames:
        toberemoved[supersetname] = []
        allvalues_perset[supersetname] = []
        fullsetnames.append(supersetname)
        for i, setname in enumerate(suprasetnames[supersetname]):
              print("Reading dataset: ", setname)
              rootdir = "../datasets/AllData/" + supersetname + "/" +setname
              labelsfilename = "../datasets/AllData/" +setname +"_labels.txt"
        
              values =\
                    commonutils.read_dataset(rootdir, labelsfilename, howmanydifs, methods,
                                             debug=DEBUG)
              
              if (values is None) or (len(values) <= 2):
                    print(setname + " No data found for this dataset")
                    print("")
                    toberemoved[supersetname].append(i)
              else:
                    fullsetname = supersetname+"_"+setname
                    fullsetnames.append(fullsetname)
                    allvalues_perset[fullsetname] = values  
                    print("Number of samples: ", len(allvalues_perset[fullsetname]))
                    print("Number of basic descriptors: ", len(allvalues_perset[fullsetname]))
              
                    allvalues_perset[supersetname] += allvalues_perset[fullsetname]
                    print("")

    for supersetname in toberemoved:
        for i in sorted(toberemoved[supersetname], reverse=True):
          del suprasetnames[supersetname][i]
    
    allvalues_perset["Full"] = []
    for supersetname in suprasetnames:
          allvalues_perset["Full"] += allvalues_perset[supersetname]  
    fullsetnames.append("Full")

    for setname in fullsetnames:
        models_results[setname] = ModelResults()

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
            print("Top correlations for set: ", setname)
            for tc in top_corr:
                print("%35s %35s %9.3f"%(tc[0], tc[1], tc[2]))
            print("")

    for setname in fullsetnames:
        moldescriptors_featues, Y, features_names = \
            commonutils.build_XY_matrix (models_results[setname].fulldescriptors, \
                                    models_results[setname].labels)
    
        # search fo the best number od components and build final model
        perc_split = 0.2

        if MODELTYPE == "PLS":
            maxcomp = moldescriptors_featues.shape[1]
            ncomps, rmses_test, rmses_train, r2s_test, r2s_train = \
                models.pls_model (0.2, moldescriptors_featues, Y, \
                          ncomp_start = 1, ncomp_max = maxcomp)
        
            r2max_comps = np.argmax(r2s_test)+1
            rmsemin_comps = np.argmin(rmses_test)+1
            compstouse = min(rmsemin_comps, r2max_comps)

            models_results[setname].num_comp = compstouse

            models_results[setname].rmse_train, \
            models_results[setname].rmse_test, \
            models_results[setname].r2_train, \
            models_results[setname].r2_test, \
            models_results[setname].rmse_full, \
            models_results[setname].r2_full , \
            models_results[setname].pls_model, \
            models_results[setname].X_train, \
            models_results[setname].X_test, \
            models_results[setname].y_train, \
            models_results[setname].y_test  = \
                    models.pls_model (perc_split, moldescriptors_featues, \
                                      Y, False, compstouse)
            
            perc_split = 0.0

            models_results[setname].rmse_loo, \
            models_results[setname].r2_loo = \
                models.pls_model (perc_split, moldescriptors_featues,\
                                   Y, False, compstouse, leaveoneout=True)
    
            scoring = 'neg_mean_squared_error'
            models_results[setname].mostimportantefeatures = []
    
            r = permutation_importance(models_results[setname].pls_model,\
                                models_results[setname].X_test, \
                                models_results[setname].y_test, \
                                n_repeats=30, \
                                random_state=0, scoring=scoring)
        
            for i in r.importances_mean.argsort()[::-1]:
                models_results[setname].mostimportantefeatures.append(\
                    features_names[i])
    
            if DEBUG:
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
                r_multi = permutation_importance(\
                    models_results[setname].pls_model,\
                    models_results[setname].X_test, \
                    models_results[setname].y_test, \
                    n_repeats=30, random_state=0, \
                    scoring=scoring)
    
                for metric in r_multi:
                    print(f"{metric}"+ " Used")
                    r = r_multi[metric]
                    for i in r.importances_mean.argsort()[::-1]:
                        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                            print(f"{features_names[i]:<30}"
                                f"{r.importances_mean[i]:.3e}"
                                f" +/- {r.importances_std[i]:.3e}")
                    print("")
    
    for setname in fullsetnames:
        models_results[setname].features_to_remove = set()
        if DEBUG:
            print("Most important features for set: ", setname)
        for mif in models_results[setname].mostimportantefeatures:
            if DEBUG:
                print("%35s"%(mif))
            for tc in models_results[setname].top_correlation:
                if mif == tc[0]:
                    models_results[setname].features_to_remove.add(tc[1])
                    if DEBUG:
                        print("Corretlated %35s %9.3f"%(tc[1], tc[2]))
                elif mif == tc[1]:
                    models_results[setname].features_to_remove.add(tc[0])
                    if DEBUG:
                        print("Corretlated %35s %9.3f"%(tc[0], tc[2]))
   
    #remove some features based on importance and correlation
    DEBUG = True
    for setname in fullsetnames:
        if DEBUG:
            print(setname, len(models_results[setname].features_to_remove))
            print(models_results[setname].X_train.shape)
        commonutils.remove_features_fromset(allvalues_perset[setname], \
                                            list(models_results[setname].features_to_remove), \
                                            methods)
        
    # models using non correlated and most important features
    for setname in fullsetnames:
        models_results[setname].fulldescriptors_rmcorr = []
        models_results[setname].labels_rmcorr = []
        for idx, val in enumerate(allvalues_perset[setname]):
            models_results[setname].fulldescriptors_rmcorr.append({})
            for method in methods:
                models_results[setname].fulldescriptors_rmcorr[idx].update(val[method+"_energydiff"])
    
            models_results[setname].labels_rmcorr.append(val["label"])
    
    for setname in fullsetnames:
        moldescriptors_featues, Y, features_names = \
        commonutils.build_XY_matrix (models_results[setname].fulldescriptors_rmcorr, \
                                     models_results[setname].labels_rmcorr)
    
        # search fo the best number od components and build final model
        perc_split = 0.2

        if MODELTYPE == "PLS":
            maxcomp = moldescriptors_featues.shape[1]
            ncomps, rmses_test, rmses_train, r2s_test, r2s_train = \
                models.pls_model (perc_split, moldescriptors_featues, Y, \
                          ncomp_start = 1, ncomp_max = maxcomp)
        
            r2max_comps = np.argmax(r2s_test)+1
            rmsemin_comps = np.argmin(rmses_test)+1
            compstouse = min(rmsemin_comps, r2max_comps)

            models_results[setname].num_comp_rmcorr = compstouse
            
            models_results[setname].rmse_train_rmcorr, \
            models_results[setname].rmse_test_rmcorr, \
            models_results[setname].r2_train_rmcorr, \
            models_results[setname].r2_test_rmcorr, \
            models_results[setname].rmse_full_rmcorr, \
            models_results[setname].r2_full_rmcorr , \
            models_results[setname].pls_model_rmcorr, \
            models_results[setname].X_train_rmcorr, \
            models_results[setname].X_test_rmcorr, \
            models_results[setname].y_train_rmcorr, \
            models_results[setname].y_test_rmcorr = \
                    models.pls_model (perc_split, moldescriptors_featues,\
                                       Y, False, compstouse)
            
            perc_split = 0.0

            models_results[setname].rmse_loo_rmcorr, \
            models_results[setname].r2_loo_rmcorr = \
                models.pls_model (perc_split, moldescriptors_featues, \
                    Y, False, compstouse, leaveoneout=True)

    if not OUTSUMMARY:
        for setname in fullsetnames:
            print("Setname: ", setname)
            pp.pprint(models_results[setname])
    else:
        fp = open("summary.csv", "w")
        fpgood = open("summary_good.csv", "w")
        fpbad = open("summary_bad.csv", "w")

        for f in [fp, fpgood, fpbad]:
            print("# , " + \
                "setname , " + \
                "rmse_best_inside , " + \
                "rmse_best_our , " + \
                "rmse_full , " + \
                "rmse_full_rmcorr , " + \
                "comp , " + \
                "comp_rmcorr , " + \
                "method_best_inside , " + \
                "method_best_our ", file=f)
        
        for setname in fullsetnames:
            dim = len(allvalues_perset[setname])
            print("%d , "%(dim) + \
                "%s , "%(setname) + \
                "%9.3f , "%(models_results[setname].bestinsidemethod_rmse) + \
                "%9.3f , "%(models_results[setname].bestourmethod_rmse) + \
                "%9.3f , "%(models_results[setname].rmse_full) + \
                "%9.3f , "%(models_results[setname].rmse_full_rmcorr) + \
                "%d , "%(models_results[setname].num_comp) + \
                "%d , "%(models_results[setname].num_comp_rmcorr) + \
                "%s , "%(models_results[setname].bestinsidemethod) + \
                "%s "%(models_results[setname].bestourmethod), file=fp)
            if models_results[setname].rmse_full <  models_results[setname].bestinsidemethod_rmse or \
               models_results[setname].rmse_full_rmcorr <  models_results[setname].bestinsidemethod_rmse:
                print("%d , "%(dim) + \
                    "%s , "%(setname) + \
                    "%9.3f , "%(models_results[setname].bestinsidemethod_rmse) + \
                    "%9.3f , "%(models_results[setname].bestourmethod_rmse) + \
                    "%9.3f , "%(models_results[setname].rmse_full) + \
                    "%9.3f , "%(models_results[setname].rmse_full_rmcorr) + \
                    "%d , "%(models_results[setname].num_comp) + \
                    "%d , "%(models_results[setname].num_comp_rmcorr) + \
                    "%s , "%(models_results[setname].bestinsidemethod) + \
                    "%s "%(models_results[setname].bestourmethod), file=fpgood)
            else:
                print("%d , "%(dim) + \
                    "%s , "%(setname) + \
                    "%9.3f , "%(models_results[setname].bestinsidemethod_rmse) + \
                    "%9.3f , "%(models_results[setname].bestourmethod_rmse) + \
                    "%9.3f , "%(models_results[setname].rmse_full) + \
                    "%9.3f , "%(models_results[setname].rmse_full_rmcorr) + \
                    "%d , "%(models_results[setname].num_comp) + \
                    "%d , "%(models_results[setname].num_comp_rmcorr) + \
                    "%s , "%(models_results[setname].bestinsidemethod) + \
                    "%s "%(models_results[setname].bestourmethod), file=fpbad)
            
        fp.close()
        fpgood.close()
        fpbad.close()

        for superset in suprasetnames:
            fp = open(superset + "_summary.csv", "w")

            print("# , " + \
                    "setname , " + \
                    "rmse_best_inside , " + \
                    "rmse_best_our , " + \
                    "rmse_full , " + \
                    "rmse_full_rmcorr , " + \
                    "comp , " + \
                    "comp_rmcorr , " + \
                    "method_best_inside , " + \
                    "method_best_our ", file=fp)
            
            dim = len(allvalues_perset[superset])
            print("%d , "%(dim) + \
                "%s , "%(superset) + \
                "%9.3f , "%(models_results[superset].bestinsidemethod_rmse) + \
                "%9.3f , "%(models_results[superset].bestourmethod_rmse) + \
                "%9.3f , "%(models_results[superset].rmse_full) + \
                "%9.3f , "%(models_results[superset].rmse_full_rmcorr) + \
                "%d , "%(models_results[superset].num_comp) + \
                "%d , "%(models_results[superset].num_comp_rmcorr) + \
                "%s , "%(models_results[superset].bestinsidemethod) + \
                "%s "%(models_results[superset].bestourmethod), file=fp)
            
            for subset in suprasetnames[superset]:
                setname = superset + "_" + subset
                dim = len(allvalues_perset[setname])
                print("%d , "%(dim) + \
                    "%s , "%(setname) + \
                    "%9.3f , "%(models_results[setname].bestinsidemethod_rmse) + \
                    "%9.3f , "%(models_results[setname].bestourmethod_rmse) + \
                    "%9.3f , "%(models_results[setname].rmse_full) + \
                    "%9.3f , "%(models_results[setname].rmse_full_rmcorr) + \
                    "%d , "%(models_results[setname].num_comp) + \
                    "%d , "%(models_results[setname].num_comp_rmcorr) + \
                    "%s , "%(models_results[setname].bestinsidemethod) + \
                    "%s "%(models_results[setname].bestourmethod), file=fp)
            
            fp.close()