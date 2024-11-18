import numpy as np
import commonutils
import pickle


from copy import deepcopy
from commonutils import ModelResults
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

PRINTALSOINSTDOUT = False
CUTDIFFPERCPLS = 0.9
CUTDIFFPERCLR = 0.001

###############################################################

def lr_test_and_rpint (lr_model, X, Y, name, features_names, fp):

    #y_pred_eq = lr_model.intercept_ + np.dot(X, lr_model.coef_.T)
    y_pred = lr_model.predict(X)
    rmse = root_mean_squared_error(Y, y_pred)
    y_pred_eq = lr_model.get_intercept() + np.dot(X, lr_model.get_coefficients().T)
    rmse_eq = root_mean_squared_error(Y, y_pred_eq)
    diffperc = np.abs(rmse - rmse_eq) / rmse * 100.0
    if diffperc > CUTDIFFPERCLR:
        print("LR %40s RMSE Diff %5.3f "%(name, diffperc), "%")
        print("%40s RMSE Diff %5.3f "%(name, diffperc), "%", file=fp)
        exit(1)


    if PRINTALSOINSTDOUT:
        print("%40s , %30s , %12.4f"%(name, "Intercept", lr_model.get_intercept()))
    print("%40s , %30s , %12.4f"%(name, "Intercept", lr_model.get_intercept()), file=fp)
    for i, f in enumerate(features_names):
        #print("  %15s %12.8e"%(f, lr_model.coef_.T[i]))
        if PRINTALSOINSTDOUT:
            print("%40s , %30s , %12.4f"%(name, f, lr_model.get_coefficients().T[i]))
        print("%40s , %30s , %12.4f"%(name, f, lr_model.get_coefficients().T[i]), file=fp)

###############################################################

def pls_test_and_rpint (pls_model, X, Y, name, features_names, fp):
    
        y_pred = pls_model.predict(X)
        rmse = root_mean_squared_error(Y, y_pred)
        X_e = X.copy()
        X_e -= pls_model._x_mean
        X_e /= pls_model._x_std
        y_pred_eq = np.dot(X_e, pls_model.coef_.T)
        y_pred_eq += pls_model._y_mean
        rmse_eq = root_mean_squared_error(Y, y_pred_eq)
        diffperc = np.abs(rmse - rmse_eq) / rmse * 100.0
        if diffperc > CUTDIFFPERCPLS:
            print("PLS %40s RMSE Diff %5.3f "%(name, diffperc), "%")
            print("%40s RMSE Diff %5.3f "%(name, diffperc), "%", file=fp)
            exit(1)
            
        for i, f in enumerate(features_names):

            if PRINTALSOINSTDOUT:
                print("%40s , %30s , %12.4f , %16.4f , %16.4f "%(\
                        name, f,\
                        pls_model.coef_.T[i],\
                        pls_model._x_mean[i],\
                        pls_model._x_std[i]))
            print("%40s , %30s , %12.4f , %16.4f , %16.4f "%(\
                    name, f,\
                    pls_model.coef_.T[i],\
                    pls_model._x_mean[i],\
                    pls_model._x_std[i]), file=fp)


###############################################################

def readdata ():

    howmanydifs = 3
    allvalues_perset = pickle.load(open("./data/allvalues_perset.p", "rb"))
    methods = pickle.load(open("./data/methods.p", "rb"))
    fullsetnames = pickle.load(open("./data/fullsetnames.p", "rb"))
    functionals = pickle.load(open("./data/functionals.p", "rb"))
    basis_sets = pickle.load(open("./data/basis_sets.p", "rb"))
    supersetnames = pickle.load(open("./data/supersetnames.p", "rb"))

    print("Printing also to stdout: ", PRINTALSOINSTDOUT) 
    
    allfeatures = set()
    for setname in fullsetnames:
        for val in allvalues_perset[setname]:
            for k in val:
                if k.find("energydiff") != -1:
                    for f in val[k]:
                        allfeatures.add(f)
    
    # set labels and sets iists
    models_results = {}
    for setname in fullsetnames:
        models_results[setname] = ModelResults()
        for val in allvalues_perset[setname]:
            models_results[setname].labels.append(val["label"]) 
            models_results[setname].supersetnames.append(val["super_setname"])
            models_results[setname].setnames.append(val["super_setname"]+"_"+val["setname"])
    
    insidemethods = ["W","D3(0)","D3(BJ)"]
    for setname in fullsetnames:
        for methodid in range(howmanydifs):
            methodname = insidemethods[methodid]
            models_results[setname].insidemethods_rmse[methodname] = float("inf")
            models_results[setname].insidemethods_mape[methodname] = float("inf")
            models_results[setname].insidemethods_wtamd[methodname] = float("inf")
            models_results[setname].insidemethods_ypred[methodname] = []
    
            y_pred = []
            for val in allvalues_perset[setname]:
                y_pred.append(val["label"] + val["difs"][methodid])
    
            models_results[setname].insidemethods_ypred[methodname].extend(y_pred)
    
            wtmad = None
            fulllist = list(supersetnames.keys()) + ["Full"]
            if setname in fulllist:
                wtmadf = commonutils.wtmad2(models_results[setname].\
                        setnames,  \
                        models_results[setname].labels, y_pred)
                wtmad = wtmadf[setname]
                models_results[setname].insidemethods_wtamd[methodname] = wtmad
    
            rmse = root_mean_squared_error(models_results[setname].\
                    labels, \
                    y_pred)
            models_results[setname].insidemethods_rmse[methodname] = rmse
            
            mape = mean_absolute_percentage_error(models_results[setname].labels, y_pred)
            models_results[setname].insidemethods_mape[methodname] = mape
            
        for j, method in enumerate(methods):
            models_results[setname].funcional_basisset_rmse[method] = float("inf")
            models_results[setname].funcional_basisset_mape[method] = float
            models_results[setname].funcional_basisset_wtamd[method] = float
            models_results[setname].funcional_basisset_ypred[method] = []
    
            y_pred = []
            for val in allvalues_perset[setname]:
                y_pred.append(val[method + "_energydiff"][method+"_FINAL_SINGLE_POINT_ENERGY"])
    
            models_results[setname].funcional_basisset_ypred[method].extend(y_pred)
    
            wtmad = None            
            fulllist = list(supersetnames.keys()) + ["Full"]
            if setname in fulllist:
                wtmadf = commonutils.wtmad2(models_results[setname].\
                        setnames,\
                        models_results[setname].labels, y_pred)
                wtmad = wtmadf[setname]
                models_results[setname].funcional_basisset_wtamd[method] = wtmad
    
            rmse = root_mean_squared_error(models_results[setname].labels,\
                    y_pred)
            models_results[setname].funcional_basisset_rmse[method] = rmse
    
            mape = mean_absolute_percentage_error(models_results[setname].labels, y_pred)
            models_results[setname].funcional_basisset_mape[method] = mape
            
    
    basicfeattouse = ["Potential_Energy",\
                    "Kinetic_Energy",\
                    "FINAL_SINGLE_POINT_ENERGY",\
                    "Dispersion_correction",\
                    "E(C)",\
                    "E(X)",\
                    "Two_Electron_Energy",\
                    "Nuclear_Repulsion",\
                    "One_Electron_Energy"]
    
    featuresvalues_perset = {}
    for setname in fullsetnames:
        featuresvalues_perset [setname] = []
        for val in allvalues_perset[setname]:
            featuresvalues_perset[setname].append({})
            for k in val:
                if k.find("energydiff") != -1:
                    torm = k.replace("energydiff", "")
                    for f in val[k]:
                        tocheck = f.replace(torm, "")
                        if tocheck in basicfeattouse:
                            keytouse = f.replace("-", "_")
                            keytouse = keytouse.replace("(", "")
                            keytouse = keytouse.replace(")", "")
                            featuresvalues_perset[setname][-1][keytouse] = val[k][f]
    
    
    equations = {"EC" :"EC" ,\
                "EX" : "EX",\
                "FSPE" : "FINAL_SINGLE_POINT_ENERGY",\
                "DC" : "Dispersion_correction",\
                "PE" : "Potential_Energy",\
                "KE" : "Kinetic_Energy",\
                "OEE" : "One_Electron_Energy",\
                "TEE" : "Two_Electron_Energy",\
                "NR" : "Nuclear_Repulsion"}
    eq_featuresvalues_perset =  \
        commonutils.equation_parser_compiler(equations, \
                                            functionals, \
                                            basis_sets, \
                                            basicfeattouse,\
                                            featuresvalues_perset, \
                                            warining=False)

    
    featuresvalues_perset = deepcopy(eq_featuresvalues_perset)

    return featuresvalues_perset, \
        fullsetnames, models_results, \
        supersetnames

###############################################################
