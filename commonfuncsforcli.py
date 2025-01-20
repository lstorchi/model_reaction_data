import numpy as np
import commonutils
import pickle


from copy import deepcopy
from commonutils import ModelResults
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
#from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

PRINTALSOINSTDOUT = False
CUTDIFFPERCPLS = 0.9
CUTDIFFPERCLR = 0.001

###############################################################

def lr_test_and_rpint (lr_model, X, Y, name, features_names, fp,
                       shiftft="", fts=None):

    #y_pred_eq = lr_model.intercept_ + np.dot(X, lr_model.coef_.T)
    y_true = Y
    y_pred = lr_model.predict(X)
    if shiftft != "":
        y_true, y_pred = shiftbackdata(Y, y_pred, fts)
    rmse = 0.0
    try:
        rmse = root_mean_squared_error(Y, y_pred)
    except:
        rmse = mean_squared_error(Y, y_pred, squared=False)
    y_pred_eq = lr_model.get_intercept() + np.dot(X, lr_model.get_coefficients().T)
    if shiftft != "": 
        y_pred_eq = shiftbackdata(Y, y_pred_eq, fts)[1]
    rmse_eq = 0.0
    try:
        rmse_eq = root_mean_squared_error(Y, y_pred_eq)
    except:
        rmse_eq = mean_squared_error(Y, y_pred_eq, squared=False)
    diffperc = np.abs(rmse - rmse_eq) / rmse * 100.0
    if diffperc > CUTDIFFPERCLR:
        print("LR %50s RMSE Diff %5.3f "%(name, diffperc), "%")
        print("%50s RMSE Diff %5.3f "%(name, diffperc), "%", file=fp)
        exit(1)


    if PRINTALSOINSTDOUT:
        print("%50s , %30s , %20.12f"%(name, "Intercept", lr_model.get_intercept()))
    print("%50s , %30s , %20.12f"%(name, "Intercept", lr_model.get_intercept()), file=fp)
    for i, f in enumerate(features_names):
        #print("  %15s %12.8e"%(f, lr_model.coef_.T[i]))
        if PRINTALSOINSTDOUT:
            print("%50s , %30s , %20.12f"%(name, f, lr_model.get_coefficients().T[i]))
        print("%50s , %30s , %20.12f"%(name, f, lr_model.get_coefficients().T[i]), file=fp)

###############################################################

def pls_test_and_rpint (pls_model, X, Y, name, features_names, fp):
    
        y_pred = pls_model.predict(X)
        rmse = 0.0
        try:
            rmse = root_mean_squared_error(Y, y_pred)
        except:
            rmse = mean_squared_error(Y, y_pred, squared=False)
        X_e = X.copy()
        X_e -= pls_model._x_mean
        X_e /= pls_model._x_std
        y_pred_eq = np.dot(X_e, pls_model.coef_.T)
        y_pred_eq += pls_model._y_mean
        rmse_eq = 0.0
        try:
            rmse_eq = root_mean_squared_error(Y, y_pred_eq)
        except:
            rmse_eq = mean_squared_error(Y, y_pred_eq, squared=False)
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

def shiftbackdata (ytrue, ypred, fts):

    assert len(ytrue) == len(ypred)
    assert len(ytrue) == len(fts)    
   
    yt = ytrue.copy()
    for i in range(len(yt)):
        yt[i] += fts[i]

    yp = ypred.copy()
    for i in range(len(yp)):
        yp[i] += fts[i]
    
    return yt, yp

###############################################################

def readdata (removeFT="", shiftusingFT="", \
            selected_functionalin="PBE", \
            selected_basisin="SVP", \
                equations = {"EC" :"EC" ,\
                            "EX" : "EX",\
                            "FSPE" : "FINAL_SINGLE_POINT_ENERGY",\
                            "DC" : "Dispersion_correction",\
                            "PE" : "Potential_Energy",\
                            "KE" : "Kinetic_Energy",\
                            "OEE" : "One_Electron_Energy",\
                            "TEE" : "Two_Electron_Energy",\
                            "NR" : "Nuclear_Repulsion"}):

    if shiftusingFT != "":
        print("Shifting using ", shiftusingFT)
        removeFT = shiftusingFT

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
            allchems = ""
            for c in val["chemicals"]:
                allchems += c + " "
            models_results[setname].chemicals.append(allchems)

    insidemethods = ["W","D3(0)","D3(BJ)"]
    for setname in fullsetnames:
        for methodid in range(howmanydifs):
            methodname = insidemethods[methodid]
            models_results[setname].insidemethods_rmse[methodname] = float("inf")
            models_results[setname].insidemethods_mape[methodname] = float("inf")
            #models_results[setname].insidemethods_wtamd[methodname] = float("inf")
            models_results[setname].insidemethods_ypred[methodname] = []
    
            y_pred = []
            for val in allvalues_perset[setname]:
                y_pred.append(val["label"] + val["difs"][methodid])
    
            models_results[setname].insidemethods_ypred[methodname].extend(y_pred)
            
            rmse = 0.0
            try:
                rmse = root_mean_squared_error(models_results[setname].\
                    labels, \
                    y_pred)
            except:
                rmse = mean_squared_error(models_results[setname].\
                    labels, \
                    y_pred, 
                    squared=False)

            models_results[setname].insidemethods_rmse[methodname] = rmse
            
            mape = mean_absolute_percentage_error(models_results[setname].labels, y_pred)
            models_results[setname].insidemethods_mape[methodname] = mape
            
        for j, method in enumerate(methods):
            models_results[setname].funcional_basisset_rmse[method] = float("inf")
            models_results[setname].funcional_basisset_mape[method] = float
            #models_results[setname].funcional_basisset_wtamd[method] = float
            models_results[setname].funcional_basisset_ypred[method] = []
    
            y_pred = []
            for val in allvalues_perset[setname]:
                y_pred.append(val[method + "_energydiff"][method+"_FINAL_SINGLE_POINT_ENERGY"])
    
            models_results[setname].funcional_basisset_ypred[method].extend(y_pred)
    
            rmse = 0.0
            try:
                rmse = root_mean_squared_error(models_results[setname].labels,\
                    y_pred)
            except:
                rmse = mean_squared_error(models_results[setname].labels,\
                    y_pred, squared=False) 
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

    #for setname in featuresvalues_perset:
    #    print("Setname: ", setname, len(featuresvalues_perset[setname]))
    #    for i, val in enumerate(featuresvalues_perset[setname]):
    #        print("Val: ",len(val))
    #        for k in val:
    #            print("   ", k, val[k])

    eq_featuresvalues_perset =  \
        commonutils.equation_parser_compiler(equations, \
                                            functionals, \
                                            basis_sets, \
                                            basicfeattouse,\
                                            featuresvalues_perset, \
                                            warining=False)

    
    featuresvalues_perset = deepcopy(eq_featuresvalues_perset)

    nrperstename = {}
    ftperstename = {}
    for setname in featuresvalues_perset:
        nrperstename[setname] = []
        ftperstename[setname] = []
        if setname not in models_results:
            print("Setname: ", setname)
            print("Setname not in models_results: ", setname)
            exit(1)
    
        ftsforfb = {}
        chemicalsforfb = {}
        for func in ["PBE", "PBE0"]:
            for basis in ["MINIX", "SVP", "TZVP", "QZVP"]:
                fts = []
                chemicals = []
                for i, val in enumerate(featuresvalues_perset[setname]):
                    chemicals.append(models_results[setname].chemicals[i])
                    for k in val:
                        
                        if removeFT != "":
                            if k.find("_"+removeFT ) != -1 and \
                                k.find(func + "_") != -1 and \
                                k.find(basis + "_") != -1:
                                fts.append(val[k])
                            
                chemicalsforfb[func + "_" + basis] = chemicals
                ftsforfb[func + "_" + basis] = fts
    
        if removeFT != "":
            for k in ftsforfb:
                if len(ftsforfb[k]) != len(chemicalsforfb[k]):
                    print("Setname: ", setname)
                    print("FT values error")
                    print(k, len(ftsforfb[k]), len(chemicalsforfb[k]))
                    exit(1)
    
            for func1 in ["PBE", "PBE0"]:
                for basis1 in ["MINIX", "SVP", "TZVP", "QZVP"]:
                    for func2 in ["PBE", "PBE0"]:
                        for basis2 in ["MINIX", "SVP", "TZVP", "QZVP"]:
                            if func1 == func2 and basis1 == basis2:
                                continue
        
                            fts1 = ftsforfb[func1 + "_" + basis1]
                            chem1 = chemicalsforfb[func1 + "_" + basis1]
                            fts2 = ftsforfb[func2 + "_" + basis2]
                            chem2 = chemicalsforfb[func2 + "_" + basis2]
                            if len(fts1) != len(fts2):
                                print("Setname: ", setname)
                                print("FT len values error")
                                print(len(fts1), len(fts2), chem1, chem2)
                                exit(1)
        
                            for i in range(len(fts1)):
                                if np.abs(fts1[i] - fts2[i]) > 1e-6:
                                    print("Setname: ", setname)
                                    print(removeFT + " Error ", func1, \
                                        basis1, \
                                        " compare to ", \
                                        func2, \
                                        basis2, \
                                        "%8.5e"%(fts1[i]), \
                                        "%8.5e"%(fts2[i]), \
                                        " systems ", \
                                        chem1[i], \
                                        chem2[i])
                                    #exit(1)

        print ("Using values from ", selected_functionalin, " ", \
               selected_basisin) 
        ftperstename[setname] = ftsforfb[selected_functionalin+\
                                        "_"+ \
                                        selected_basisin]
        #print(len(nrperstename[setname]))


    if shiftusingFT != "":
        for setname in featuresvalues_perset: 
            assert len(ftperstename[setname]) == len(models_results[setname].labels)

            models_results[setname].fts = ftperstename[setname]

            if shiftusingFT != "":
                # shift labels using nrperstename
                for i, val in enumerate(ftperstename[setname]):
                    models_results[setname].labels[i] -= val

    if removeFT != "":   
        for setname in featuresvalues_perset:
            toremove = []
            for i, val in enumerate(featuresvalues_perset[setname]):
                for k in val:
                    if k.find("_"+removeFT ) != -1:
                        toremove.append((i, k))
            # remove starting from the end  
            toremove = sorted(toremove, key=lambda x: x[0], reverse=True)
            for i, k in toremove:
                del featuresvalues_perset[setname][i][k] 

    return featuresvalues_perset, \
        fullsetnames, models_results, \
        supersetnames

###############################################################
