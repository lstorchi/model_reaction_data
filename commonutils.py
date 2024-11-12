from contextlib import contextmanager,redirect_stderr,redirect_stdout
from dataclasses import dataclass
from dataclasses import field
from os import devnull

import numpy as np
import pandas as pd
import math 
import os 
import re

import token
import tokenize

from io import StringIO

from numpy import exp, sqrt, fabs, log, power, multiply, divide

####################################################################################################
@dataclass(slots=True)
class ModelResults:
    setnames : list = field(default_factory=list)
    supersetnames : list = field(default_factory=list)
    labels: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    uncorrelated_features: dict = field(default_factory=dict)

    funcional_basisset_rmse: dict = field(default_factory=dict)
    funcional_basisset_wtamd: dict = field(default_factory=dict)
    funcional_basisset_mape : dict = field(default_factory=dict)
    funcional_basisset_ypred : dict = field(default_factory=dict)

    insidemethods_rmse: dict = field(default_factory=dict)
    insidemethods_wtamd: dict = field(default_factory=dict)
    insidemethods_mape : dict = field(default_factory=dict)
    insidemethods_ypred : dict = field(default_factory=dict)
@dataclass(slots=False)
class ModelsStore:
    
    plsmodel = None
    plsmodel_splitted = None
    lr_model = None
    lr_model_splitted = None
    lr_custom_model = None
    lr_custom_model_splitted = None

####################################################################################################

def equation_parser_compiler (equations, functionals, basis_sets, basicfeattouse, \
                              featuresvalues_perset, warining=True):
    eq_featuresvalues_perset = {}

    for setname in featuresvalues_perset:
        #print("Equations for ", setname , " set ", len(featuresvalues_perset[setname]))

        eq_featuresvalues_perset[setname] = []
        for entry in featuresvalues_perset[setname]:
            eq_featuresvalues_perset[setname].append({})

        for func in functionals:
            for basis in basis_sets:
                #print(func + "_" + basis)
                touseforequation = {}
                for k in basicfeattouse:
                    ktouse = k.replace("(", "")
                    ktouse = ktouse.replace(")", "")
                    touseforequation[ktouse] = []
                for entry in featuresvalues_perset[setname]:
                    #print(len(entry))
                    for propname in entry:
                        if propname.find(func + "_" + basis) != -1:  
                            newk = propname.replace(func + "_" + basis + "_", "")
                            #print(newk, propname, entry[propname])
                            touseforequation[newk].append(entry[propname])
                maxdim = 0
                for k in touseforequation:
                    if len(touseforequation[k]) > maxdim:
                        maxdim = len(touseforequation[k])
                torm = []
                for k in touseforequation:
                    if len(touseforequation[k]) < maxdim:
                        torm.append(k)
                for k in torm:
                    del touseforequation[k]
                dtouseforequation = pd.DataFrame(touseforequation)

                for eqname in equations:
                    eq = equations[eqname]
                    sio = StringIO(eq)
                    tokens = tokenize.generate_tokens(sio.readline)

                    variables = []
                    for toknum, tokval, _, _, _  in tokens:
                        if toknum == token.NAME:
                            #print(tokval)
                            if (tokval != "exp") and (tokval != "sqrt") \
                                and (tokval != "fabs") and (tokval != "log") \
                                and (tokval != "power") and (tokval != "multiply") \
                                and (tokval != "sum") and (tokval != "divide"):
                                variables.append(tokval)
                                if not (tokval in dtouseforequation.columns):
                                    if warining:
                                        print("Warning ", tokval, \
                                          " not in or undefined function ", \
                                            func, basis)
                    exettherest = True
                    for var in variables:
                        if not (var in dtouseforequation.columns):
                            if warining:
                                print("Warning ", var, 
                                  " not in or undefined function ", func, basis)
                            exettherest = False

                    if exettherest:
                        toexe = ""
                        for vname in variables:
                            toexe += vname + " = np.array(dtouseforequation[\""+vname+"\"].tolist())"
                            toexe += "\n" 
                    
                        exec(toexe) 
                        toexe = eqname +" = " + eq
                        #print(toexe)
                        exec(eqname +" = " + eq)
                    
                        keyname = func + "_" + basis + "_" + eqname
                        toexe = "for idx in range(len(" + eqname + ")): \n" + \
                                "  #print(idx)\n" + \
                                "  value = float(" + eqname + "[idx])\n" + \
                                "  eq_featuresvalues_perset[setname][idx][\""+keyname+"\"] = value"
                        #print(toexe)
                        exec(toexe)

    return eq_featuresvalues_perset    

####################################################################################################

def remove_features_fromset (allvalues, features_to_remove,
                             methods):
    """
    for i, val in enumerate(allvalues):
        for method in methods:
            fs = method+"_energydiff"
            print("BEFORE ", i, fs, len(allvalues[i][fs]))
    """

    for i, val in enumerate(allvalues):
        for method in methods:
            fs = method+"_energydiff"
            for ftr in features_to_remove:
                if (ftr in val[fs]):
                    #print("Removing", ftr, "from", fs)
                    del allvalues[i][fs][ftr]
    
    """
    for i, val in enumerate(allvalues):
        for method in methods:
            fs = method+"_energydiff"
            print("AFTER ", i, fs, len(allvalues[i][fs]))
    """

    return allvalues

####################################################################################################

def remove_features (allvalues, features_to_remove, featuressets):

    for val in allvalues:
        for fs in featuressets:
            for ftr in features_to_remove:
                if (ftr in val[fs]):
                    print("Removing", ftr, "from", fs)
                    del val[fs][ftr]

    return

####################################################################################################

def read_dataset (rootdir, labelfilename, howmanydifs, methods, debug=True):

    autokcalmol = 627.5096080305927

    allvalues = []
    
    fp = open (labelfilename, 'r')
    for line in fp:
        sline = line.replace("\t", " ").replace("\n", "").rstrip().lstrip().split()

        stechio_ceofs = []
        chemicals = []

        all = []
        for j in range(1, len(sline)-howmanydifs-1):
            all.append(sline[j])

        if len(all) % 2 != 0:
            if debug:
                print("Error: len(all) % 2 != 0")
                print("file format error in line:", line)
                print("chemicals differ from stechio_ceofs")
                print(line)
            return None
        
        for i in range(0, int(len(all)/2)):
            #appysome filters
            tostore = all[i]
            #tostore = tostore.replace(",", "")
            #tostore = tostore.replace("/$A", "")
            chemicals.append(tostore)

        for i in range(int(len(all)/2), len(all)):
            stechio_ceofs.append(int(all[i]))

        if len(sline) < howmanydifs+2:
            if debug:
                print("Error: len(sline) < howmanydifs+2")
                print(line)
        else:
            label = float(sline[-1*howmanydifs-1])  
            difs = []
            for i in range(howmanydifs):
                difs.append(float(sline[-1*(i+1)]))
            difs.reverse()
        
            allvalues.append({"stechio_ceofs" : stechio_ceofs, \
                              "chemicals" : chemicals, \
                              "label" : label, \
                              "difs" : difs})

    fp.close()

    for method in methods:
        descriptor = {}
        first = True
        desclist = []
        for file in os.listdir(rootdir+'/'+method+'/'):
            if file.endswith('.out'):
                molname = file.split('.out')[0]
                molname = re.split("\.mpi\d+", molname)[0]
                #if re.search("S\d+", molname):
                #    molname = molname.replace("S", "")
                moldesc = {}
                fp = open(rootdir+'/'+method+'/'+file, 'r')
                toinsert = True
                for line in fp:
                    for val in methods[method]:
                        if line.find(val[0]) != -1:
                            keyval = val[1]
                            keyval = keyval.rstrip().lstrip().replace(" ", "_")
                            #if val.find(":") != -1:
                            #    keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                            #elif val.find("=") != -1:
                            #    keyval = val.replace("=", "").rstrip().lstrip().replace(" ", "_")
                            sline = line.rstrip().lstrip().split()
                            for sval in sline:
                                try:
                                    firstnumvalue = float(sval)
                                    break
                                except:
                                    continue
                            
                            moldesc[method+"_"+keyval] = firstnumvalue
                            #print(molname, keyval, sval, sline)
                            #exit(1)
                fp.close()
                if first:
                    first = False
                    desclist = list(moldesc.keys())
                else:
                    if desclist != list(moldesc.keys()):
                        toinsert = False
                        if debug:
                            print("Error: desclist != list(moldesc.keys())")
                            print("Looking for ",desclist)
                            print("Found ", moldesc.keys())
                            print("Cannot find features in Molname:", \
                                  rootdir+'/'+method+'/'+file)
                        #return None
       
                if toinsert:    
                    descriptor[molname] = moldesc
    
        for i, val in enumerate(allvalues):
       
            energydiff = {}
       
            for desc in set(desclist):
                sum = 0.0
                for j, chemical in enumerate(val["chemicals"]):
                    if chemical not in descriptor:
                        if debug:
                            print(chemical + " not found in PBE descriptors")
                        sum = float("nan")
                        break
                    else:
                        sum += val["stechio_ceofs"][j]*descriptor[chemical][desc]
                energydiff[desc] = sum*autokcalmol
                  
       
            allvalues[i][method+"_energydiff"] = energydiff

        # check if label or values in descriptor is nan
        idxtoremovs = []
        for i, val in enumerate(allvalues):
            if math.isnan(val["label"]):
                idxtoremovs.append(i)
            else:
                for k,v in val[method+"_energydiff"].items():
                    if math.isnan(v):
                        idxtoremovs.append(i)
                        break
    
        for i in sorted(idxtoremovs, reverse=True):
            if debug:
                print("Molname to remove:", allvalues[i]["chemicals"], "index:", i)
            del allvalues[i]

    return allvalues

####################################################################################################

def readandcheckdata (rootdirqdata, rootdirdata, howmanydifs):

    molnames = []
    labels = []
    diffs_toothermethods = []
    chemical_reacts = []
    stechio_ceofs = []
    moldescriptors = []
    chemicals_descriptors = {}
    pbe_hf_nonenergy_descriptors = []
    pbe_diff_energy_descriptors = []
    hf_diff_energy_descriptors = []
    
    hflist = ["Nuclear Repulsion  :", \
          "One Electron Energy:", \
          "Two Electron Energy:", \
          "Potential Energy   :", \
          "Kinetic Energy     :", \
          "Dispersion correction", \
          "Total Charge", \
          "Multiplicity", \
          "Number of Electrons", \
          "FINAL SINGLE POINT ENERGY"]

    pbelist = ["Nuclear Repulsion  :", \
            "One Electron Energy:", \
            "Two Electron Energy:", \
            "Potential Energy   :", \
            "Kinetic Energy     :", \
            "E(X)               :"  , \
            "E(C)               :"  , \
            "Dispersion correction", \
            "Total Charge"   , \
            "Multiplicity"   , \
            "Number of Electrons", \
            "FINAL SINGLE POINT ENERGY"]

    #Read molecules labels and more
    fp = open(rootdirdata + '/labels.txt', 'r')

    for line in fp:
        sline = line.replace("\t", " ").replace("\n", "").rstrip().lstrip().split()
        molname = sline[1]
        
        difvals = []   
        for i in range(howmanydifs):
            difvals.append(float(sline[-1*(i+1)]))
    
        schechio = []
        reacts = []
        for i in range(2,len(sline)-howmanydifs-1):
            nospace = sline[i].replace(" ", "")
            if nospace.isdigit():
                schechio.append(int(nospace))
            elif nospace.startswith("-") and nospace[1:].isdigit():
                schechio.append(int(nospace))
            else:
                reacts.append(nospace)
    
        stechio_ceofs.append(schechio)
        chemical_reacts.append(reacts)
        diffs_toothermethods.append(difvals)
        labels.append(float(sline[-1*howmanydifs-1]))
        molnames.append(molname)
        moldescriptors.append({})
    
    fp.close()

    #Read PBE data
    pbedescriptor = {}

    for file in os.listdir(rootdirqdata+'/PBE/'):
        if file.endswith('.out'):
            molname = file.split('.out')[0]
            molname = re.split("\.mpi\d+", molname)[0]
            #print(molname)
            moldesc = {}
            fp = open(rootdirqdata+'/PBE/'+file, 'r')
            for line in fp:
                for val in pbelist:
                    if line.find(val) != -1:
                        keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                        sline = line.rstrip().lstrip().split()
                        for sval in sline:
                            try:
                                firstnumvalue = float(sval)
                                break
                            except:
                                continue
                        
                        moldesc["PBE_"+keyval] = firstnumvalue
                        #print(molname, keyval, sval)
            fp.close()
            pbedescriptor[molname] = moldesc
    
    for i, molname in enumerate(molnames):
        if molname in pbedescriptor:
            for k in pbedescriptor[molname].keys():
                moldescriptors[i][k] = pbedescriptor[molname][k]
        else:
            print(molname + " not found in PBE descriptors")

    #read HF data
    hfdescriptor = {}

    for file in os.listdir(rootdirqdata+'/HF/'):
        if file.endswith('.out'):
            molname = file.split('.out')[0]
            molname = re.split("\.mpi\d+", molname)[0]
            #print(molname)
            moldesc = {}
            fp = open(rootdirqdata+'/HF/'+file, 'r')
            for line in fp:
                for val in hflist:
                    if line.find(val) != -1:
                        keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                        sline = line.rstrip().lstrip().split()
                        for sval in sline:
                            try:
                                firstnumvalue = float(sval)
                                break
                            except:
                                continue
                        
                        moldesc["HF_"+keyval] = firstnumvalue
                        #print(molname, keyval, sval)
            fp.close()
            hfdescriptor[molname] = moldesc
    
    for i, molname in enumerate(molnames):
        if molname in pbedescriptor:
            for k in hfdescriptor[molname].keys():
                moldescriptors[i][k] = hfdescriptor[molname][k]
        else:
            print(molname + " not found in HF descriptors")

    #Remove molecules with some missing descriptor
    alldims = set([len(val) for val in moldescriptors])
    idxtoremovs = []
    for i, val in enumerate(moldescriptors):
        if len(val) != max(alldims):
            idxtoremovs.append(i)
    
    for i in sorted(idxtoremovs, reverse=True):
        print("Molname to remove:", molnames[i], "index:", i)
        del moldescriptors[i]
        del labels[i]
        del molnames[i]
        del diffs_toothermethods[i]
        del chemical_reacts[i]
        del stechio_ceofs[i]

    #Remove molecules with None Label 
    for i, v in enumerate(labels):
        if v is None:
            print("None value found in labels:", i, molnames[i])
            del moldescriptors[i]
            del labels[i]
            del molnames[i]
            del diffs_toothermethods[i]
            del chemical_reacts[i]
            del stechio_ceofs[i]

    # If a descriptor is nan at least for a molecule remove from all
    nandescriptors = set()
    for index, molname in enumerate(molnames):
        if any(math.isnan(val) for val in moldescriptors[index].values()):
            print("Nan value found in descriptors:", molname)
            for k,v in moldescriptors[index].items():
                if math.isnan(v):
                    nandescriptors.add(k)

    print("Removing the following Descriptors ", nandescriptors)
    print("Removing ", len(nandescriptors), " descriptors")
    for i, v in enumerate(moldescriptors):
        for k in nandescriptors:
            del moldescriptors[i][k]

    # red chemicals
    for v in chemical_reacts:
        for chem in v:
    
            moldesc = {}
            fp = open(rootdirqdata + '/PBE/'+chem+'.out', 'r')
            for line in fp:
                for val in pbelist:
                    if line.find(val) != -1:
                        keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                        sline = line.rstrip().lstrip().split()
                        for sval in sline:
                            try:
                                firstnumvalue = float(sval)
                                break
                            except:
                                continue
                        
                        moldesc["PBE_"+keyval] = firstnumvalue
    
            fp.close
    
            fp = open(rootdirqdata + '/HF/'+chem+'.out', 'r')
            for line in fp:
                for val in hflist:
                    if line.find(val) != -1:
                        keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                        sline = line.rstrip().lstrip().split()
                        for sval in sline:
                            try:
                                firstnumvalue = float(sval)
                                break
                            except:
                                continue
                        
                        moldesc["HF_"+keyval] = firstnumvalue
            fp.close()
    
            chemicals_descriptors[chem] = moldesc

    #Build PBE and HF differences and thus descriptors
    pbeenergylist = ["PBE_Nuclear_Repulsion", \
                "PBE_One_Electron_Energy", \
                "PBE_Two_Electron_Energy", \
                "PBE_Potential_Energy", \
                "PBE_Kinetic_Energy", \
                "PBE_E(X)"  , \
                "PBE_E(C)"  , \
                "PBE_Dispersion_correction", \
                "PBE_FINAL_SINGLE_POINT_ENERGY"]
    
    hfenergylist = ["HF_Nuclear_Repulsion", \
              "HF_One_Electron_Energy", \
              "HF_Two_Electron_Energy", \
              "HF_Potential_Energy", \
              "HF_Kinetic_Energy", \
              "HF_Dispersion_correction", \
              "HF_FINAL_SINGLE_POINT_ENERGY"]
    
    for mi, molname in enumerate(molnames):
        diff_desc = {}
        for desc in hfenergylist:
            y = moldescriptors[mi][desc]
            si = 1
            comp = 0.0
            for ci, chem in enumerate(chemical_reacts[mi]):
                stecchio = stechio_ceofs[mi][si]
                comp += stecchio*chemicals_descriptors[chem][desc]
                si += 1
            
            diff_desc[desc] = comp-y
        
        hf_diff_energy_descriptors.append(diff_desc)
    
    for mi, molname in enumerate(molnames):
        diff_desc = {}
        for desc in pbeenergylist:
            y = moldescriptors[mi][desc]
            si = 1
            comp = 0.0
            for ci, chem in enumerate(chemical_reacts[mi]):
                stecchio = stechio_ceofs[mi][si]
                comp += stecchio*chemicals_descriptors[chem][desc]
                si += 1
            
            diff_desc[desc] = comp+stechio_ceofs[mi][0]*y
        
        pbe_diff_energy_descriptors.append(diff_desc)
    
    pbenonenergylist = ["PBE_Total_Charge", \
                "PBE_Multiplicity", \
                "PBE_Number_of_Electrons"]
    
    hfnonenergylist = ["HF_Total_Charge", \
                "HF_Multiplicity", \
                "HF_Number_of_Electrons"]
    
    if (len(pbenonenergylist) != len(hfnonenergylist)):
        print("Error: len(hfnonenergylist) != len(pbenonenergylist)")
        exit(1) 
    
    # check if they are equal 
    for mi, molname in enumerate(molnames):
        nondiff_desc = {}
    
        for idx in range(len(pbenonenergylist)):
    
            pbe_desc = pbenonenergylist[idx]
            hf_desc = hfnonenergylist[idx]  
            diff = moldescriptors[mi][pbe_desc] - moldescriptors[mi][hf_desc]
    
            if diff != 0.0:
                print("Error: diff != 0.0")
                print("molname:", molname)
                exit(1)
            else:  
                basidescname = pbe_desc.replace("PBE_", "").replace("HF_", "")
                nondiff_desc[basidescname] = moldescriptors[mi][pbe_desc]
    
        pbe_hf_nonenergy_descriptors.append(nondiff_desc)

    return molnames, labels, diffs_toothermethods, chemical_reacts, \
        stechio_ceofs, moldescriptors, chemicals_descriptors, \
        pbe_hf_nonenergy_descriptors, pbe_diff_energy_descriptors, \
        hf_diff_energy_descriptors

####################################################################################################
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
                    read_dataset(rootdir, labelsfilename, \
                                             howmanydifs, methods, \
                                             debug=DEBUG)
              for i in range(len(values)):
                    values[i]["setname"] = setname
                    values[i]["super_setname"] = super_setname
                  
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

####################################################################################################

def build_XY_matrix (fulldescriptors, labels):

    # build features matrix and labels
    moldescriptors_featues = []
    Y = []
    features_names = []

    for k in fulldescriptors:
        moldescriptors_featues.append(fulldescriptors[k])
        features_names.append(k)
    
    Y = np.array(labels)
    moldescriptors_featues = np.array(moldescriptors_featues)
    moldescriptors_featues = moldescriptors_featues.T

    return  moldescriptors_featues, Y, features_names

####################################################################################################

def build_features_matrix_and_labels (molnames, descriptors, labels):
    # build features matrix and labels
    moldescriptors_featues = []
    Y = []
    features_names = []

    for idx, _ in enumerate(molnames):
        val = []
        for k,v in descriptors[idx].items():
            if idx == 0:
                features_names.append(k)
            val.append(v)
        moldescriptors_featues.append(val)
        Y.append(labels[idx])

    Y = np.array(Y)
    moldescriptors_featues = np.array(moldescriptors_featues)

    return  moldescriptors_featues, Y, features_names

####################################################################################################

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

####################################################################################################

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

####################################################################################################

def get_top_correlations_blog(df, threshold=0.4):
    """
    df: the dataframe to get correlations from
    threshold: the maximum and minimum value to include for correlations. For eg, if this is 0.4, only pairs haveing a correlation coefficient greater than 0.4 or less than -0.4 will be included in the results. 
    """
    orig_corr = df.corr()
    c = orig_corr.abs()
    so = c.unstack()

    #print("|    Variable 1    |    Variable 2    | Correlation Coefficient    |")
    #print("|------------------|------------------|----------------------------|")
    
    #i=0
    pairs=set()
    result = []
    for index, value in so.sort_values(ascending=False).items():
        #print(index, value)
        # Exclude duplicates and self-correlations
        if (value >= threshold) \
            and (index[0] != index[1]) \
            and ((index[0], index[1]) not in pairs):
            #\
            #and ((index[1], index[0]) not in pairs):
            
            #print(f'|    {index[0]}    |    {index[1]}    |    {orig_corr.loc[(index[0], index[1])]}    |')
            #result.loc[i, ['Variable 1', 'Variable 2', 'Correlation Coefficient']] = \
            #    [index[0], index[1], orig_corr.loc[(index[0], index[1])]]
            result.append([index[0], index[1], orig_corr.loc[(index[0], index[1])]])
            pairs.add((index[0], index[1]))
            #i+=1
            
    #return result.reset_index(drop=True).set_index(['Variable 1', 'Variable 2'])
    return result

####################################################################################################

def wtmad2(identifier_list, labels_list, predictions_list):

    wtmad2_df = None
    wtmadtoret = None

    if len(identifier_list)==len(labels_list)==len(predictions_list):
        
        df = pd.DataFrame({
            'Identifier': identifier_list,
            'Label': labels_list,
            'Prediction': np.round(predictions_list,2)
        })

        iterss = set(identifier_list)
        supersetlist = []
        datasetslist=[]

        for element in iterss:
            pos = element.rfind("_")
            dset = element[pos+1:]
            sset = element[:pos]
            datasetslist.append(dset)
            supersetlist.append(sset)

        ssetlist = set(supersetlist)
        ssetlist.add("Full")
        datasetslist = set(datasetslist)

        df["AbsE"] = abs(df["Label"])
        df["Delta"] = abs(df["Prediction"]-df['Label'])

        N_Full = len(df)
        deltaE = []
        wtmad2_df = pd.DataFrame(columns=['Set',"WTMAD-2"])
        partials_Full = 0

        for sset in ssetlist:
            if sset != "Full":
                sset_cond = df["Identifier"].str.startswith(sset)
                sset_df = df[sset_cond]  
                N_t = len(sset_df)
                meanE_sset = []
                sset_partial = 0

                for dataset in datasetslist:
                    dataset_cond = sset_df["Identifier"].str.endswith(dataset)
                    dataset_df = sset_df[dataset_cond]
                    n_i = len(dataset_df)
                    if n_i==0: 
                        continue
                    else:
                        mad_i = dataset_df['Delta'].mean()
                        meanE_i = dataset_df['AbsE'].mean()
                        deltaE.append(meanE_i)
                        meanE_sset.append(meanE_i)
                        partial = n_i*mad_i/meanE_i
                        sset_partial += partial
                        partials_Full += partial

                meanE = sum(meanE_sset)/len(meanE_sset) # Use this parameter instead of the constant below in case we want to use a specific mean for each SuperSet
                wtmad2_sset = sset_partial*57.81/N_t # The 57.81 is the mean of the absolute mean energies of each dataset. 
                new_row = {'Set': sset, 'WTMAD-2': round(wtmad2_sset,2)}
                wtmad2_df.loc[len(wtmad2_df)] = new_row
        
                meanE_Full = sum(deltaE)/len(deltaE)
                wtmad2_Full = partials_Full*meanE_Full/N_Full
                wtmad2_df.loc[len(wtmad2_df)] = {'Set':"Full","WTMAD-2":round(wtmad2_Full,2)}
                
        wtmadtoret = {}
        for v in wtmad2_df.values:
            wtmadtoret[v[0]] = v[1]

        ssetlist.remove("Full")
        if len(set(ssetlist))==1: 
            del wtmadtoret["Full"]

    return wtmadtoret

####################################################################################################
