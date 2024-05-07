from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

import numpy as np
import pandas as pd
import math 
import os 
import re

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
                for line in fp:
                    for val in methods[method]:
                        if line.find(val) != -1:
                            keyval = val.replace(":", "").rstrip().lstrip().replace(" ", "_")
                            sline = line.rstrip().lstrip().split()
                            for sval in sline:
                                try:
                                    firstnumvalue = float(sval)
                                    break
                                except:
                                    continue
                            
                            moldesc[method+"_"+keyval] = firstnumvalue
                            #print(molname, keyval, sval)
                fp.close()
                if first:
                    first = False
                    desclist = list(moldesc.keys())
                else:
                    if desclist != list(moldesc.keys()):
                        if debug:
                            print("Error: desclist != list(moldesc.keys())")
                            print("Looking for ",desclist)
                            print("Found ", moldesc.keys())
                            print("Cannot find features in Molname:", \
                                  rootdir+'/'+method+'/'+file)
                        return None
       
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

def build_XY_matrix (fulldescriptors, labels):

    # build features matrix and labels
    moldescriptors_featues = []
    Y = []
    features_names = []

    for idx, descriptors in enumerate(fulldescriptors):
        val = []
        for k,v in descriptors.items():
            if idx == 0:
                features_names.append(k)
            val.append(v)
        moldescriptors_featues.append(val)
        Y.append(labels[idx])

    Y = np.array(Y)
    moldescriptors_featues = np.array(moldescriptors_featues)

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

def wtmad_ref(fullsetnames, metricsets, howmanydifs, allvalues_perset, includeFull = True):

    datasets = []
    partials = []
    Nis = []
    Es = []

    for setname in fullsetnames:
        for supersets in metricsets: # This loop iterates over supersets
            if setname.startswith(supersets+"_"):
                inner_partials=[]

                for methodid in range(howmanydifs): # This loop iterates over methods
                    diff_list = []
                    label_list = []
                    for val in allvalues_perset[setname]:

                        diff_list.append(val["difs"][methodid])
                        label_list.append(val["label"])

                    MAD_list = [abs(x) for x in diff_list]
                    MAD = np.mean(MAD_list)
                    MAD = round(MAD,2)

                    meanE_list = [abs(x) for x in label_list]
                    meanE = np.mean(meanE_list)
                    meanE = round(meanE,2)

                    Ni = len(meanE_list)
                    partial = round(Ni*MAD/meanE,2)


                    inner_partials.append(partial)


                Nis.append(Ni)
                Es.append(meanE)
                partials.append(inner_partials)
                datasets.append(setname)

    n = len(partials[0])
    column_names = [f'partials_{i}' for i in range(n)]


    # This line creates the base data frame for the calculation

    df_wtmad = pd.DataFrame({
                            'datasets': datasets,
                            'Nis': Nis,
                            'Es': Es,
                            **{column_names[i]: [item[i] for item in partials] for i in range(n)}
                            })

    supersets = []
    wtmad2 = []

    for ssets in metricsets:
        supersets.append(ssets)
        tempdf = df_wtmad[df_wtmad['datasets'].str.startswith(ssets+"_")]

        totalNi = tempdf["Nis"].sum()
        total_meanE = tempdf["Es"].mean() 
        
        summations = []

        for i in range(n):
            summations.append(tempdf[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    if includeFull == True:

        supersets.append("Full")

        totalNi = df_wtmad["Nis"].sum()
        total_meanE = df_wtmad["Es"].mean()

        summations = []
        
        for i in range(n):
            summations.append(df_wtmad[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    # This part creates the final dataframe with the results

    n = len(wtmad2[0])
    column_names = [f'WTMAD-2_{i}' for i in range(n)]

    df_wtmad2 = pd.DataFrame({
                            'Superset': supersets,
                            **{column_names[i]: [item[i] for item in wtmad2] for i in range(n)}
                            })
                
    #print(df_wtmad2, "\n")

    return df_wtmad2

####################################################################################################

def wtmad(fullsetnames, metricsets, methods, allvalues_perset, includeFull = True):

    datasets = []
    partials = []
    Nis = []
    Es = []

    keys_list = list(methods.keys())

    for setname in fullsetnames:
        for supersets in metricsets: # This loop iterates over supersets
            if setname.startswith(supersets+"_"):
                inner_partials=[]

                for j, method in enumerate(methods): # This loop iterates over methods
                    diff_list = []
                    label_list = []
                    for val in allvalues_perset[setname]:

                        diff_list.append(val['label']- \
                                         val[method + "_energydiff"][method+"_FINAL_SINGLE_POINT_ENERGY"])
                        label_list.append(val["label"])
                    
                    MAD_list = [abs(x) for x in diff_list]
                    MAD = np.mean(MAD_list)
                    MAD = round(MAD,2)

                    meanE_list = [abs(x) for x in label_list]
                    meanE = np.mean(meanE_list)
                    meanE = round(meanE,2)

                    Ni = len(meanE_list)
                    partial = round(Ni*MAD/meanE,2)

                    inner_partials.append(partial)

                Nis.append(Ni)
                Es.append(meanE)
                partials.append(inner_partials)
                datasets.append(setname)

    n = len(partials[0])
    column_names = [f'partials_{i}' for i in range(n)]

    # This line creates the base data frame for the calculation

    df_wtmad = pd.DataFrame({
                            'datasets': datasets,
                            'Nis': Nis,
                            'Es': Es,
                            **{column_names[i]: [item[i] for item in partials] for i in range(n)}
                            })

    supersets = []
    wtmad2 = []

    for ssets in metricsets:
        supersets.append(ssets)
        wtmads = []
        tempdf = df_wtmad[df_wtmad['datasets'].str.startswith(ssets+"_")]

        totalNi = tempdf["Nis"].sum()
        total_meanE = tempdf["Es"].mean() 
        
        summations = []

        for i in range(n):
            summations.append(tempdf[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    if includeFull == True:

        supersets.append("Full")

        totalNi = df_wtmad["Nis"].sum()
        total_meanE = df_wtmad["Es"].mean()

        summations = []
        
        for i in range(n):
            summations.append(df_wtmad[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    # This part creates the final dataframe with the results

    n = len(wtmad2[0])
    column_names = [f'WTMAD-2_{i}' for i in keys_list]

    df_wtmad2 = pd.DataFrame({
                            'Superset': supersets,
                            **{column_names[i]: [item[i] for item in wtmad2] for i in range(n)}
                            })
                
    #print(df_wtmad2, "\n")

    return df_wtmad2

####################################################################################################

def wtmad_calc(supersetlist, setlist, predicted, labels, includeFull = True):

    metricsets = list(set(supersetlist))
    fullsetnames = list(set(setlist)) + list(set(supersetlist))
    methods = {"Predicted": predicted}
    allvalues_perset = {}
    for i, setname in enumerate(setlist):
        if setname in allvalues_perset:
            allvalues_perset[setname].append({"label": labels[i], \
                                              "Predicted_energydiff": predicted[i]})
        else:
            allvalues_perset[setname] = []
            allvalues_perset[setname].append({"label": labels[i], \
                                             "Predicted_energydiff": predicted[i]})

    datasets = []
    partials = []
    Nis = []
    Es = []

    keys_list = list(methods.keys())    

    for setname in fullsetnames:
        for superset in metricsets: # This loop iterates over supersets
            if setname.startswith(superset+"_"):
                inner_partials=[]
                inner_E = []

                for j, method in enumerate(methods): # This loop iterates over methods
                    diff_list = []
                    label_list = []
                    for val in allvalues_perset[setname]:

                        diff_list.append(val['label']- \
                                         val[method + "_energydiff"])
                        MAD_list = [abs(x) for x in diff_list]
                        MAD = np.mean(MAD_list)
                        MAD = round(MAD,2)

                        label_list.append(val["label"])
                        meanE_list = [abs(x) for x in label_list]
                        meanE = np.mean(meanE_list)
                        meanE = round(meanE,2)

                        Ni = len(meanE_list)
                        partial = round(Ni*MAD/meanE,2)

                    inner_partials.append(partial)
                    inner_E.append(meanE)

                Nis.append(Ni)
                Es.append(meanE)
                partials.append(inner_partials)
                datasets.append(setname)

    n = len(partials[0])
    column_names = [f'partials_{i}' for i in range(n)]

    # This line creates the base data frame for the calculation

    df_wtmad = pd.DataFrame({
                            'datasets': datasets,
                            'Nis': Nis,
                            'Es': Es,
                            **{column_names[i]: [item[i] for item in partials] for i in range(n)}
                            })

    supersets = []
    wtmad2 = []

    for ssets in metricsets:
        supersets.append(ssets)
        wtmads = []
        tempdf = df_wtmad[df_wtmad['datasets'].str.startswith(ssets+"_")]

        totalNi = tempdf["Nis"].sum()
        total_meanE = tempdf["Es"].mean() 
        
        summations = []

        for i in range(n):
            summations.append(tempdf[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    if includeFull == True:

        supersets.append("Full")

        totalNi = df_wtmad["Nis"].sum()
        total_meanE = df_wtmad["Es"].mean()

        summations = []
        
        for i in range(n):
            summations.append(df_wtmad[f'partials_{i}'].sum())

        metrics = []

        for i in range(n):
            metric = round(total_meanE * summations[i] / totalNi, 2)
            metrics.append(metric)

        metric_list = [metrics[i] for i in range(n)]

        wtmad2.append(metric_list)

    # This part creates the final dataframe with the results

    result = {}
    for i, superset in enumerate(supersets):
        result[superset] = wtmad2[i][0]

    return result

######################################################################################


def wtmad2_lf(identifier_list, labels_list, predictions_list):

    if len(identifier_list)==len(labels_list)==len(predictions_list):
        
        df = pd.DataFrame({
            'Identifier': identifier_list,
            'Label': labels_list,
            'Prediction': np.round(predictions_list,2)
        })
        
        iterss = set(identifier_list)
        ssetlist = []
        datasetslist=[]

        for element in iterss:
            pos = element.rfind("_")
            dset = element[pos+1:]
            sset = element[:pos]
            datasetslist.append(dset)
            ssetlist.append(sset)

        ssetlist = set(ssetlist)
        ssetlist.add("Full")
        datasetslist = set(datasetslist)

        df["AbsE"] = abs(df["Label"])
        df["Delta"] = abs(df["Prediction"]-df['Label'])

        N_Full = len(df)
        meanE_sum = 0
        meanE_counter = 0
        partials_Full = 0
        wtmad2 = {}
        for sset in ssetlist:
            if sset != "Full":
                sset_cond = df["Identifier"].str.startswith(sset)
                sset_df = df[sset_cond]  
                meanE_sset_sum  = 0
                meanE_sset_counter = 0
                N_t = len(sset_df)
                partial = 0

                for dataset in datasetslist:
                    dataset_cond = sset_df["Identifier"].str.endswith(dataset)
                    dataset_df = sset_df[dataset_cond]
                    n_i = len(dataset_df)
                    if n_i==0: 
                        continue
                    else:
                        mad_i = dataset_df['Delta'].mean()
                        meanE_i = dataset_df['AbsE'].mean()
                        partial += n_i*mad_i/meanE_i
                        meanE_sum += meanE_i
                        meanE_counter += 1
                        meanE_sset_sum += meanE_i
                        meanE_sset_counter += 1 

                meanE = meanE_sset_sum/meanE_sset_counter
                meanE_Full = meanE_sum/meanE_counter
                partials_Full += partial
                wtmad2_sset = partial*meanE/N_t #Evaluate the use of a constant value or a calculated one for each superset
                wtmad2[sset] = round(wtmad2_sset,2)
                    
        wtmad2_Full = partials_Full*meanE_Full/N_Full
        wtmad2["Full"]=round(wtmad2_Full,2)
    
    return wtmad2