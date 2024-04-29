# WTMAD-2 calculation program
# Author: Carlos Jacinto, UNIPG

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import os


def reading(file_name):
    file_path = file_name
    data_frame = pd.read_csv(file_path, delimiter=';', header=None, skiprows=1)
    first_column = data_frame.iloc[:, 1]
    last_columns = data_frame.iloc[:, 3:17]             # This indexing is based on the input files format 
    df = pd.concat([first_column, last_columns], axis=1) 
    df.columns = ['System', 'Reference', 'Without', 'D3(0)', 'D3(BJ)', 'PBE',
                  'PBE0', 'ZORA', 'TPSSh', 'Pred', 'Gen', 'Full',
                  'PredRM', 'GenRM','FullRM']
    return(df)

def stat_calc(data_frame):
    df = reading(data_frame)
    df['Abs'] = abs(df['Reference']-df['FullRM'])               # In the input files format used to date, total energies are presented
    #df['Abs'] = abs(df["D3(BJ)"])                              # therefore, the difference with respect to the reference has to be
    df['Abs_E'] = abs(df['Reference'])                          # calculated
    df['SqE'] = df['Abs']**2
    sumRMSE = df['SqE'].sum()
    mad = df['Abs'].mean()
    mean_E = df['Abs_E'].mean()
    N_i = len(df)
    #partial_PBE = 56.84*N_i*mad_PBE/mean_E
    partial = N_i*mad/mean_E
    
    return partial, N_i, sumRMSE, mean_E                        # This function returns all partial statistics needed to calculate
                                                                # the final values to present: WTMAD-2, RMSE

if __name__ == "__main__":
    
    subsets = sys.argv[1:]
        
    summation=0                     # All starting varibales to iterate.
    n_elements = 0                  # Iterations are carried out over all the input files.
    sumRMSE = 0
    total_meanE = 0
    for element in subsets:
        element = element.strip()
        file_path = os.path.abspath(element)
        n_subsets = len(subsets)

        try: 
            with open(file_path, 'r') as file:
                stats,df_length,column_sum,mean_Es = stat_calc(file)
                summation += stats
                n_elements += df_length
                sumRMSE += column_sum
                total_meanE += mean_Es

        except:
            print('Error:', element)

    total_meanE = total_meanE/n_subsets
    wtmad2 = (1/n_elements)*total_meanE*summation              # This formula was taken from:
    wtmad2 = round(wtmad2, 2)                                  # Phys. Chem. Chem. Phys., 2017,19, 32184-32215
    rmeanse = np.sqrt(sumRMSE/n_elements)
    rmeanse = round(rmeanse, 2)
    print("Total Mean Energy:","\n",round(total_meanE,2)) 
    print("Number of entries:","\n",n_elements)
    print('WTMAD-2:',"\n",wtmad2)
    print('RMSD:', "\n",rmeanse)
