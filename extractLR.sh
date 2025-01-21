#!/bin/bash

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 SETNAME"
    exit 1
fi

val01=$(grep "Full , using Custom LR Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val02=$(grep "Full , using Custom LR Full split MAPE" modelsresults.csv | awk -F, '{print $3}') 
val03=$(grep "Full , using Custom LRRF MAPE" modelsresults.csv | awk -F, '{print $3}' )
val04=$(grep "Full , using Custom LRRF split MAPE" modelsresults.csv | awk -F, '{print $3}')
val05=$(grep "GMTK , Custom LR RF MAPE" modelsresults.csv | awk -F, '{print $3}')
val06=$(grep "GMTK , Custom LR RF split MAPE" modelsresults.csv | awk -F, '{print $3}')
val07=$(grep "GMTK , Custom LR Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val08=$(grep "GMTK , Custom LR Full split MAPE" modelsresults.csv | awk -F, '{print $3}')
val09=$(grep "FLPs , Custom LR RF MAPE" modelsresults.csv | awk -F, '{print $3}')
val10=$(grep "FLPs , Custom LR RF split MAPE" modelsresults.csv | awk -F, '{print $3}')
val11=$(grep "FLPs , Custom LR Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val12=$(grep "FLPs , Custom LR Full split MAPE" modelsresults.csv | awk -F, '{print $3}')

echo $1 , \
     $val01 , \
     $val02 , \
     $val03 , \
     $val04 , \
     $val05 , \
     $val06 , \
     $val07 , \
     $val08 , \
     $val09 , \
     $val10 , \
     $val11 , \
     $val12

# check if a dir exists and if not create it
if [ ! -d $1 ]; then
    mkdir $1
fi

cd $1
rm *.csv
cd ..

mv modelsresults.csv modelscoefficients.csv $1
