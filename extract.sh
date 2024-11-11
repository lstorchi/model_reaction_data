#!/bin/bash

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 SETNAME"
    exit 1
fi

val1=$(grep "using PLS SS MAPE" modelsresults.csv | awk -F, '{print $3}')
val2=$(grep "using PLS SS split MAPE" modelsresults.csv | awk -F, '{print $3}')
val3=$(grep "using LR SS MAPE" modelsresults.csv | awk -F, '{print $3}')
val4=$(grep "using LR SS split MAPE" modelsresults.csv | awk -F, '{print $3}')
val5=$(grep "using Custom LR SS MAPE" modelsresults.csv | awk -F, '{print $3}')
val6=$(grep "using Custom LR SS split MAPE" modelsresults.csv | awk -F, '{print $3}')
val7=$(grep "using PLS Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val8=$(grep "using PLS Full split MAPE" modelsresults.csv | awk -F, '{print $3}')
val9=$(grep "using LR Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val10=$(grep "using LR Full split MAPE" modelsresults.csv | awk -F, '{print $3}')
val11=$(grep "using Custom LR Full MAPE" modelsresults.csv | awk -F, '{print $3}')
val12=$(grep "using Custom LR Full split MAPE" modelsresults.csv | awk -F, '{print $3}')
val13=$(grep "using PLSRF MAPE" modelsresults.csv | awk -F, '{print $3}')
val14=$(grep "using PLSRF split MAPE" modelsresults.csv | awk -F, '{print $3}')
val15=$(grep "using LRRF MAPE" modelsresults.csv | awk -F, '{print $3}')
val16=$(grep "using LRRF split MAPE" modelsresults.csv | awk -F, '{print $3}')
val17=$(grep "using Custom LRRF MAPE" modelsresults.csv | awk -F, '{print $3}')
val18=$(grep "using Custom LRRF split MAPE" modelsresults.csv | awk -F, '{print $3}')

echo $1 , $val1 , $val2 , $val3 , $val4 , $val5 , $val6 , $val7 , $val8 , $val9 , $val10 , $val11 , $val12 , $val13 , $val14 , $val15 , $val16 , $val17 , $val18

# check if a dir exists and if not create it
if [ ! -d $1 ]; then
    mkdir $1
fi

cd $1
rm *.csv
cd ..

mv modelsgeneral.csv modelsresults.csv modelscoefficients.csv $1
