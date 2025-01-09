export codetorun="generaldataset_LR_work1.py" 
export extractor="./extractLR.sh"

python3 $codetorun PBE MINIX > out
$extractor PBE_MINIX
python3 $codetorun PBE SVP >> out
$extractor PBE_SVP
python3 $codetorun PBE TZVP >> out
$extractor PBE_TZVP
python3 $codetorun PBE QZVP >> out
$extractor PBE_QZVP

#python3 $codetorun PBE SVP PBE MINIX  >> out
#$extractor PBE_SVP_MINIX
#
#python3 $codetorun PBE TZVP PBE MINIX >> out
#$extractor PBE_TZVP_MINIX
#python3 $codetorun PBE TZVP PBE SVP >> out
#$extractor PBE_TZVP_SVP
#
#python3 $codetorun PBE QZVP PBE MINIX >> out
#$extractor PBE_QZVP_MINIX
#python3 $codetorun PBE QZVP PBE SVP >> out
#$extractor PBE_QZVP_SVP
#python3 $codetorun PBE QZVP PBE TZVP >> out
#$extractor PBE_QZVP_TZVP

python3 $codetorun PBE0 MINIX > out
$extractor PBE0_MINIX
python3 $codetorun PBE0 SVP >> out
$extractor PBE0_SVP
python3 $codetorun PBE0 TZVP >> out
$extractor PBE0_TZVP
python3 $codetorun PBE0 QZVP >> out
$extractor PBE0_QZVP

#python3 $codetorun PBE0 SVP PBE0 MINIX  >> out
#$extractor PBE0_SVP_MINIX
#
#python3 $codetorun PBE0 TZVP PBE0 MINIX >> out
#$extractor PBE0_TZVP_MINIX
#python3 $codetorun PBE0 TZVP PBE0 SVP >> out
#$extractor PBE0_TZVP_SVP
#
#python3 $codetorun PBE0 QZVP PBE0 MINIX >> out
#$extractor PBE0_QZVP_MINIX
#python3 $codetorun PBE0 QZVP PBE0 SVP >> out
#$extractor PBE0_QZVP_SVP
#python3 $codetorun PBE0 QZVP PBE0 TZVP >> out
#$extractor PBE0_QZVP_TZVP
