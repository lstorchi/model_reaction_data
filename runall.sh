
python3 generaldataset_PLS_work1.py PBE MINIX > out
./extract.sh PBE_MINIX
python3 generaldataset_PLS_work1.py PBE SVP >> out
./extract.sh PBE_SVP
python3 generaldataset_PLS_work1.py PBE TZVP >> out
./extract.sh PBE_TZVP
python3 generaldataset_PLS_work1.py PBE QZVP >> out
./extract.sh PBE_QZVP

python3 generaldataset_PLS_work1.py PBE SVP PBE MINIX  >> out
./extract.sh PBE_SVP_MINIX

python3 generaldataset_PLS_work1.py PBE TZVP PBE MINIX >> out
./extract.sh PBE_TZVP_MINIX
python3 generaldataset_PLS_work1.py PBE TZVP PBE SVP >> out
./extract.sh PBE_TZVP_SVP

python3 generaldataset_PLS_work1.py PBE QZVP PBE MINIX >> out
./extract.sh PBE_QZVP_MINIX
python3 generaldataset_PLS_work1.py PBE QZVP PBE SVP >> out
./extract.sh PBE_QZVP_SVP
python3 generaldataset_PLS_work1.py PBE QZVP PBE TZVP >> out
./extract.sh PBE_QZVP_TZVP

python3 generaldataset_PLS_work1.py PBE0 MINIX > out
./extract.sh PBE0_MINIX
python3 generaldataset_PLS_work1.py PBE0 SVP >> out
./extract.sh PBE0_SVP
python3 generaldataset_PLS_work1.py PBE0 TZVP >> out
./extract.sh PBE0_TZVP
python3 generaldataset_PLS_work1.py PBE0 QZVP >> out
./extract.sh PBE0_QZVP

python3 generaldataset_PLS_work1.py PBE0 SVP PBE0 MINIX  >> out
./extract.sh PBE0_SVP_MINIX

python3 generaldataset_PLS_work1.py PBE0 TZVP PBE0 MINIX >> out
./extract.sh PBE0_TZVP_MINIX
python3 generaldataset_PLS_work1.py PBE0 TZVP PBE0 SVP >> out
./extract.sh PBE0_TZVP_SVP

python3 generaldataset_PLS_work1.py PBE0 QZVP PBE0 MINIX >> out
./extract.sh PBE0_QZVP_MINIX
python3 generaldataset_PLS_work1.py PBE0 QZVP PBE0 SVP >> out
./extract.sh PBE0_QZVP_SVP
python3 generaldataset_PLS_work1.py PBE0 QZVP PBE0 TZVP >> out
./extract.sh PBE0_QZVP_TZVP
