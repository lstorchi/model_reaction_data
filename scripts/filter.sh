
for name in  BH76 BHDIV10 BHPERI \
             BHROT27 INV24 PX13 WCPT18 \
             ACONF ICONF IDISP MCONF \
             PCONF21 SCONF UPU23 \
             SMALL_MOLECULES \
             AL2X6 ALK8 ALKBDE10 BH76 \
             DC13 DIPCS10 FH51 G21EA \
             G21IP G2RC HEAVYSB11 NBPRC \
             PA26 RC21 SIE4x4 TAUT15 \
             W4-11 YBDE18 ADIM6 AHB21 CARBHB12 \
             CHB6 HAL59 HEAVY28 IL16 \
             PNICO23 RG18 S22 S66 WATER27 \
             BSR36 C60ISO CDIE20 DARC \
             ISO34 ISOL24 MB16-43 PArel RSE43
do 
    wget http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/results/"$name"/PBE/result.html

    sed "s/<td>//g"  result.html > /tmp/1
    sed "s/<\/td>//g"  /tmp/1 > /tmp/2
    sed "s/<\/tr>//g"  /tmp/2 > /tmp/3
    sed "s/<tr align=right>//g"  /tmp/3 > /tmp/4
    head -n -6 /tmp/4 > /tmp/5
    tail -n +33 /tmp/5 > "$name"_labels.txt

    rm -f result.html
done
