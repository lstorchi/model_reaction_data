
wget http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/results/"$1"/PBE/result.html

sed "s/<td>//g"  result.html > /tmp/1
sed "s/<\/td>//g"  /tmp/1 > /tmp/2
sed "s/<\/tr>//g"  /tmp/2 > /tmp/3
sed "s/<tr align=right>//g"  /tmp/3 > "$1"_labels.txt

rm -f result.html
