 for name in * ; do cd $name; obabel -ixyz struc.xyz -opdb -O struc.pdb ; cd ..; done
