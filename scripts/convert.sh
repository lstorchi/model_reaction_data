> /tmp/allmol.sdf

for name in ./data/* ; do 
   cd $name
   obabel -ixyz struc.xyz -opdb -O struc.pdb 
   obabel -ixyz struc.xyz -osdf -O struc.sdf
   cat struc.sdf >> /tmp/allmol.sdf
   cd -
 done
