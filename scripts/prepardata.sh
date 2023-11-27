for name in *
do
    if [ -d $name ]
    then
        echo $name
        cd $name
        mkdir PBE
        cd PBE
        mv ../PBE.tar .
        tar -xvf PBE.tar
        rm PBE.tar
        cd ..
        mkdir HF
        cd HF
        mv ../HF.tar .
        tar -xvf HF.tar
        rm HF.tar
        cd ..
        cd ..
    fi
done