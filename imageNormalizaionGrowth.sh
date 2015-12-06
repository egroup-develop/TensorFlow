dir=$1

if [ ! -d "$dir" ]
then
        echo "$dir is not directory" 1>&2
        exit
fi

for d in "$dir"/*
do
        if [ ! -d "$d" ]
        then
                echo "$d is not directory" 1>&2
                continue
        fi

        for f in "$d"/*
        do
                if [ ${f##*.} = jpeg ]
                then
                	cp $f ${f%.*}_origin.jpeg
                	convert $f -equalize $f
                fi
        done
done
