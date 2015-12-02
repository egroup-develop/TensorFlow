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
                #python ./faceCutOut.py ${f} 2> /dev/null
                convert $f -resize x28 -resize '28x<' -gravity center -crop 28x28+0+0 +repage ${f%.*}_convert.jpeg
        done
done
