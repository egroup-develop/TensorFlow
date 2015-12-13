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
                	#convert $f -resize x28 -resize '28x<' -gravity center -crop 28x28+0+0 +repage ${f%.*}_convert.jpeg
                	convert $f -resize x128 -resize '128x<' -gravity center -crop 128x128+0+0 +repage ${f%.*}_convert.jpeg
                	#convert $f -resize x64 -resize '64x<' -gravity center -crop 64x64+0+0 +repage ${f%.*}_convert.jpeg
                	#convert $f -resize x32 -resize '32x<' -gravity center -crop 32x32+0+0 +repage ${f%.*}_convert.jpeg
                fi
        done
done
