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
#                if [ ${f##*.} != png ]
#                then
#                        echo "$f is not png" 1>&2
#                        continue
#                fi
#                python ./faceCutOut.py ${f} 2> /dev/null
                python ./faceCutOut.py ${f} >> log.txt
        done
done
