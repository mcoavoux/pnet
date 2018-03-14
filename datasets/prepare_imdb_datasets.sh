

imdb=aclImdb

if ! [ -f ${imdb}_v1.tar.gz ]
then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
fi

if ! [ -d "${imdb}" ]
then
    tar xzvf ${imdb}_v1.tar.gz
fi


for t in pos neg
do
    for set in train test
    do
        > ${imdb}/${set}/${t}_examples
        for file in ${imdb}/${set}/${t}/*.txt
        do
            cat ${file} >> ${imdb}/${set}/${t}_examples
            echo >> ${imdb}/${set}/${t}_examples
        done
    done
done