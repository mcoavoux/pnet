

# http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

if ! [ -e newsspace200.xml.bz ]
then
    wget http://www.di.unipi.it/~gulli/newsspace200.xml.bz
fi

bzip2 -dk newsspace200.xml.bz