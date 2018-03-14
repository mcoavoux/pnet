
if ! [ -d src ]
then
    git clone https://bitbucket.org/lowlands/release/src/
fi

for file in ./src/WWW2015/data/*.zip
do
    yes | unzip $file -d src/.
done

for file in src/*.tmp
do
    grep  "gender" ${file} | grep birth > ${file}_filtered
done
