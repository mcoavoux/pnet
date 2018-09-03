

if ! [[ -e blogs.zip ]]
then
    wget http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip
    unzip blogs.zip
fi

dir=blogdataset

mkdir -p ${dir}/m_1/
mkdir -p ${dir}/m_2/
mkdir -p ${dir}/m_3/


mkdir -p ${dir}/f_1/
mkdir -p ${dir}/f_2/
mkdir -p ${dir}/f_3/


cp blogs/*.male.1*    ${dir}/m_1/.
cp blogs/*.male.2*    ${dir}/m_2/.
cp blogs/*.male.[3-9]*    ${dir}/m_3/.

cp blogs/*.female.1*  ${dir}/f_1/.
cp blogs/*.female.2*  ${dir}/f_2/.
cp blogs/*.female.[3-9]*  ${dir}/f_3/.

