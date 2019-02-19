

git clone https://github.com/rodgomrod/kaggle_microsoft

cd kaggle_microsoft

pip3 install -r requirements2.txt

mkdir data


cd ..

mv train.csv kaggle_microsoft/data/train.csv

mv test.csv kaggle_microsoft/data/test.csv

cd kaggle_microsoft

chmod +x tablon_amazon.sh

source enviroment.sh

./tablon_amazon.sh