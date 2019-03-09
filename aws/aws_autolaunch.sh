#
# This file launch automatically the ETL process to get the Train&Test dataset that we use for this competition.
# 
# I assume that you have train.csv and test.csv from Kaggle on /home/__user__ 
#


# This is for up to root project
cd ..

# Install requirements
pip3 install -r requirements.txt

# Create data folder
mkdir data

# Move train and test to data folder
cd ..
mv train.csv kaggle_microsoft/data/train.csv
mv test.csv kaggle_microsoft/data/test.csv

cd kaggle_microsoft

#Set execution permission to our ETL launcher
chmod +x tablon_amazon.sh

#Set enviroment variables
source enviroment.sh

# Launch all ETLs
./tablon_amazon.sh
