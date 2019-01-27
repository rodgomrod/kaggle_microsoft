echo " "
echo "######################"
echo "# AÑADIR REPOSITORIO #"
echo "######################"
echo " "

sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927

echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list

echo " "
echo "#######################"
echo "# DESCARGANDO MongoDB #"
echo "#######################"
echo " "

sudo apt-get update

sudo apt-get install -y mongodb-org

echo " "
echo "#########################################"
echo "# AÑADIENDO COMANDOS RAPIDOS AL .bashrc #"
echo "#########################################"
echo " "

echo 'alias activateMongodb="sudo systemctl start mongodb && sudo systemctl enable mongodb"' >> ~/.bashrc
echo 'alias desactivateMongodb="sudo systemctl stop mongodb"' >> ~/.bashrc


