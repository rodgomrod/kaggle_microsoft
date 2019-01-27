echo "Creando bbdd microsoft si no existe"
echo " "
mongo --eval "use microsoft"

echo "Importando TRAIN a MongoDB"
echo " "
mongoimport -d microsoft -c train --type csv --file ../../data/train.csv --headerline


echo "Importando TEST a MongoDB"
echo " "
mongoimport -d microsoft -c test --type csv --file ../../data/test.csv --headerline
