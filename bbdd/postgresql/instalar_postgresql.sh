clear
echo ""
echo "COMIENZA INSTALACION POSTGRESQL"
echo ""
echo "Descargando PostgreSQL"
echo ""

sudo apt update
sudo apt install postgresql postgresql-contrib -y

echo ""
echo "Creamos usuario de Administracion"
echo ""
sudo -u postgres createuser --interactive

echo ""
echo "Fin de instalacion"
