import pandas as pd
import sys

print("Inicio del cargador de datos")

name = sys.argv[1]
path = sys.argv[2]
user = sys.argv[3]
psw = sys.argv[4]

print("name = {0}, path = {1}\nuser = {2}, psw = {3}".format(name, path, user, psw))

print("Cargando CSV")

df = pd.read_csv(path, sep=',')

from sqlalchemy import create_engine

print("Guardando CSV en Postgresql")

engine = create_engine('postgresql://{0}:{1}@localhost:5432/microsoft'.format(user, psw))

df.to_sql(name, engine)



