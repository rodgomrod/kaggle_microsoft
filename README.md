# Microsoft Kaggle Competition

## Descripción.

Este es el repositorio de código de `Mapsa Team`  para la [competición de Kaggle de Microsoft](https://www.kaggle.com/c/microsoft-malware-prediction).

Se obtuvo la novena posición en el [leaderboard privado](https://www.kaggle.com/c/microsoft-malware-prediction/leaderboard).


## Informacion del repositorio

Realizado por:

| Nombre | Email |
| ---- | ---- |
| [Rodrigo Gomez Rodriguez](https://www.linkedin.com/in/rodrigo-gomez/)  | rodgomrod@gmail.com |
| [Carlos Sevilla Barceló](https://www.linkedin.com/in/carlos-sevilla-barceló/)  | c.sevilla.barcelo@gmail.com |


Para obtener los datos, hay que lanzar `tablon.sh` o `tablon_amazon.sh`, teniendo previamente los [datos de la competición](https://www.kaggle.com/c/microsoft-malware-prediction/data) en la carpeta `data`.

Los modelos están en la carpeta `model`. Todos requieren que los datos de train y test estén generados.

El término __Tablón__ se refiere al script que genera el dataset de train y test. 

### Estructura del repositorio:
- `AWS`: Scripts para lanzar el tablón en AWS.
- `bbdd`: Scripts para instalar postgre y mongosql. No se usaron finalmente durante la competición
- `doc`: Distintos .txt con información de las variables, que proporciona Kaggle.
- `ETL`: Aquí están los scripts que utiliza el tablón para generar los datos
- `gist`: Diferentes scripts importantes que se han utilizado a lo largo de la competición que deben guardarse.
- `model`: Scripts que lanzan los modelos utilizados durante la competición. Están adaptados para que reciban los hyperparametros del modelo desde la consola.
- `Notebooks`: Diferentes notebooks usados durante la competición. No están pensados para su lectura, debido a que no están todos bien documentados.


