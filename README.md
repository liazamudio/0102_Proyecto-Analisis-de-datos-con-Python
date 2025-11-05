# Proyecto hecho para el Módulo de Análisis de datos con Python del Certificado de Cienca de Datos en el 2023

Este proyecto tiene cosntantes actualizaciones conforme se vaya requiriendo.

# Análisis de Datos de Airbnb - Ciudad de México

Este proyecto contiene un análisis completo de datos de Airbnb en la Ciudad de México utilizando Python y diversas librerías de ciencia de datos.

## Requisitos

- Python 3.13 o superior
- Entorno virtual configurado

## Instalación

### 1. Crear y activar el entorno virtual

El proyecto ya incluye un entorno virtual en la carpeta `env`. Para activarlo:

**Windows (PowerShell):**
```powershell
.\env\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\env\Scripts\activate.bat
```

### 2. Librerías instaladas

Las siguientes librerías ya están instaladas en el entorno virtual:

- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualización de datos
- **seaborn**: Visualización estadística
- **scipy**: Cálculos científicos y estadísticos
- **scikit-learn**: Machine Learning
- **plotly**: Visualizaciones interactivas
- **folium**: Mapas interactivos
- **nltk**: Procesamiento de lenguaje natural

### 3. Recursos de NLTK

El notebook descarga automáticamente los recursos necesarios de NLTK:
- `punkt`: Para tokenización de texto
- `stopwords`: Para filtrar palabras comunes

## Estructura del Proyecto

```
.
├── 01-datos_airbnb.ipynb    # Notebook principal con el análisis
├── datasets/
│   └── listings-airbnb-mex.csv  # Dataset de Airbnb (se descarga automáticamente)
├── env/                      # Entorno virtual de Python
├── Proyect/                  # Archivos del proyecto
└── README.md                 # Este archivo
```

## Contenido del Análisis

El notebook incluye los siguientes análisis:

### 1. Análisis Exploratorio de Datos
- Carga y exploración del dataset
- Estadísticas descriptivas
- Estimados de locación y variabilidad

### 2. Visualización de Datos
- Distribuciones de precios
- Boxplots y gráficos de densidad
- Análisis por alcaldía y tipo de habitación
- Mapas de calor de correlaciones

### 3. Análisis Multivariable
- Correlaciones entre variables
- Análisis por región geográfica
- Visualizaciones avanzadas (treemaps, scatterplots, binnings hexagonales)
- Mapas coropléticos

### 4. Modelado Estadístico
- Regresión lineal
- Bootstraping
- Intervalos de confianza
- Validación cruzada

### 5. Machine Learning
- **Clasificación No Supervisada**: K-Means clustering
- **Clasificación Supervisada**: Regresión logística
- Matrices de confusión
- Curvas ROC/AUC

### 6. Procesamiento de Lenguaje Natural
- Análisis de texto con NLTK
- Tokenización y frecuencia de palabras
- Eliminación de stopwords
- Expresiones regulares (regex)

## Uso

1. Abre el notebook `01-datos_airbnb.ipynb` en VS Code o Jupyter
2. Asegúrate de que el kernel está configurado correctamente (env)
3. Ejecuta las celdas en orden secuencial
4. Las primeras celdas se encargan de:
   - Confirmar la instalación de librerías
   - Importar todas las dependencias
   - Descargar recursos de NLTK
   - Cargar el dataset desde GitHub

## Dataset

El dataset se carga automáticamente desde el repositorio de GitHub:
- URL: `https://raw.githubusercontent.com/liazamudio/Proy-Analisis-de-datos-con-Python/main/datasets/listings-airbnb-mex.csv`
- Contiene información sobre propiedades de Airbnb en la Ciudad de México

## Notas

- El notebook procesa y filtra los datos para eliminar valores atípicos (outliers)
- Se utiliza un filtro del 95% (percentiles 2.5% y 97.5%) para análisis más precisos
- Algunas visualizaciones requieren conexión a internet para cargar mapas

## Solución de Problemas

### El kernel no se inicia
```powershell
# Recrear el entorno virtual
python -m venv env --clear
.\env\Scripts\Activate.ps1
```

### Falta alguna librería
El notebook configurará automáticamente el entorno al ejecutarse por primera vez.

### Errores de NLTK
La celda de configuración de NLTK descarga automáticamente los recursos necesarios. Si hay problemas, ejecuta manualmente:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Créditos

- Dataset original: Airbnb CDMX
- Análisis: Proyecto de curso de Análisis de Datos con Python

## Licencia

Este proyecto es con fines educativos.
