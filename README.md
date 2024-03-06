# ARJ-SRI-System

## Sistema de Recuperación de Información

### Autores
- [Raudel Gomez](https://github.com/raudel25) C411.
- [Juan Carlos Espinosa](https://github.com/Jky45) C411.
- [Alex Sierra](https://github.com/alexsierra45) C411.

### Definición de los modelos de SRI implementados
#### Modelo Booleano:
El Modelo Booleano de recuperación de información (MRIB) es uno de los modelos clásicos de RI y el primero en ser ampliamente adoptado.
Está basado en la Lógica Booleana y la Teoría de Conjuntos.
En este modelo, tanto los documentos a buscar como la consulta del usuario se conciben como conjuntos de términos.
La recuperación se basa en si los documentos contienen o no los términos de la consulta.
Dada una consulta en forma normal disyuntiva o conjuntiva, se realizan operaciones entre conjuntos para obtener los documentos relevantes.

#### Modelo Vectorial:
En el Modelo Vectorial, cada documento se representa como un vector en un espacio multidimensional.
Cada componente del vector corresponde a un término asociado al documento.
El peso del término se mide por su frecuencia de término normalizada.
Este modelo permite calcular similitudes entre documentos y consultas mediante operaciones vectoriales.

#### LSI (Indexación Semántica Latente):
LSI es una técnica que busca descubrir relaciones semánticas ocultas entre términos y documentos.
Se basa en el análisis de descomposición de valores singulares (SVD) aplicado a la matriz término-documento.
LSI reduce la dimensionalidad de los datos y agrupa términos y documentos similares.
Permite una recuperación más eficiente y considera palabras con semejante significado y relaciones contextuales.

### Consideraciones al Desarrollar la Solución
- Se ha optado por utilizar el lenguaje de programación Python debido a su amplia disponibilidad de bibliotecas especializadas en procesamiento de lenguaje natural y análisis de datos.
- La solución se divide en varios módulos para facilitar la modularidad y la mantenibilidad del código.
- Se ha empleado la biblioteca `simpy` para el trabajo con formas normales disyuntivas para el modelo booleano, `streamlit` para la interfaz visual, `gensim` para la representación de texto, `spacy` para tokenizar y y `nltk` para facilitar el trabajo con la similitud de significados de las palabras.
- Contamos con un dataset de películas para poner en práctica nuestros modelos implementados y las funcionalidades extras
- Se recomienda revisar detenidamente la documentación de cada módulo y biblioteca utilizada para comprender mejor su funcionamiento y aplicaciones específicas.

### Ejecución del Proyecto
Para ejecutar el proyecto en tu entorno local, sigue estos pasos:

1. Instala las dependencias necesarias utilizando `pip` y el archivo `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```

2. Ubicar la carpeta con el dataset dentro de la carpeta `data`

3. Ejecuta el archivo `main.py` para construir el sistema con los datos descargados y procesados especificando la cantidad de líneas(100 por defecto):
    ```
    python src/main.py 100
    ```

4. Ejecuta el archivo `app.py` para iniciar la aplicación:
    ```
    streamlit run src/app.py 100
    ```

### Explicación de la Solución Desarrollada
1. **Procesamiento del dataset para el modelo vectorial:**
Se emplea la biblioteca `spacy` para realizar el análisis sintáctico y semántico del texto, realizando tokenización y lematización. Luego el modelo TF-IDF asigna un peso a cada término en función de su frecuencia en el documento y su frecuencia inversa en el corpus completo.

2. **Procesamiento del dataset para el modelo booleano:**
Al igual que en el modelo anterior, se emplea la biblioteca `spacy` para realizar el análisis sintáctico y semántico del texto, realizando tokenización y lematización y luego se guardan los términos con el mismo peso.

3. **Procesamiento del dataset para el modelo lsi:**
En este caso, luego del análisis sintáctico y semántico del texto, usamos la matriz de TF-IDF, a la cual le hacemos una descomposición usando SVD (descomposición en valores singulares) para agrupar los términos por su semejanza de significados.

4. **Realización de consultas para el modelo vectorial:**
Cada término de la consulta luego de ser procesada(tokenizada y lematizada) es ponderado teniendo en cuenta el preprocesamiento realizado, y mediante la similitud del coseno se hayan los documentos con mayor similitud a la consulta devolviéndolos en orden.

5. **Realización de consultas para el modelo booleano:**
En este caso la consulta puede contener una serie de operadores(and, not, or) y mediante la biblioteca `simpy` se convierte en una forma normal disyuntiva. Solo se devolverán documentos que cumplan las restricciones de al menos una de las cláusulas disyuntivas.

6. **Realización de consultas para el modelo lsi:**
Con el modelo guradado en el preprocesamiento, al igual que el modelo vectorial se trata la query como un documento más y se halla la simulitud del coseno entre esta y los documentos preprocesados.

7. **Funcionalidad de autocompletado:**
Para esta funcionalidad hacemos uso de la estructura de datos Trie, donde tenemos guardado un vocabulario del idioma inglés, y hacemos recomendación de autocompletado de palabras teniendo en cuenta la cercanía entre estas y las que las tienen como prefijo haciendo un DFS por el Trie. Revisar esto 

8. **Funcionalidad de retroalimentación:**
En este caso usamos el algoritmo de Rocchio, el cual es un método de clasificación de documentos según la relevancia de sus términos. En primer lugar, selecciona un conjunto de documentos de entrenamiento etiquetados y se calculan sus vectores de características. Luego, cuando se presenta un nuevo documento (consulta), también se calcula su vector de características. El algoritmo ajusta los pesos de los términos en función de su relevancia: si un término es relevante, aumenta su peso; si no lo es, lo reduce. Finalmente, se clasifica el documento de consulta según su similitud con los documentos de entrenamiento. Para su aplicación, en la interfaz gráfica luego de mostrar los resultados de una consulta damos la posibilidad de etiquetar documentos para ajustar dichos resultados a las necesidades del usuario.

8. **Expansión de la consulta:**
Para expandir las consultas usamos 2 herramientas, una es añadiendo palabras con similar significado a los términos de la consulta a partir de un diccionario, y otra es haciendo un chequeo semántico. Para este último verificamos si los tokens de la consulta son válidos y en caso de que no lo sean añadimos los más cercanos teniendo en cuenta la distancia de Levenshtein y otras heurísticas.

### Métricas

| Métrica                      | Vectorial | LSI        |
| ---------------------------- | ---------- | ---------- |
| Precisión Relevante (R-Precision) | 0.01500326 | 0.02087410 |
| Precisión Media (AP)         | 0.01431289 | 0.01912281 |
| Precisión en el Top 10 (P@10) | 0.02054794 | 0.02328767 |
| Recobrado en el Top 10 (R@10) | 0.05061969 | 0.05238095 |
| Reciprocidad (Reciprocity)    | 0.03190367 | 0.04136225 |
| nDCG en el Top 10 (nDCG@10)   | 0.02829954 | 0.03555599 |

- Precisión Relevante (R-Precision): El modelo "LSI" muestra un ligerísimo margen superior sobre el modelo "Vectorial", con un valor de 0.02087410 en comparación con 0.01500326 del modelo "Vectorial". Este pequeño diferencial sugiere que el modelo "LSI" es ligeramente más efectivo en la identificación de documentos relevantes entre los primeros 10 resultados.
- Precisión Media (AP): El modelo "Vectorial" presenta un margen ligeramente superior en la precisión media, lo que indica una mejor capacidad para calcular la precisión promedio a lo largo de toda la lista de resultados.
- Precisión en el Top 10 (P@10): Ambas métricas muestran un rendimiento cercano entre los modelos, con el modelo "LSI" ligeramente por encima del modelo "Vectorial", lo que sugiere una mejora en la precisión dentro del top 10 documentos recuperados.
- Recobrado en el Top 10 (R@10): El modelo "LSI" supera al modelo "Vectorial" en este aspecto, lo que indica una mejor capacidad para identificar documentos relevantes dentro del top 10.
- Reciprocidad (Reciprocity): El modelo "Vectorial" presenta un mejor rendimiento en esta métrica, lo que sugiere una mayor eficacia en la identificación de documentos relevantes que también son relevantes para el usuario.
- nDCG en el Top 10 (nDCG@10): El modelo "LSI" muestra un ligerísimo margen superior sobre el modelo "Vectorial", lo que indica una ligeramente mejor capacidad para ordenar los documentos recuperados de manera que los más relevantes aparezcan primero.

### Aspectos a mejorar
- Experimentar con diferentes tamaños de espacio semántico (número de dimensiones) para encontrar un equilibrio entre precisión y eficiencia.
- Considerar la inclusión de palabras con igual significado en las opciones de autocompletado.
- Evaluar la calidad de las palabras consimilar significado utilizadas para saber si son realmente relevantes en el dominio de nuestra colección de documentos.
- El modelo lsi tiene una desventaja que es costo computacional de la factorizacion svd la cual es **n^3** donde **n** la dimensión de ña matriz de documentos a indexar, lo que impide procesar una gran cantidad de documentos.












       
