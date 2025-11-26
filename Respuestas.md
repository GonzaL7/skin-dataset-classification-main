
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Es necesario porque una MLP solo acepta vectores de entrada de tamaño fijo.
Al redimensionar todas las imágenes a 64×64, garantizamos que el Flatten() produzca siempre un vector de tamaño 64×64×3 = 12288, que coincide con el tamaño de entrada del modelo.

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Albumentations es más rápido, más flexible, tiene transformaciones más avanzadas y está optimizado para imágenes médicas. Por eso es preferible a torchvision.transforms para este TP.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

A.Normalize() estandariza los valores de los píxeles para que tengan media y varianza controladas. Esto evita inestabilidades y permite que la red aprenda más rápido y de forma más estable.

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Necesitamos ToTensorV2() porque PyTorch solo acepta tensores en formato CHW. Albumentations produce imágenes NumPy, así que la conversión final es obligatoria para que el modelo pueda usarlas.

******************************************************
## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Se usa MLP para tener un modelo simple y baseline, pero es limitado porque no aprovecha la estructura espacial de la imagen y escala muy mal. Una CNN es mucho mejor para imágenes y usa muchos menos parámetros.

- ¿Qué hace la capa `Flatten()` al principio de la red?

La capa flatten convierte una matriz en un arreglo de una dimensión. (3x64x64) --> (12288)

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Se usa ReLU porque entrenan más rápido y sin saturación. No se usan Sigmoid/Tanh porque generan gradientes muy pequeños y dificultan el entrenamiento.

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Deberíamos cambiar el parámetro input_size y por lo tanto el tamaño de la primera capa lineal de la MLP para que coincida con el número de píxeles flatten de la nueva imagen.

******************************************************
## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?

Limpia los gradientes acumulados de la iteración anterior para que el backward del batch actual sea correcto.

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Usamos CrossEntropyLoss porque es la loss correcta para clasificación multiclase con logits sin normalizar. Calcula softmax + log-loss en una forma estable y eficiente.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

El batch size modifica el ruido del gradiente, la estabilidad del entrenamiento y la velocidad. Batches pequeños generalizan mejor pero entrenan más lento; batches grandes entrenan más rápido pero pueden 'overfittear'.

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

Sin model.eval(), el modelo se comporta como si estuviera entrenando: Dropout se activa y BatchNorm usa estadísticas del batch. Esto altera las predicciones y genera una validación incorrecta.

******************************************************
## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Que el modelo está 'overfitteando': aprende demasiado bien el entrenamiento (90%) pero no generaliza (70%). Igualmente, siempre se espera un mejor rendimiento en entrenamiento que en validación.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

Precision, recall, F1-score, balanced accuracy y AUC son métricas más relevantes que accuracy, especialmente en problemas médicos. Es más grave pasar por alto a una lesión/patología que diagnosticar mal a un paciente sano.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

La matriz de confusión muestra específicamente qué clases se confunden entre sí, cuántos falsos positivos / falsos negativos hay y dónde falla el modelo, cosa que la accuracy no revela.

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

Precision: qué proporción de predicciones positivas fueron correctas.

Recall: qué proporción de positivos reales fueron detectados.

F1-score: promedio armónico entre precision y recall, balanceando ambas.

******************************************************
## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

TensorBoard permite monitorear el entrenamiento visualmente: curvas de pérdida, accuracy, pesos, imágenes y métricas. Facilita detectar problemas y comparar modelos de forma clara y rápida.

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

add_scalar: registra valores numéricos;
add_image: registra imágenes (útil en modelos de visión);
add_text: registra texto como notas o descripciones.

Cada uno aparece en un panel distinto de TensorBoard.

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Porque permite verificar cómo el modelo ve las imágenes, si las transformaciones están bien aplicadas y si la CNN aprende patrones correctos o se confunde.

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Guardando cada experimento en un log_dir distinto, TensorBoard permite cargar todos a la vez y superponer curvas de loss/accuracy para comparar rendimiento entre modelos.

******************************************************
## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Solo hay que cambiar la capa final para que tenga 100 neuronas y construir el modelo con num_classes=100. El resto del pipeline funciona igual. De todos modos, una MLP no funcionaría y habría que usar una CNN sí o sí.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Una CNN es más adecuada porque usa convoluciones que capturan patrones locales, requiere menos parámetros, es robusta a traslaciones y generaliza mucho mejor que una MLP en tareas de visión.

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

El modelo tendría mucho overfitting y memorizaría los datos de entrenamiento. No podría generalizar, por lo que fallaría mucho en validación.

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Lo único que cambia es la cantidad de canales, por lo que las imágenes pasarían de (64x64x3)--->(64x64x1). Habría que ajustar el input_size de la MLP y el primer Conv2d de la CNN.

******************************************************
## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

Regularización es el conjunto de técnicas usadas para reducir el overfitting y mejorar la capacidad del modelo de generalizar a datos no vistos. Por ejemplo, weight decay, early stoppage, etc.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Dropout elimina neuronas aleatoriamente para evitar co-adaptación, mientras que L2 penaliza pesos grandes para suavizar el modelo. Son regularizadores complementarios.

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

BatchNorm (Batch normalization)  normaliza las activaciones de cada capa, estabilizando los gradientes y acelerando el entrenamiento. Usa la distribución normal.



- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

BatchNorm acelera el entrenamiento al estabilizar activaciones y gradientes, permitiendo learning rates más altos y convergencia más veloz.

- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Sí. BatchNorm introduce ruido estocástico en las activaciones (por depender del batch), lo cual reduce el overfitting.

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

El principal indicador va a ser una precisión muy alta en entrenamiento pero baja o estancada en validación. También se pueden ver pesos muy altos.

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

La regularización reduce la complejidad del modelo: Dropout evita que memorice, L2 limita la magnitud de los pesos, BatchNorm estabiliza activaciones, Data augmentation aumenta la diversidad del dataset, Early stopping evita sobreentrenamiento. Todo esto fuerza al modelo a aprender patrones generales, no ruido.

******************************************************
### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

BatchNorm mejoró un poco el rendimiento en validación y fue un poco más rápido en converger. Se ve que empeoró un poco la precisión de entrenamiento pero ganó en validación.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

Mejoró 2 puntos respecto a no tener nada, pero mejoró más significativamente si lo comparamos a cuando usamos dropout con p=0.5. Dropout con p=0.5 empeoró el rendimiento, lo que puede indicar que no había overfitting en un principio.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

La combinación de BatchNorm + L2 (weight decay) fue la que produjo la mejor generalización: aunque el accuracy de entrenamiento cayó, la pérdida y el accuracy en validación mejoraron respecto al modelo base.

- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Noté cambios respecto a la versión base. Mejoró de 1.387 a 1.09.

******************************************************
## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?

Es importante porque evita gradientes explosivos o que desaparecen, estabiliza las activaciones y permite que la red pueda aprender desde el inicio.

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

La red no aprende porque todas las neuronas reciben los mismos gradientes y evolucionan igual. No se rompe la simetría y la red se vuelve incapaz de aprender funciones no triviales.

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

Xavier es apropiada para Tanh/Sigmoid, mientras que He (Kaiming) es mejor para ReLU porque mantiene estable la varianza cuando la mitad de las activaciones se anulan.



- ¿Por qué en una red con ReLU suele usarse la inicialización de He?

Suele usarse la inicialización He en las redes con ReLU porque este método de inicialización corrige la pérdida generada por "cortar" la mitad de las activaciones con la ReLU.

- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Las capas de pesos y bias necesitan una inicialización explícita. Las funciones como ReLU, tanh, sigmoid, MaxPool/AvgPool, flatten, no necesitan inicialización. Algunas capas ya vienen como BatchNorm tienen parámetros entrenables, pero tienen una inicialización implícita que funciona bien. 

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

La inicialización He fue la más rápida.

- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

Sí, la inicialización aleatoria fue inestable.

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?

Es claro que el modelo tiene mejor precisión cuando es inicializado con He, luego con Xavier, y tiene el peor rendimiento cuando es inicializado aleatoriamente. Me sorprendió que la precisión fuera menor a lo visto sin inicializar (50% vs 57%). Esto se debe a que PyTorch hace una inicialización implícita que está mejor calibrada que la mía (Kaiming Uniform). 

- ¿Por qué `bias` se suele inicializar en cero?

Es un punto de partida simple y claro, y no tiene el problema de la simetría porque los pesos están inicializados en valores != 0.
