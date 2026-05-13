# Conclusiones Finales — Clasificación de Imágenes Médicas
### **Autoras:** Marta Esteban, Natalia Jiménez y Sofía Sebastián 
**Asignatura:** Análisis de Datos No Estructurados  
**Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle  
**Tarea:** Clasificación binaria NORMAL vs PNEUMONIA  

**Enlace al repositorio:** https://github.com/sofiasebastiansanchez/Proyecto-Imagen.git 
---

## 1. Contexto del problema

El dataset Chest X-Ray Pneumonia está compuesto por radiografías de tórax etiquetadas en dos clases: NORMAL y PNEUMONIA. El análisis exploratorio (EDA) reveló dos condicionantes que han influido en todas las decisiones de modelado posteriores: un **desbalanceo de ~3:1** en el split de entrenamiento (74.3% PNEUMONIA vs 25.7% NORMAL) y un **split de validación artificialmente reducido a 16 imágenes**, lo que hace que las métricas de validación durante el entrenamiento sean inestables y no representativas. Las imágenes son en escala de grises (1 canal), con resolución variable normalizada a 224×224, y presentan diferencias estadísticas entre clases ya a nivel de píxel crudo (PNEUMONIA tiende a mayor brillo medio por los infiltrados pulmonares).

Dado que un modelo trivial que predijera siempre PNEUMONIA alcanzaría un 74% de accuracy sin aprender nada, se descartó la accuracy como métrica principal. En todos los notebooks se utilizaron **AUC-ROC y F1-score** como métricas de referencia, aplicando class_weight y data augmentation para compensar el desbalanceo.

---

## 2. Progresión de resultados

| Modelo | AUC-ROC | Sensibilidad | Especificidad | FN |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression (HOG+PCA) | 0.9270 | 0.9900 | 0.4300 | 2 |
| CNN from scratch v1 | 0.9497 | 0.9800 | 0.6450 | 6 |
| CNN from scratch v2 (GAP) | 0.9356 | 0.9128 | 0.8034 | 33 |
| VGG16 FT (block5) | 0.8816 | 0.8692 | 0.7521 | 51 |
| VGG16 v2 FT (block4+block5) | 0.9234 | 0.9821 | 0.4573 | 7 |
| ResNet50 FT | 0.9555 | 0.9282 | 0.8632 | 28 |
| MobileNetV2 FT | 0.6898 | 0.5667 | 0.7094 | 169 |
| MobileNetV2 v2 FT (norm. simple) | 0.6352 | 0.9385 | 0.1068 | 24 |
| DenseNet121 FT | 0.6186 | 1.0000 | 0.0000 | 0 |
| EfficientNetB0 FT (umbral=0.50) | 0.9547 | 0.9615 | 0.7479 | 15 |
| **EfficientNetB0 FT (umbral=0.10)** | **0.9547** | **0.9897** | **0.5299** | **4** |

---

## 3. Hallazgos por fase

### ML Clásico — Baseline sólido con limitaciones claras

Logistic Regression con HOG+PCA alcanzó AUC 0.927 y solo 2 falsos negativos, demostrando que existe señal estadística suficiente para discriminar entre clases incluso sin Deep Learning. Sin embargo, la especificidad del 43% (134 falsos positivos) revela que el feature engineering manual no captura la complejidad espacial de las radiografías. LinearSVC y Random Forest obtuvieron peores resultados en todas las métricas relevantes, confirmando a Logistic Regression como el mejor modelo clásico y el **baseline a superar**.

### Deep Learning from Scratch — Mejora real sin ingeniería de features

La CNN v1 mejoró el AUC hasta 0.9497 y redujo los falsos positivos de 134 a 83, aprendiendo representaciones espaciales directamente de los píxeles sin feature engineering manual. La inestabilidad de las curvas de validación en las primeras épocas es consecuencia directa del val set de solo 16 imágenes, no de un problema del modelo. La CNN v2 (GAP + Dropout 0.6), con muchos menos parámetros, no consiguió mejorar los resultados: la mayor regularización ralentizó la convergencia sin aportar beneficio en test, lo que sugiere que con ~5.000 imágenes el modelo v1 ya aprovecha bien la capacidad disponible.

### Transfer Learning con ImageNet — El domain gap como obstáculo principal

Los modelos preentrenados en ImageNet no superaron en general a la CNN entrenada desde cero. La causa principal es el **domain gap**: los filtros de ImageNet están optimizados para fotografías naturales en color, mientras que las radiografías son imágenes en escala de grises con patrones anatómicos que no aparecen en ningún dataset de imágenes naturales. La conversión pseudo-RGB (replicar el canal tres veces) no genera información real y dificulta la adaptación.

- **MobileNetV2** colapsó en todas sus variantes, prediciendo casi siempre PNEUMONIA. Su arquitectura de convoluciones separables en profundidad es especialmente sensible a la distribución de entrada, y ni cambiar la normalización resolvió el problema de fondo.
- **VGG16** mejoró al descongelar más capas (block4+block5), alcanzando AUC 0.9234 y solo 7 FN, pero a costa de una especificidad del 46% — demasiados falsos positivos para uso clínico.
- **ResNet50 FT** fue el único modelo de Transfer Learning que igualó a la CNN from scratch (AUC 0.9555), gracias a sus skip connections que facilitan el flujo de gradiente durante el fine-tuning y permiten una mejor adaptación al dominio médico.
- **DenseNet121** sufrió overfitting severo, prediciendo siempre PNEUMONIA en test a pesar de buenas métricas de entrenamiento. Sus dense connections favorecen la memorización cuando el dataset es pequeño y el dominio difiere mucho del preentrenamiento.

### EfficientNetB0 con mejoras metodológicas — Mejor resultado clínico

El notebook final introdujo tres correcciones metodológicas respecto a los anteriores: reconstrucción del split de validación con un 15% estratificado del train (~782 imágenes), preprocesado unificado aplicado de forma consistente a train, val y test, y búsqueda del umbral óptimo de decisión sobre validación maximizando F1. Estas mejoras, combinadas con EfficientNetB0 como backbone, permitieron obtener el mejor resultado clínico del proyecto: con umbral 0.10, **solo 4 pacientes con neumonía no detectados sobre 390 casos reales** (sensibilidad 98.97%).

---

## 4. Conclusión clínica

En detección de neumonía, el coste de un falso negativo (paciente enfermo clasificado como sano) es muy superior al de un falso positivo. Bajo este criterio, el mejor modelo del proyecto es **EfficientNetB0 FT con umbral optimizado (0.10)**: 4 falsos negativos, sensibilidad del 98.97% y AUC-ROC de 0.9547. La reducción de especificidad que conlleva el umbral bajo (52.99%) es un trade-off clínicamente aceptable en un sistema de cribado, donde el objetivo es no dejar pasar ningún enfermo y los falsos positivos se resuelven con revisión médica posterior.

En un contexto clínico real, este sistema operaría como **herramienta de apoyo al diagnóstico bajo supervisión médica**, no como sustituto del criterio clínico.

---

## 5. Limitaciones y trabajo futuro

La limitación principal de todo el experimento es el uso de pesos preentrenados en ImageNet en lugar de en datos radiológicos. Modelos como **CheXNet** (DenseNet121 preentrenado en ChestX-ray14) o backbones de **RadImageNet** eliminarían el domain gap y previsiblemente mejorarían tanto la sensibilidad como la especificidad sin necesidad de ajustar el umbral de decisión.

Otras líneas de mejora identificadas:

- **Ensemble ResNet50 FT + EfficientNetB0**: ambos modelos tienen perfiles de error complementarios (ResNet50 tiene mejor especificidad, EfficientNetB0 mejor sensibilidad con umbral bajo).
- **Data augmentation específico para radiografías**: variaciones de contraste, ruido gaussiano, transformaciones elásticas que simulen artefactos reales de adquisición.
- **Regularización más agresiva para DenseNet121**: dropout, label smoothing y weight decay podrían evitar el colapso observado.
- **Validación clínica externa**: evaluar el modelo sobre un dataset externo (distinto hospital, distinto equipo de rayos) para estimar la degradación de rendimiento en producción real.
