# Guía operativa

## Flujo esperado

1. Ejecutar `python -m churn_project.train` para regenerar los datos sintéticos, entrenar candidatos, registrar parámetros y métricas en MLflow, y guardar el modelo seleccionado.
2. Ejecutar `uvicorn churn_project.api:app --reload` para iniciar el endpoint local. La ruta `/health` informa si existe un modelo entrenado; `/predict` recibe features y devuelve clase y probabilidad.
3. Ejecutar `python -m churn_project.monitor --simulate-shift` para comparar un cohorte sintético con drift contra `data/reference.csv` y guardar un reporte en `reports/drift_report.json` y `.csv`.
4. Consultar `POST /assistant` para recuperar documentación local y, si se envían atributos, usar la herramienta `score_customer`. Sin `OPENAI_API_KEY` opera en modo demo determinista. Con credenciales configura un LLM compatible y habilita generación y selección de herramientas.

## Interpretación del drift

El monitor usa Population Stability Index (PSI) por variable. En esta demo el umbral predeterminado es 0.20 y se puede modificar. Es una heurística configurable, no una norma universal. PSI detecta cambios en la distribución de las features; no demuestra que las etiquetas hayan cambiado ni que el rendimiento del modelo haya empeorado. En producción se necesitan datos etiquetados, evaluación de calidad y un proceso aprobado de reentrenamiento.
