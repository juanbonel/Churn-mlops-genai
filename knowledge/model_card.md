# Ficha del modelo de churn

El objetivo del ejemplo es clasificar si un cliente sintético abandonaría una suscripción (`churn=1`) o permanecería (`churn=0`). Las variables de entrada son antigüedad, cargos mensuales, tickets de soporte, uso de datos, fallas de pago, autopago, satisfacción y tipo de contrato.

Los datos son generados artificialmente con una semilla fija. No representan clientes reales, un mercado específico ni tasas reales de abandono. Las métricas solo sirven para comprobar el pipeline reproducible.

El entrenamiento compara regresión logística y Random Forest con una partición estratificada de entrenamiento/prueba. El candidato con mayor ROC-AUC en el conjunto de prueba se guarda como modelo seleccionado. Para un caso real habría que validar calidad temporal, calibración, costos de falsos positivos y falsos negativos, sesgos, estabilidad y utilidad de negocio.

El resultado del endpoint incluye una probabilidad de churn y una clase con umbral de clasificación por defecto de scikit-learn. No se debe usar para contactar, penalizar o excluir clientes reales.
