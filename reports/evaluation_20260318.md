# Evaluación del modelo de clústeres (clusters_20260318)

- Entrenado con datos hasta **2026-03-18**; evaluado en ventanas de 30 días posteriores (sin fuga).
- Etiqueta: cliente con al menos una transacción is_fraud en la ventana de 30 días (solo para evaluar).
- Métricas evaluadas: num_transacciones_monthly, monto_total_usd_monthly, monto_promedio_usd, monto_mediano_usd, monto_maximo_usd.

## Ventana 2026-03-19 a 2026-04-17

38,991 clientes con actividad, 57 con fraude (tasa base 0.1%).

| Método | Precisión promedio (AP) | ROC-AUC |
|---|---|---|
| Modelo de clústeres | 0.004 | 0.546 |
| fraud_score del banco | 0.636 | 0.843 |
| Aleatorio | 0.002 | 0.535 |
| Reglas del agente (muestra ponderada de 3,057) | 0.002 | 0.558 |

| Métricas sospechosas ≥ | Marcados | Precisión | Recall | F1 | Lift | fraud_score, misma cantidad | Aleatorio, misma cantidad |
|---|---|---|---|---|---|---|---|
| 1 | 17,821 | 0.2% | 50.9% | 0.003 | 1.11 | 0.3% | 0.2% |
| 2 | 8,646 | 0.2% | 28.1% | 0.004 | 1.27 | 0.5% | 0.2% |
| 3 | 4,689 | 0.2% | 14.0% | 0.003 | 1.17 | 0.8% | 0.2% |
| 4 | 330 | 0.3% | 1.8% | 0.005 | 2.07 | 11.2% | 0.0% |
| 5 | 0 | 0.0% | 0.0% | 0.000 | 0.0 | – | – |

| Reglas del agente: puntaje ≥ | Marcados (estimado) | Precisión | Recall |
|---|---|---|---|
| 30 | 3,060 | 0.3% | 17.5% |
| 60 | 13 | 0.0% | 0.0% |

## Ventana 2026-04-18 a 2026-05-17

38,624 clientes con actividad, 59 con fraude (tasa base 0.2%).

| Método | Precisión promedio (AP) | ROC-AUC |
|---|---|---|
| Modelo de clústeres | 0.002 | 0.514 |
| fraud_score del banco | 0.617 | 0.806 |
| Aleatorio | 0.002 | 0.528 |
| Reglas del agente (muestra ponderada de 3,059) | 0.002 | 0.542 |

| Métricas sospechosas ≥ | Marcados | Precisión | Recall | F1 | Lift | fraud_score, misma cantidad | Aleatorio, misma cantidad |
|---|---|---|---|---|---|---|---|
| 1 | 17,504 | 0.2% | 50.8% | 0.003 | 1.12 | 0.3% | 0.2% |
| 2 | 8,356 | 0.1% | 20.3% | 0.003 | 0.94 | 0.5% | 0.2% |
| 3 | 4,487 | 0.1% | 10.2% | 0.003 | 0.88 | 1.0% | 0.2% |
| 4 | 354 | 0.3% | 1.7% | 0.005 | 1.85 | 10.4% | 0.0% |
| 5 | 0 | 0.0% | 0.0% | 0.000 | 0.0 | – | – |

| Reglas del agente: puntaje ≥ | Marcados (estimado) | Precisión | Recall |
|---|---|---|---|
| 30 | 3,145 | 0.2% | 13.6% |
| 60 | 51 | 0.0% | 0.0% |

## Ventana 2026-05-19 a 2026-06-17

39,572 clientes con actividad, 59 con fraude (tasa base 0.1%).

| Método | Precisión promedio (AP) | ROC-AUC |
|---|---|---|
| Modelo de clústeres | 0.002 | 0.581 |
| fraud_score del banco | 0.528 | 0.805 |
| Aleatorio | 0.002 | 0.437 |
| Reglas del agente (muestra ponderada de 3,059) | 0.002 | 0.499 |

| Métricas sospechosas ≥ | Marcados | Precisión | Recall | F1 | Lift | fraud_score, misma cantidad | Aleatorio, misma cantidad |
|---|---|---|---|---|---|---|---|
| 1 | 18,375 | 0.2% | 59.3% | 0.004 | 1.28 | 0.3% | 0.1% |
| 2 | 8,772 | 0.2% | 25.4% | 0.003 | 1.15 | 0.5% | 0.1% |
| 3 | 4,692 | 0.1% | 11.9% | 0.003 | 1.0 | 0.8% | 0.1% |
| 4 | 344 | 0.6% | 3.4% | 0.010 | 3.9 | 9.0% | 0.6% |
| 5 | 0 | 0.0% | 0.0% | 0.000 | 0.0 | – | – |

| Reglas del agente: puntaje ≥ | Marcados (estimado) | Precisión | Recall |
|---|---|---|---|
| 30 | 3,364 | 0.1% | 8.5% |
| 60 | 79 | 0.0% | 0.0% |

## Todas las ventanas juntas

117,187 cliente-ventanas, 175 con fraude.

| Método | AP | ROC-AUC |
|---|---|---|
| Modelo de clústeres | 0.002 | 0.547 |
| fraud_score del banco | 0.593 | 0.818 |
