# tfm-ucm-crypto-whale-trades
Pipeline experimental y escalable para análisis avanzado de grandes operaciones de compra de criptomonedas y predicción de su impacto en la cotización. Desarrollado en Databricks + Docker (Spark, Python). Ingesta por API REST y Websocket, almacenamiento Delta Lake siguiendo la Arquitectura Medallion (raw, bronze, silver, gold)

Folders:
  - exchange_info: contiene el codigo para obtener y procesar la información de las monedas con las que operar en los exchanges binance, okx y bybit.
  - ETL:calculate_score: contiene el código para el cálculo del score personalizado relativo a la variabilidad de la cotización de una criptomoneda en base a whale trades.
  - ETL+Train: contiene 5 notebooks para la extraccion y proceso de la informacion necesaria para entrenar el modelo predictivo asi como su entrenamiento
  - binance_wesocket: contiene las componentes del contenedor desarrollado que ejecuta un codigo python que se descarga el modelo entrenado por el pipeline ETL+Train, conecta con el websocket de binance y realiza predicciones con mínima latencia.

