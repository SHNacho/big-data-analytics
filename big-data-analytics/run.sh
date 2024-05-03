#!/bin/bash
#Local
/opt/spark/bin/spark-submit --master local[*] \
    --executor-memory 10G --num-executors 8 \
    --packages com.microsoft.azure:synapseml_2.12:1.0.4,ml.dmlc:xgboost4j-spark_2.12:1.5.2,saurfang:spark-knn:0.3.0\
    --class Main ./target/spark-scala-template-1.0-SNAPSHOT.jar
