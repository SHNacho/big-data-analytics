import org.apache.spark.sql.{SaveMode, SparkSession, Row, DataFrame}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.NumericType
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.KNNClassifier
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier

/* 
 * Pipelines:
 * 	- scale, ros
 * 	- scale, rus
 * 	- outliers, scale, ros
 * 	- outliers, scale, rus
 * 	- outliers, scale, ros, pca (better)
 * 	- outliers, scale, pca, ros (better between ros and rus)
 * 	- outliers, scale, polynomial, ros (better)
 */


object Main {
	
	def evaluateModels(
		trainData: DataFrame, 
		testData: DataFrame,
		labelCol: String = "label", 
		featuresCol: String = "features"
	){
		// Random Forest
		var randomForest = new RandomForestClassifier()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
		var randomForestModel = randomForest.fit(trainData)
		var randomForestPredictions = randomForestModel.transform(testData)

		// Decision Tree
		var decisionTree = new DecisionTreeClassifier()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
		var decisionTreeModel = decisionTree.fit(trainData)
		var decisionTreePredictions = decisionTreeModel.transform(testData)

		// Logistic Regression
		var logisticRegression = new LogisticRegression()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
		var logisticRegressionModel = logisticRegression.fit(trainData)
		var logisticRegressionPredictions = logisticRegressionModel.transform(testData)
		
		// KNN
		var knn = new KNNClassifier()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
			
		// LightGBM
		var lightGBM = new LightGBMClassifier()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
		var lightGBModel = lightGBM.fit(trainData)
		var lightGBMPredictions = lightGBModel.transform(testData)
			
		// XGBoost
		var XGBoost = new XGBoostClassifier()
			.setFeaturesCol(featuresCol)
			.setLabelCol(labelCol)
		var _XGBoostModel = XGBoost.fit(trainData)
		var XGBoostPredictions = _XGBoostModel.transform(testData)

		// Evaluators
		var tprEvaluator = new TruePositiveRateEvaluator()
			.setLabelCol(labelCol)
			.setPredictionCol("prediction")
		var tnrEvaluator = new TrueNegativeRateEvaluator()
			.setLabelCol(labelCol)
			.setPredictionCol("prediction")
		var binaryEvalutor = new BinaryClassificationEvaluator()
			.setLabelCol(labelCol)
		
		// Print results:
		println("-------Results-------")
		//println("- Random Forest:")
		//println("\t - TPR: " + tprEvaluator.evaluate(randomForestPredictions))
		//println("\t - TNR: " + tnrEvaluator.evaluate(randomForestPredictions))
		//println("- Decision Tree:")
		//println("\t - TPR: " + tprEvaluator.evaluate(decisionTreePredictions))
		//println("\t - TNR: " + tnrEvaluator.evaluate(decisionTreePredictions))
		println("- Logistic Regression:")
		println("\t - TPR: " + tprEvaluator.evaluate(logisticRegressionPredictions))
		println("\t - TNR: " + tnrEvaluator.evaluate(logisticRegressionPredictions))
		println("\t - ROC-AUC: " + binaryEvalutor.evaluate(logisticRegressionPredictions))
		println("- LightGBM:")
		println("\t - TPR: " + tprEvaluator.evaluate(XGBoostPredictions))
		println("\t - TNR: " + tnrEvaluator.evaluate(XGBoostPredictions))
		println("\t - ROC-AUC: " + binaryEvalutor.evaluate(XGBoostPredictions))
	}
	
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    
    val ftrain ="/home/spark/datasets/susy-10k-tra.data" //Local
    val ftest  ="/home/spark/datasets/susy-10k-tst.data" //Local
    //val ftrain ="/home/spark/datasets_complete/susyMaster-Train.data" //Local
    //val ftest ="/home/spark/datasets_complete/susyMaster-Test.data" //Local

    //Leemos el conjunto de entrenamiento
    var dfTra = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(ftrain)


    //Leemos el conjunto de test
    var dfTst = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(ftest)

    //Le damos nombre a la variable clase y la convertimos a en entera
    dfTra=dfTra.withColumnRenamed("_c18","class")
    dfTra=dfTra.withColumn("class",dfTra("class").cast("Integer"))

    dfTst=dfTst.withColumnRenamed("_c18","class")
    dfTst=dfTst.withColumn("class",dfTst("class").cast("Integer"))
		
		dfTra.groupBy("class").count().show

		// Definición de métodos de preprocesamiento
    var variables = dfTra.columns.slice(0, dfTra.columns.length-2)
    
    var outlierRemover = new OutlierRemover()
      .setInputCols(variables)
      .setOutputCols(variables)
    
    var assembler = new VectorAssembler()
      .setInputCols(variables)
      .setOutputCol("features")
      
    var scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
      
    var pca = new PCA()
      .setInputCol("scaled_features")
      .setOutputCol("pca_scaled_features")
      .setK(5)
			
		var featureSelector = new ChiSqSelector()
			.setNumTopFeatures(10)
			.setLabelCol("class")
			.setFeaturesCol("scaled_features")
			.setOutputCol("selected_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 1: ")
		println("\t- Standard Scaler")
		println("\t- Random Under Sampling")

		var pipeline1 = new Pipeline().setStages(Array(assembler, scaler))
		var pipeline1Model = pipeline1.fit(dfTra)
		var preprocData = pipeline1Model.transform(dfTra).select("scaled_features", "class")
		var testData = pipeline1Model.transform(dfTst).select("scaled_features", "class")

		var rus = new RUS("class")
		var preprocDataUndersampled = rus.transform(preprocData)
		
		evaluateModels(preprocDataUndersampled, testData, labelCol = "class", featuresCol = "scaled_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 2: ")
		println("\t- Standard Scaler")
		println("\t- Random Over Sampling")

		// Mismo pipeline que el anterior ...
		var ros = new ROS("class") 
		//var preprocDataOversampled = ros.transform(preprocData)
		//evaluateModels(preprocDataOversampled, testData, labelCol = "class", featuresCol = "scaled_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 3: ")
		println("\t- Outliers removal")
		println("\t- Standard Scaler")
		println("\t- Random Over Sampling")
    println("----------------------------------------------------------------------------")
		
		//var pipeline3 = new Pipeline()
		//	.setStages(Array(outlierRemover, assembler, scaler))
		//var pipeline3Model = pipeline3.fit(dfTra)
		//var preprocData3 = pipeline3Model.transform(dfTra).select("scaled_features", "class")
		//var testData3 = pipeline3Model.transform(dfTst).select("scaled_features", "class")
		//var preprocData3OverSampled = ros.transform(preprocData3)
		//evaluateModels(preprocData3OverSampled, testData3, labelCol = "class", featuresCol = "scaled_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 4: ")
		println("\t- Standard Scaler")
		println("\t- PCA")
		println("\t- Random Over Sampling")
    println("----------------------------------------------------------------------------")
		//var pipeline4 = new Pipeline()
		//	.setStages(Array(assembler, scaler, pca))
		//var pipeline4Model = pipeline4.fit(dfTra)
		//var preprocData4 = pipeline4Model.transform(dfTra).select("pca_scaled_features", "class")
		//var testData4 = pipeline4Model.transform(dfTst).select("pca_scaled_features", "class")

		//var preprocData4OverSampled = ros.transform(preprocData4)

		//evaluateModels(preprocData4OverSampled, testData4, labelCol = "class", featuresCol = "pca_scaled_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 5: ")
		println("\t- Standard Scaler")
		println("\t- Polynomial expansion")
		println("\t- Random Over Sampling")
    println("----------------------------------------------------------------------------")
		var pipeline5 = new Pipeline()
			.setStages(Array(assembler, scaler, featureSelector))
		var pipeline5Model = pipeline5.fit(dfTra)
		var preprocData5 = pipeline5Model.transform(dfTra).select("selected_features", "class")
		var testData5 = pipeline5Model.transform(dfTst).select("selected_features", "class")
		var preprocData5OverSampled = ros.transform(preprocData5)

		evaluateModels(preprocData5OverSampled, testData5, labelCol = "class", featuresCol = "selected_features")

    println("----------------------------------------------------------------------------")
    println("Pipeline 6: ")
		println("\t- Standard Scaler")
		println("\t- Random Over Sampling")
		println("\t- Polynomial expansion")
		println("\t- PCA")
    println("----------------------------------------------------------------------------")
  }
}
