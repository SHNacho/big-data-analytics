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
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.Model
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import scala.collection.mutable.Map
import java.nio.channels.Pipe
import org.apache.spark.ml.classification.ProbabilisticClassifier
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.classification.ClassificationModel
import shapeless.Tuple
import scala.collection.mutable.ArrayBuffer

object Main {
  def main(args: Array[String]): Unit = {
		// Start spark session
    val spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
	
		val bestScores: Map[String, Double] = Map()
    
		// Read data
    //val ftrain ="/home/spark/datasets/susy-10k-tra.data" //Local
    //val ftest  ="/home/spark/datasets/susy-10k-tst.data" //Local
    val ftrain ="/home/spark/datasets_complete/susyMaster-Train.data" //Local
    val ftest ="/home/spark/datasets_complete/susyMaster-Test.data" //Local

    var dfTra = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(ftrain)
		
    var dfTst = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(ftest)

		// Change class name
    dfTra=dfTra.withColumnRenamed("_c18","label")
    dfTra=dfTra.withColumn("label",dfTra("label").cast("Integer"))
		
		dfTra.groupBy("label").count().show()

    dfTst=dfTst.withColumnRenamed("_c18","label")
    dfTst=dfTst.withColumn("label",dfTst("label").cast("Integer"))
		

		// Preprocessing methods
    var variables = dfTra.columns.slice(0, dfTra.columns.length-2)
    
    var outlierRemover = new OutlierRemover()
      .setInputCols(variables)
      .setOutputCols(variables)
    
    var assembler = new VectorAssembler()
      .setInputCols(variables)
      .setOutputCol("assembled_features")
      
    var standardScaler = new StandardScaler()
      .setInputCol("assembled_features")
      .setOutputCol("scaled_features")
		
		var minMaxScaler = new MinMaxScaler()
      .setInputCol("assembled_features")
      .setOutputCol("scaled_features")
      
    var pca = new PCA()
      .setInputCol("scaled_features")
      .setOutputCol("pca_scaled_features")
      .setK(5)
			
		var rus = new RUS("label")
		var ros = new ROS("label")
		
		// Check which model perform better with basic preprocessing
		// and select the top 2 to compare in the test.
		
		println("Finding top 2 methods...")


		// Preproces data
		var basicPipeline = new Pipeline().setStages(Array(assembler, standardScaler))
		var basicPreprocData = basicPipeline.fit(dfTra).transform(dfTra)
		basicPreprocData = rus.transform(basicPreprocData)
		
		// Logistic Regression
		println("----------- Logistic Regression -----------")
		var lr = new LogisticRegression()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")
		
		var lrParamGrid = new ParamGridBuilder()
			.addGrid(lr.elasticNetParam, Array(0.0, 0.3, 0.5, 0.8, 1.0))
			.addGrid(lr.maxIter, Array(100, 300, 600))
			.build()
			
		
		var result: (ParamMap, Double) = getBestModel(lr, lrParamGrid, basicPreprocData)
		var bestLr = result._1
		var bestLrScore = result._2
		bestScores += ("Logistic Regression" -> bestLrScore)

		// Decision Tree
		println("----------- Decision Tree -----------")
		var dt = new DecisionTreeClassifier()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")

		var dtParamGrid = new ParamGridBuilder()
			.addGrid(dt.maxDepth, Array(3, 5, 7, 10))
			.addGrid(dt.minInfoGain, Array(0.0, 0.15, 0.3))
			.build()
		
		result = getBestModel(dt, dtParamGrid, basicPreprocData)
		var bestDt = result._1
		var bestDtScore = result._2
		bestScores += ("Decision Tree" -> bestDtScore)

		// Random Forest
		println("----------- Random Forest -----------")
		var rf = new RandomForestClassifier()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")
			
		var rfParamGrid = new ParamGridBuilder()
			.addGrid(rf.impurity, Array("entropy", "gini"))
			.addGrid(dt.maxDepth, Array(3, 5, 7, 10))
			.addGrid(dt.minInfoGain, Array(0.0, 0.15, 0.3))
			.build()
			
		result = getBestModel(rf, rfParamGrid, basicPreprocData)
		var bestRf = result._1
		var bestRfScore = result._2
		bestScores += ("Random Forest" -> bestRfScore)
		
		// Gradient Boosting Tree
		println("----------- Gradient Boosted Tree -----------")
		var gbt = new GBTClassifier()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")
			
		var gbtParamGrid = new ParamGridBuilder()
			.addGrid(gbt.maxDepth, Array(5, 7, 10))
			.addGrid(gbt.maxIter, Array(20, 30))
  		.addGrid(gbt.subsamplingRate, Array(0.6, 0.8, 1.0))
  		//.addGrid(gbt.stepSize, Array(0.1, 0.05))
			.build()
		
		result = getBestModel(gbt, gbtParamGrid, basicPreprocData)
		var bestGbt = result._1
		var bestGbtScore = result._2
		bestScores += ("GBT" -> bestGbtScore)
		
		// LightGBM
		println("----------- LightGBM -----------")
		var lgbm = new LightGBMClassifier()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")
			.setVerbosity(-2)
			
		var lgbmParamGrid = new ParamGridBuilder()
			.addGrid(lgbm.learningRate, Array(0.1, 0.3, 0.5))
			.addGrid(lgbm.numIterations, Array(100, 300, 500))
			.addGrid(lgbm.maxDepth, Array(3, 5, 7, 10))
			.build()
		
		result = getBestModel(lgbm, lgbmParamGrid, basicPreprocData)
		var bestLgbm = result._1
		var bestLgbmScore = result._2
		bestScores += ("LightGBM" -> bestLgbmScore)

		// XGBoost
		println("----------- XGBoost -----------")
		var xgboost = new XGBoostClassifier()
			.setFeaturesCol("scaled_features")
			.setLabelCol("label")
		
		var xgbParamGrid = new ParamGridBuilder()
  		.addGrid(xgboost.eta, Array(0.1, 0.05))
  		// Maximum depth of trees
  		.addGrid(xgboost.maxDepth, Array(4, 6, 8, 10))
  		// Minimum child weight
  		.addGrid(xgboost.minChildWeight, Array(1.0, 2.0, 5.0))
  		// Fraction of features to consider for each tree
  		.addGrid(xgboost.colsampleBytree, Array(0.6, 0.8, 1.0))
  		// Minimum loss reduction required to make a further partition
  		.addGrid(xgboost.gamma, Array(0.0, 0.1, 0.2))
			.build()
			
		result = getBestModel(xgboost, xgbParamGrid, basicPreprocData)
		var bestXgboost = result._1
		var bestXgboostScore = result._2
		bestScores += ("XGBoost" -> bestXgboostScore)
		
		bestScores.foreach { case (key, value) =>
			println(s"$key -> AUC ROC = $value")
		}
		
		// Best two models:
		// - LightGBM
		// - GBT
		// Trying different preprocessing pipelines ...
		// Train with the whole train dataset and predict on test
		
		// ========  Best Models ========
		lgbm = new LightGBMClassifier()
		gbt = new GBTClassifier()

		var pipelines: ArrayBuffer[String] = ArrayBuffer()
		var lgbmResults: ArrayBuffer[Tuple3[Double, Double, Double]] = ArrayBuffer()
		var gbtResults: ArrayBuffer[Tuple3[Double, Double, Double]] = ArrayBuffer()
		
		// Pipeline 1
		// - scale
		// - RUS
		
		pipelines += "Standard scaler - RUS"
		
		var pipeline = new Pipeline()
			.setStages(Array(assembler, standardScaler))
		
		var pipelineModel = pipeline.fit(dfTra)
		var trainData = pipelineModel.transform(dfTra)
		var testData = pipelineModel.transform(dfTst)
		trainData = rus.transform(trainData)	
		
		lgbmResults += evalModel(lgbm, bestLgbm, "scaled_features", trainData, testData)
		gbtResults += evalModel(gbt, bestGbt, "scaled_features", trainData, testData)
		
		printResults(pipelines, lgbmResults, gbtResults)

		// Pipeline 2
		// - scale
		// - ROS
		
		pipelines += "Standard scaler - ROS"
		
		pipeline = new Pipeline()
			.setStages(Array(assembler, standardScaler))
		
		pipelineModel = pipeline.fit(dfTra)
		trainData = pipelineModel.transform(dfTra)
		testData = pipelineModel.transform(dfTst)
		trainData = ros.transform(trainData)	
		
		lgbmResults += evalModel(lgbm, bestLgbm, "scaled_features", trainData, testData)
		gbtResults += evalModel(gbt, bestGbt, "scaled_features", trainData, testData)
		
		printResults(pipelines, lgbmResults, gbtResults)
		
		// Pipeline 3
		// 	- Remove outliers
		//	- Standard Scaler
		// 	- ROS 
		pipelines += "Remove outliers - Standard scaler - ROS"
		pipeline = new Pipeline()
			.setStages(Array(outlierRemover, assembler, standardScaler))

		pipelineModel = pipeline.fit(dfTra)
		trainData = pipelineModel.transform(dfTra)
		testData = pipelineModel.transform(dfTst)
		trainData = ros.transform(trainData)	
		
		lgbmResults += evalModel(lgbm, bestLgbm, "scaled_features", trainData, testData)
		gbtResults += evalModel(gbt, bestGbt, "scaled_features", trainData, testData)
		
		printResults(pipelines, lgbmResults, gbtResults)

		// Pipeline 4
		// 	- Standard Scaler 
		//	- ROS
		// 	- PCA
		pipelines += "Standard scaler - ROS - PCA"
		pipeline = new Pipeline()
			.setStages(Array(assembler, standardScaler, pca))

		pipelineModel = pipeline.fit(dfTra)
		trainData = pipelineModel.transform(dfTra)
		testData = pipelineModel.transform(dfTst)
		trainData = ros.transform(trainData)	
		
		lgbmResults += evalModel(lgbm, bestLgbm, "pca_scaled_features", trainData, testData)
		gbtResults += evalModel(gbt, bestGbt, "pca_scaled_features", trainData, testData)
		
		printResults(pipelines, lgbmResults, gbtResults)

		// Pipeline 5
		// 	- MinMaxScaler
		//	- RUS
		pipelines += "MinMax scaler - RUS"
		pipeline = new Pipeline()
			.setStages(Array(assembler, minMaxScaler, pca))

		pipelineModel = pipeline.fit(dfTra)
		trainData = pipelineModel.transform(dfTra)
		testData = pipelineModel.transform(dfTst)
		trainData = ros.transform(trainData)	
		
		lgbmResults += evalModel(lgbm, bestLgbm, "scaled_features", trainData, testData)
		gbtResults += evalModel(gbt, bestGbt, "scaled_features", trainData, testData)
		
		printResults(pipelines, lgbmResults, gbtResults)
  }
	
	
	def getBestModel(model: Estimator[_], paramGrid: Array[ParamMap], data: DataFrame): (ParamMap, Double) = {
		// Evaluator
		var tprxtnr = new TPRxTNR()
			.setLabelCol("label")
		val trainValidationSplit = new TrainValidationSplit()
  			.setEstimator(model)
  			.setEvaluator(tprxtnr)
  			.setEstimatorParamMaps(paramGrid)
  			// 80% of the data will be used for training and the remaining 20% for validation.
  			.setTrainRatio(0.8)
  			// Evaluate up to 2 parameter settings in parallel
  			.setParallelism(2)
			
		val fittedModel = trainValidationSplit.fit(data)
		
		// Get the best model parameters
		val bestParams: ParamMap = fittedModel.bestModel.extractParamMap()
		
		//(fittedModel.bestModel, fittedModel.validationMetrics.max)
		(bestParams, fittedModel.validationMetrics.max)
	}
	
	/* 
	 * Función que entrena y evalua los modelos sobre el conjunto de Test.
	 * Return:
	 * - True Positive Rate
	 * - True Negative Rate
	 * - ROC-AUC score
	 */
	def evalModel(
		classifier: Classifier[_ , _, _ <: ClassificationModel[_, _]], 
		paramMap: ParamMap, featuresCol: String,  
		trainData: DataFrame, testData: DataFrame): (Double, Double, Double) = {

		classifier.setFeaturesCol(featuresCol)
		var model: Model[_] = classifier.fit(trainData, paramMap)
		var prediction: DataFrame = model.transform(testData)

		var tprEvaluator = new TruePositiveRateEvaluator()
			.setLabelCol("label")
			.setPredictionCol("prediction")
		var tnrEvaluator = new TrueNegativeRateEvaluator()
			.setLabelCol("label")
			.setPredictionCol("prediction")

		var tpr: Double = tprEvaluator.evaluate(prediction)
		var tnr: Double = tprEvaluator.evaluate(prediction)
		var tpr_tnr: Double = tpr * tnr
			
		return(
			tpr,
			tnr,
			tpr_tnr
		)
	}

	/* 
	 * Función para mostrar los resultados finales de las predicciones sobre Test
	*/
	def printResults(pipelines: ArrayBuffer[String], lgbmResults: ArrayBuffer[_ <: Tuple3[_, _, _]], gbtResults: ArrayBuffer[_ <: Tuple3[_,_,_]]) {
		pipelines.zip(lgbmResults).zip(gbtResults).foreach { case ((pipeline, lgbm), gbt) =>
			println(s"Pipeline: $pipeline")
			println(s"\t-LightGBM:")
			println(s"\t\t-TPR: ${lgbm._1}")
			println(s"\t\t-TNR: ${lgbm._2}")
			println(s"\t\t-TPR x TNR: ${lgbm._3}")
			println(s"\t-GBT:")
			println(s"\t\t-TPR: ${gbt._1}")
			println(s"\t\t-TNR: ${gbt._2}")
			println(s"\t\t-TPR x TNR: ${gbt._3}")
		}
	}
}
