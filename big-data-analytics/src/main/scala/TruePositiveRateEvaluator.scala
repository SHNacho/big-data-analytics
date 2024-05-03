import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Dataset, functions => F}
import org.apache.spark.ml.param.Param

class TruePositiveRateEvaluator(override val uid: String) extends Evaluator with HasPredictionCol with HasLabelCol{

  def this() = this(Identifiable.randomUID("TruePositiveRateEvaluator"))

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  def evaluate(dataset: Dataset[_]): Double = {
    val truePositive = F.sum(((F.col(getLabelCol) === 1) && (F.col(getPredictionCol) === 1)).cast(IntegerType))
    val predictedPositive = F.sum((F.col(getPredictionCol) === 1).cast(IntegerType))
    val actualPositive = F.sum((F.col(getLabelCol) === 1).cast(IntegerType))

    val recall = truePositive / actualPositive

    dataset.select(recall).collect()(0)(0).asInstanceOf[Double]
  }

  override def copy(extra: ParamMap): Evaluator = defaultCopy(extra)
}
