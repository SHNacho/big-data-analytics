import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Dataset, functions => F}
import org.apache.spark.ml.param.Param

class TPRxTNR(override val uid: String) extends Evaluator with HasPredictionCol with HasLabelCol{

  def this() = this(Identifiable.randomUID("TPRxTNR"))

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  def evaluate(dataset: Dataset[_]): Double = {
    val trueNegative = F.sum(((F.col(getLabelCol) === 0) && (F.col(getPredictionCol) === 0)).cast(IntegerType))
    val actualNegative = F.sum((F.col(getLabelCol) === 0).cast(IntegerType))
    val truePositive = F.sum(((F.col(getLabelCol) === 1) && (F.col(getPredictionCol) === 1)).cast(IntegerType))
    val actualPositive = F.sum((F.col(getLabelCol) === 1).cast(IntegerType))

    val tnr = trueNegative / actualNegative
    val tpr = truePositive / actualPositive
    val tprtnr = tpr * tnr

    dataset.select(tprtnr).collect()(0)(0).asInstanceOf[Double]
  }

  override def copy(extra: ParamMap): Evaluator = defaultCopy(extra)
}
