import org.apache.spark.ml.param.Param
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.NumericType
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.param.ParamMap

class OutlierRemover(override val uid: String) extends Transformer
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("outlierRemover"))

  // Define parameters for input and output columns
  val inputCols: Param[Array[String]] = new Param(this, "inputCols", "Input columns to check for outliers")
  val outputCols: Param[Array[String]] = new Param(this, "outputCols", "Output columns after outlier removal")

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  def getInputCols: Array[String] = $(inputCols)
  def getOutputCols: Array[String] = $(outputCols)

  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {
    // Input validation
    $(inputCols).foreach { col =>
      require(schema(col).dataType.isInstanceOf[NumericType],
        s"Column $col must be numeric type for outlier removal")
    }
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // Filter out rows where values in inputCols are outliers
    var filteredData = dataset.toDF
    $(inputCols).foreach { col =>
      val quantiles = dataset.stat.approxQuantile(col, Array(0.25, 0.75), 0.0)
      val q1 = quantiles(0)
      val q3 = quantiles(1)
      val iqr = q3 - q1
      val lowerBound = q1 - 1.5 * iqr
      val upperBound = q3 + 1.5 * iqr
      filteredData = filteredData.filter(filteredData(col) >= lowerBound && filteredData(col) <= upperBound)
    }
    filteredData.toDF
  }

  override def copy(extra: ParamMap): OutlierRemover = {
    defaultCopy(extra)
  }
}

object OutlierRemover extends DefaultParamsReadable[OutlierRemover]

