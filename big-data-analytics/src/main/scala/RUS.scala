import org.apache.spark.sql.DataFrame

class RUS(_labelCol: String) {
  var labelCol: String = _labelCol
  
	def transform(df: DataFrame): DataFrame = {
    var undersample: DataFrame = df.limit(0)  //empty DF

    val train_positive = df.where(s"$labelCol == 1")
    val train_negative = df.where(s"$labelCol == 0")
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble

    if (num_pos > num_neg) {
      val fraction = num_neg / num_pos
      undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
    } else {
      val fraction = num_pos / num_neg
      undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
    }
    undersample.repartition(df.rdd.getNumPartitions)
	}
}
