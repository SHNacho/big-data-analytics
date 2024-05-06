import org.apache.spark.sql.DataFrame

class ROS(_labelCol: String = "label", _overRate: Double = 1.0) {
  var labelCol: String = _labelCol
	var overRate: Double = _overRate

	def transform(df: DataFrame): DataFrame = {
		var oversample: DataFrame = df.limit(0) //empty DF

    val train_positive = df.where(s"$labelCol == 1")
    val train_negative = df.where(s"$labelCol == 0")
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble

    if (num_pos > num_neg) {
      val fraction = (num_pos * this.overRate) / num_neg
      oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
    } else {
      val fraction = (num_neg * this.overRate) / num_pos
      oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
    }
    oversample.repartition(df.rdd.getNumPartitions)
	}
}
