organization := sys.env.get("APP_ORGANIZATION").getOrElse("org.company.dev")

name := sys.env.get("APP_NAME").getOrElse("spark-scala-template") // the project's name

version := sys.env.get("APP_VERSION").getOrElse("1.0-SNAPSHOT") // the application version

scalaVersion := sys.env.get("SCALA_VERSION").getOrElse("2.12.19") // version of Scala we want to use (this should be in line with the version of Spark framework)

crossTarget := baseDirectory.value / "target"

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  "spark-scala-template-1.0-SNAPSHOT.jar"
}

val sparkVersion = sys.env.get("SPARK_VERSION").getOrElse("3.5.1")

resolvers += "SynapseML" at "https://mmlspark.azureedge.net/maven"
resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"
resolvers += "XGBoost4J Snapshot Repo" at "https://s3-us-west-2.amazonaws.com/xgboost-maven-repo/"
resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

//addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "ml.dmlc" %% "xgboost4j" % "1.5.2",
  "ml.dmlc" %% "xgboost4j-spark" % "1.5.2",
  "com.microsoft.azure" % "synapseml_2.12" % "1.0.4",
  "saurfang" % "spark-knn" % "0.3.0"
)

