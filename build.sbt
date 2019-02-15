name := "sparkLr"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.0",
  "org.apache.spark" %% "spark-sql" % "2.3.0",
  "com.rabbitmq" % "amqp-client" % "5.1.1",
  "org.scalactic" %% "scalactic" % "3.0.5",
  "org.scalatest" %% "scalatest" % "3.0.5",
  "com.twitter" %% "finagle-http" % "18.7.0",

  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.12",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.12",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.12",
  //"org.apache.hadoop" %% "hadoop-core" % "1.2.0",
  //"org.apache.hive" %% "hive-exec" % "0.14.0"
  //,"com.datastax.spark" %% "spark-cassandra-connector" % "1.0.0"
  "co.theasi" %% "plotly" % "0.2.0",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.2.2",
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.2.2",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta3",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3"
)

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)