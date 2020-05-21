import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater

object sgdtest {
  def main(args: Array[String]) {
    val inputFile =  "E://eclipse//iris.data"
    val conf = new SparkConf().setAppName("svmtest").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.textFile(inputFile)
    val parsedData = data.map { line => val parts = line.split(',')
      //println(parts.length)
      LabeledPoint( if( parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)
        =="Iris-versicolor") 1.toDouble else
        2.toDouble, Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts
      (2).toDouble,parts(3).toDouble))
    }
    val splits = parsedData.filter { point => point.label != 2 }.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    val numIterations = 1000
    val model = SVMWithSGD.train(training, numIterations)
    model.clearThreshold()
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }
    scoreAndLabels.foreach(println)
     model.setThreshold(0.0)
    scoreAndLabels.foreach(println)
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)

  }
}
