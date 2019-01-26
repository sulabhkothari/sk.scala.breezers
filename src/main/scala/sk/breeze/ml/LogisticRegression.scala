package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, sigmoid}
import breeze.optimize._
import breeze.plot.plot
import sk.breeze.ml.Util.{TrainingData, isConverged, prependOnesColumn}

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val (trainingDataPos, trainingDataNeg) = Util.prepareTrainingData(1000)
    val trainingData = TrainingData(trainingDataNeg, trainingDataPos)
    Util.plotXY(trainingData.trainingData)
    scala.io.StdIn.readLine()

    val gd = StochasticGradientDescent[DenseVector[Double]]()
    val state1 = gd.minimizeAndReturnState(f(trainingData), trainingData.randomTheta)
    val cs1 = gd.convergenceCheck(state1, state1.convergenceInfo)
    println(":::::::::::::::::>" + cs1.get.reason)
    println("=================>" + state1.x)

    val minThetaGd = gradientDescent(trainingData)
    println("GD :=>"+minThetaGd)
    println("SGD :=>"+state1.x)

    println(state1.x)
    println(hTheta(minThetaGd, DenseVector(1, 6000, 6000)))
    println(hTheta(minThetaGd, DenseVector(1, -6000, -6000)))

    println(hTheta(state1.x, DenseVector(1, 6000, 6000)))
    println(hTheta(state1.x, DenseVector(1, -6000, -6000)))

    val newData = Util.prepareClassificationData(5)
    val y = sigmoid(newData * state1.x)
    val plotData = DenseMatrix.create(newData.rows, newData.cols + 1, newData.data ++ y.data)
    //println(plotData)
    Util.plotXY(plotData)

    scala.io.StdIn.readLine()
    val y2 = sigmoid(newData * minThetaGd)
    val plotData2 = DenseMatrix.create(newData.rows, newData.cols + 1, newData.data ++ y.data)
    Util.plotXY(plotData)
  }

  def hTheta(theta: DenseVector[Double], x: DenseVector[Double]) = {
    sigmoid(theta.t * x)
  }

  val f = (td: TrainingData) => new DiffFunction[DenseVector[Double]] {
    def calculate(theta: DenseVector[Double]) = {
      (cost(td, theta), gradient(td, theta))
    }
  }

  def cost(td: TrainingData, theta: DenseVector[Double]) = {
    sum(-log(sigmoid(td.xPos * theta))) + sum(-log(td.negOnes - sigmoid(td.xNeg * theta)))
  }

  def gradient(td: TrainingData, theta: DenseVector[Double]) = {
    ((sigmoid(td.x * theta) - td.y).t * td.x).t
  }

  def gradientDescent(td: TrainingData): DenseVector[Double] = {
    var theta = DenseVector.zeros[Double](td.randomTheta.length)
    var thetaTemp = td.randomTheta
    val count: Double = td.trainingData.rows
    val alpha: Double = 0.05 / count
    val precision = 0.000001 // count

    while (!isConverged(theta, thetaTemp, precision)) {
      theta = thetaTemp
      val diff = gradient(td, theta)
      //println("==>" + thetaTemp)
      thetaTemp = theta - (diff :* alpha :/ (2 * count))
    }
    thetaTemp
  }
}
