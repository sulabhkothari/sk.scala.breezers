package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import breeze.optimize.DiffFunction
import breeze.plot.plot

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val trainingData = prepareClassificationSample(300)

    //Util.plotXY(trainingData)
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val theta = DenseVector.rand(trainingData.cols - 1, normal01)
    val x = trainingData(*, 0 to 1).underlying
    val y = trainingData(*, 2).underlying
    println(theta)
    println(sigmoid(theta))
    //println(trainingData(*,0 to 1))

    val result = (hTheta(theta, x) - y).t * x
    println(theta - result.t)
  }

//  val f = (x:DenseMatrix[Double], y:DenseVector[Double]) => new DiffFunction[DenseVector[Double]] {
//    def calculate(theta: DenseVector[Double]) = {
//      (hTheta(theta, x) - y).t * x
//    }
//  }

  def hTheta(theta: DenseVector[Double], x: DenseMatrix[Double]) = {
    sigmoid(theta*x)
  }

  def prepareClassificationSample(dataSize: Int): DenseMatrix[Double] = {
    val rg = new scala.util.Random
    DenseMatrix.tabulate[Double](dataSize, 3)((i, j) =>
      j match {
        case 0 => dataSize.asInstanceOf[Double] / 2 - i
        case 1 => (dataSize.asInstanceOf[Double] / 2 - i) * rg.nextDouble()
        case 2 => if (dataSize.asInstanceOf[Double] / 2 - i < 0) 0 else 1
      })
  }
}
