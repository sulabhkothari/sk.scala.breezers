package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, sigmoid}
import breeze.optimize.{DiffFunction, LBFGS}
import breeze.plot.plot
import sk.breeze.ml.Util.{isConverged, prependOnesColumn}

object LogisticRegression {

  case class TrainingData(trainingDataPos: DenseMatrix[Double], trainingDataNeg: DenseMatrix[Double]) {
    val xPos = trainingDataPos(*, 0 to trainingDataPos.cols - 2).underlying
    val yPos = trainingDataPos(*, trainingDataPos.cols - 1).underlying
    val xNeg = trainingDataNeg(*, 0 to trainingDataNeg.cols - 2).underlying
    val yNeg = trainingDataNeg(*, trainingDataNeg.cols - 1).underlying

    lazy val posOnes = DenseVector.ones[Double](yPos.length)
    lazy val negOnes = DenseVector.ones[Double](yNeg.length)

    lazy val trainingData = DenseMatrix.vertcat(trainingDataPos, trainingDataNeg)

    def x = trainingData(*, 0 to trainingData.cols - 2).underlying

    def y = trainingData(*, trainingData.cols - 1).underlying

    lazy val randomTheta = {
      val normal01 = breeze.stats.distributions.Gaussian(0, 1)
      DenseVector.rand(trainingDataPos.cols - 1, normal01)
    }
  }

  def main(args: Array[String]): Unit = {
    val (trainingDataPos, trainingDataNeg) = prepareTrainingData(10000)
    val trainingData = TrainingData(trainingDataNeg, trainingDataPos)
    Util.plotXY(trainingData.trainingData)
    scala.io.StdIn.readLine()
    //Util.plotXY(trainingData)
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10000, m = 4)
    val minTheta = lbfgs.minimize(f(trainingData), trainingData.randomTheta)
    val minThetaGd = gradientDescent(trainingData)
    println(trainingData.randomTheta)
    println(minTheta)
    println(minThetaGd)
    println(hTheta(minThetaGd, DenseVector(1, 6000, 6000)))
    println(hTheta(minThetaGd, DenseVector(1, -6000, -6000)))
    val newData = prepareClassificationData(5)
    val y = sigmoid(newData * minThetaGd)
    val plotData = DenseMatrix.create(newData.rows, newData.cols + 1, newData.data ++ y.data)
    println(plotData)
    Util.plotXY(plotData)

  }

  //−1/m[∑(i=1m)y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i)))]
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
      println("==>" + thetaTemp)
      thetaTemp = theta - (diff :* alpha :/ (2 * count))
    }
    thetaTemp
  }

  def prepareTrainingData(dataSize: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val rg = new scala.util.Random
    (
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i
          case 1 => i * rg.nextDouble
          case 2 => 1
        })),
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i - dataSize.asInstanceOf[Double] / 2
          case 1 => (i - dataSize.asInstanceOf[Double] / 2) * rg.nextDouble
          case 2 => if (i - dataSize.asInstanceOf[Double] / 2 < 0) 0 else 1
        })))
  }

  def prepareClassificationData(dataSize: Double): DenseMatrix[Double] = {
    val points = for{
      x <- -dataSize to dataSize by 0.1
      y <- -dataSize to dataSize by 0.1
    }
      yield (x,y)

    prependOnesColumn(DenseMatrix(points:_*))
  }
}
