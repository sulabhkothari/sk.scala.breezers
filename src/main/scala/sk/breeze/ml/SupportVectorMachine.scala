package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import breeze.optimize.{DiffFunction, StochasticGradientDescent}
import sk.breeze.ml.LogisticRegression.{cost, f, gradient}
import sk.breeze.ml.Util.TrainingData

object SupportVectorMachine {
  def main(args: Array[String]): Unit = {
    val (pos, neg) = Util.prepareTrainingData(1000,1.0,-1.0)
    val td = TrainingData(pos, neg)

    Util.plotXY(td.trainingData)
    scala.io.StdIn.readLine()

    val gd = StochasticGradientDescent[DenseVector[Double]]()
    val state1 = gd.minimizeAndReturnState(f(td), td.randomTheta)
    val cs1 = gd.convergenceCheck(state1, state1.convergenceInfo)
    println(":::::::::::::::::>" + cs1.get.reason)
    println("=================>" + state1.x)

    val gdL = StochasticGradientDescent[DenseVector[Double]]()
    val stateL = gd.minimizeAndReturnState(LogisticRegression.f(td), td.randomTheta)
    val csL = gd.convergenceCheck(state1, state1.convergenceInfo)
    println(":::::::::::::::::>" + csL.get.reason)
    println("=================>" + stateL.x)

    val newData = Util.prepareClassificationData(5000)
    val y = classify(state1.x, newData, 10)
    val plotData = DenseMatrix.create(newData.rows, newData.cols + 1, newData.data ++ y.data)
    //println(plotData)
    Util.plotXY(plotData)

    scala.io.StdIn.readLine()

    val yL = sigmoid(newData * stateL.x)
    val plotDataL = DenseMatrix.create(newData.rows, newData.cols + 1, newData.data ++ yL.data)
    //println(plotData)
    Util.plotXY(plotDataL)
  }

  def gradient(td: TrainingData, theta: DenseVector[Double]): DenseVector[Double] = {
    (DenseVector(((td.x * theta).data, td.y.data).zipped.map((p, q) =>
      if ((p >= 1 && q == -1) || (p < 1 && q == 1)) 1.0 else 0.0)).t * td.x).t
  }

  def cost(td: TrainingData, theta: DenseVector[Double]): Double = {
    (DenseVector(((td.x * theta).data, td.y.data).zipped.map((p, q) =>
      if ((p >= 1 && q == -1) || (p < 1 && q == 1)) 1 - p * q else 0.0))).data.sum
  }

  val f = (td: TrainingData) => new DiffFunction[DenseVector[Double]] {
    def calculate(theta: DenseVector[Double]) = {
      (cost(td, theta), gradient(td, theta))
    }
  }

  def classify(theta: DenseVector[Double], x: DenseMatrix[Double], b: Double): DenseVector[Double] = {
    (x * theta :- b :> 1.0).map(if(_) 1.0 else -1.0)
  }
}
