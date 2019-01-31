package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, sigmoid}
import breeze.optimize.{DiffFunction, StochasticGradientDescent}
import sk.breeze.ml.LogisticRegression.{cost, gradient}
import sk.breeze.ml.Util.{TrainingData, isConverged, prependOnesColumn}

import scala.collection.mutable
import scala.collection.mutable.ArraySeq

object NeuralNetwork {

  def main(args: Array[String]): Unit = {
//    val layerWiseNeuronCount = Seq(2, 3, 1,2)
//    val weights = DenseVector[Double](Array(1, 1, 1, 1, 1, 0.9, 0.9, 9.9, 0, 8, 0, 8, 0,7,6,7,6))
//    var w = weights.data
//    var init = 0
//    val wei = tuplize(layerWiseNeuronCount)
//      .map {
//        case (row, col) =>
//          println(init)
//          println(init + (row + 1) * col - 1)
//          w = weights.slice(init, init + (row + 1) * col).data
//          println("N")
//          w.foreach(print)
//          println("E")
//          init += (row+1) * col
//          DenseMatrix.create[Double](row + 1, col, w)
//      }
//    wei.foreach(println)
//    scala.io.StdIn.readLine()
    val (pos, neg) = Util.prepareTrainingDataNoBiasAlt(10000)
    val td = TrainingData(pos, neg)
    //val sample = DenseMatrix.create(5,1,Array(1.0,9.0,8.0,-.99,-1.01))
    //sample.data.foreach(print)
    //println(td.trainingData)
    //scala.io.StdIn.readLine()
    val layerConfig = Seq(2, 3, 5, 6, 7, 8, 6, 5, 6, 7, 8, 9, 1)
    val vectorSize = tuplize(layerConfig).map(t => (t._1 + 1) * t._2).sum

    println("V+++++" + vectorSize)

    val stochasticGradientDescent = StochasticGradientDescent[DenseVector[Double]](1, 1000)
    val state = stochasticGradientDescent.minimizeAndReturnState(f(td, layerConfig), DenseVector.rand[Double](vectorSize))
    println(state.x)

    val convergenceStatus = stochasticGradientDescent.convergenceCheck(state, state.convergenceInfo)
    println(convergenceStatus.get.reason)

    val trainingData = Util.prependOnesColumn(td.trainingData)
    Util.plotXY(trainingData)

    scala.io.StdIn.readLine()

    val nn = NeuralNetwork(layerConfig)

    val input = Util.prepareClassificationDataNoBias(10000)
    val output = nn.classify(input)
    println(output)
    val graphableInput = Util.prependOnesColumn(input)
    Util.plotXY(DenseMatrix.create(graphableInput.rows, graphableInput.cols + 1, graphableInput.data ++ output.data))
  }

  def gradientDescentNN = {
    val layerConfig = Seq(2, 3, 5, 6, 7, 8, 6, 5, 6, 7, 8, 9, 1)

    val nn = NeuralNetwork(layerConfig)
    val (pos, neg) = Util.prepareTrainingDataNoBias(100)
    println("DG")
    val td = TrainingData(pos, neg)
    while (!nn.backPropagation(td.x, DenseMatrix.create(td.y.length, 1, td.y.data))) {
      //println(nn.weights(0))
    }
    val trainingData = Util.prependOnesColumn(td.trainingData)
    Util.plotXY(trainingData)
    scala.io.StdIn.readLine()
    val input = Util.prepareClassificationDataNoBias(10000)
    val output = nn.classify(input)
    println(output)
    val graphableInput = Util.prependOnesColumn(input)
    Util.plotXY(DenseMatrix.create(graphableInput.rows, graphableInput.cols + 1, graphableInput.data ++ output.data))
  }

  def tuplize(seq: Seq[Int]) = {
    var mutableSeq = Seq.empty[(Int, Int)]
    seq.foldLeft(Seq.empty[Int])((seq, x) => {
      if (seq.nonEmpty) mutableSeq ++= ArraySeq((seq.head, x))
      Seq(x)
    })
    mutableSeq
  }

  private def getPackedWeights(weights: Seq[DenseMatrix[Double]]) = {
    DenseVector(weights.flatMap(_.data).toArray)
  }

  val f =
    (td: TrainingData, layerWiseNeuronCount: Seq[Int])
    => new DiffFunction[DenseVector[Double]] {
      def calculate(theta: DenseVector[Double]) = {
        val nn = NeuralNetwork(layerWiseNeuronCount, theta)
        val y = nn.classify(td.x)
        val expectedy = td.y.asDenseMatrix.t
        //println(expectedy.t)
        //expectedy..data.foreach(n=>if(n<0) print(s"($n)") else print(s"[$n]"))
        //        println(expectedy - y)
        //        println((expectedy - y).data.map(math.abs).sum)
        //        println(sum(abs(expectedy - y)))
        //        println(expectedy(9000,0))

        //        scala.io.StdIn.readLine()
        val cost = sum(abs(expectedy - y))
        val gradient = getPackedWeights(nn.gradient(td.x, expectedy))
        (cost, gradient)
      }
    }

  def apply(layerWiseNeuronCount: Seq[Int]): NeuralNetwork = {
    println(tuplize(layerWiseNeuronCount))
    val weightsArr = for ((row, col) <- tuplize(layerWiseNeuronCount))
      yield DenseMatrix.rand[Double](row + 1, col)

    new NeuralNetwork(weightsArr, 0.000019, 0.0005)
  }

  private def apply(layerWiseNeuronCount: Seq[Int], weights: DenseVector[Double]): NeuralNetwork = {
    var w = weights.data
    var init = 0
    val weightsArr = tuplize(layerWiseNeuronCount)
      .map {
        case (row, col) =>
          w = weights.data.slice(init, init + (row + 1) * col - 1)
          init += (row+1) * col
          DenseMatrix.create[Double](row + 1, col, w)
      }

    new NeuralNetwork(weightsArr, 0.000019, 0.0005)
  }
}

class NeuralNetwork private(var weights: Seq[DenseMatrix[Double]], learningRate: Double, convergencePrecision: Double) {

  def classify(input: DenseMatrix[Double]) = {
    weights.foldLeft(input)((x, w) => {
      sigmoid(Util.prependOnesColumn(x) * w)
    })
  }

  private def forwardPropagation(input: DenseMatrix[Double]) = {
    var activationQueue: mutable.Queue[DenseMatrix[Double]] = new mutable.Queue[DenseMatrix[Double]]()
    val actual = weights.foldLeft(input)((x, w) => {
      val a = sigmoid(Util.prependOnesColumn(x) * w)
      activationQueue.enqueue(a)
      a
    })
    activationQueue.toArray
  }

  private def calculateDeltas(activations: Array[DenseMatrix[Double]], y: DenseMatrix[Double]) = {
    val newActivations = activations.take(activations.length - 1)
    val newWeights = weights.drop(1)
    val activationf = activations(activations.length - 1)
    val deltaf = (activationf - y) :* (activationf :* (DenseMatrix.ones[Double](activationf.rows, activationf.cols) - activationf))
    val queueOfDelta = mutable.Queue(deltaf)
    (newActivations, newWeights).zipped.foldRight(deltaf)((wa, d) => {
      val mat = d * wa._2.t
      val nextDelta = mat(*, 1 to mat.cols - 1).underlying :* (wa._1 :* (DenseMatrix.ones[Double](wa._1.rows, wa._1.cols) - wa._1))
      queueOfDelta.enqueue(nextDelta)
      nextDelta
    })
    queueOfDelta.toArray.reverse
  }

  private def getSameDimMat(mat: DenseMatrix[Double], value: Double) = {
    DenseMatrix.tabulate[Double](mat.rows, mat.cols)((_, _) => value)
  }

  def backPropagation(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]): Boolean = {
    val activations = forwardPropagation(input)
    val deltas = calculateDeltas(activations, expectedOutput)
    val gradient = (Seq(input) ++ activations, deltas).zipped.map((activation, delta) => {
      val del = delta.t * activation
      val biasColumn = DenseMatrix.ones[Double](1, delta.rows) * delta
      val gradDelta = DenseMatrix.create(del.rows, del.cols + 1, biasColumn.data ++ del.data).t
      getSameDimMat(gradDelta, (1.0 / input.rows)) :* gradDelta
    })
    //println(weights(3))
    println(gradient(3))

    //println("=============")
    val newWeights = (weights, gradient).zipped.map((w, g) => w :- (getSameDimMat(g, learningRate) :* g))
    //println(newWeights(0))
    //scala.io.StdIn.readLine()
    val isConverged = hasConverged(weights, newWeights)
    weights = newWeights
    isConverged
  }

  def gradient(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]): Seq[DenseMatrix[Double]] = {
    val activations = forwardPropagation(input)
    val deltas = calculateDeltas(activations, expectedOutput)
    (Seq(input) ++ activations, deltas).zipped.map((activation, delta) => {
      val del = delta.t * activation
      val biasColumn = DenseMatrix.ones[Double](1, delta.rows) * delta
      val gradDelta = DenseMatrix.create(del.rows, del.cols + 1, biasColumn.data ++ del.data).t
      getSameDimMat(gradDelta, (1.0 / input.rows)) :* gradDelta
    })
  }

  private def hasConverged(w1: Seq[DenseMatrix[Double]], w2: Seq[DenseMatrix[Double]]) = {
    println("_____" + (w1, w2).zipped.map((x, y) => (x - y).data.map(math.abs(_)).sum).sum)
    (w1, w2).zipped.map((x, y) => (x - y).data.map(math.abs(_)).sum).sum <= convergencePrecision
  }

  private def gradientSum(g: Seq[DenseMatrix[Double]]) = {
    g.map(_.data.map(math.abs).sum).sum
  }

  //private val layerWiseWeightArray:Seq[DenseMatrix[Double]]
}