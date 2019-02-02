package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, log, pow, sigmoid}
import breeze.optimize.{DiffFunction, StochasticGradientDescent}
import sk.breeze.ml
import sk.breeze.ml.LogisticRegression.{cost, gradient}
import sk.breeze.ml.Util.{TrainingData, isConverged, prependOnesColumn}

import scala.collection.mutable
import scala.collection.mutable.ArraySeq

object NeuralNetwork {

  def main(args: Array[String]): Unit = {
    //    val layerWiseNeuronCount = Seq(2, 3, 1, 2)
    //    val weights = DenseVector[Double](Array(1, 1, 1, 1, 1, 0.9, 0.9, 9.9, 0, 8, 0, 8, 0, 7, 6, 7, 6))
    //    val wei = UnpackWeights(layerWiseNeuronCount, weights)
    //    wei.foreach(println)
    //    println(getPackedWeights(wei))
    //    scala.io.StdIn.readLine()
    val (pos, neg) = Util.prepareTrainingDataNoBiasAlt(10000)
    val td = TrainingData(pos, neg)
    //println(td.trainingData(9900 to 9980,*))
    //scala.io.StdIn.readLine()
    //val sample = DenseMatrix.create(5,1,Array(1.0,9.0,8.0,-.99,-1.01))
    //sample.data.foreach(print)
    //println(td.trainingData)
    //scala.io.StdIn.readLine()
    //val layerConfig = Seq(2, 7, 9,13,17,29,39,50,33,16,9,19, 17, 15, 1)
    val layerConfig = Seq(2, 9,9,1)

    val vectorSize = tuplize(layerConfig).map(t => (t._1 + 1) * t._2).sum

    println("V+++++" + vectorSize)

    val stochasticGradientDescent = StochasticGradientDescent[DenseVector[Double]](1, 100000)
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
        val expectedy = td.y.asDenseMatrix.t //splitOutputMat(td.y)
        //val cost = sum(-log(nn.classify(td.xPos)(*,0).underlying) + sum(-log(td.negOnes - nn.classify(td.xNeg)(*,0).underlying)))
        val cost = sum(pow(expectedy - nn.classify(td.x), 2.0)) / td.trainingData.rows
        println(cost)
        val gradient = getPackedWeights(nn.gradient(td.x, expectedy))
        (cost, gradient)
      }
    }

  def apply(layerWiseNeuronCount: Seq[Int]): NeuralNetwork = {
    val weightsArr = for ((row, col) <- tuplize(layerWiseNeuronCount))
      yield DenseMatrix.rand[Double](row + 1, col)

    new NeuralNetwork(weightsArr)
  }

  def apply(weights: Array[DenseMatrix[Double]]): NeuralNetwork = {
    new NeuralNetwork(weights)
  }

  def UnpackWeights(layerWiseNeuronCount: Seq[Int], weights: DenseVector[Double]) = {
    var w = weights.data
    var init = 0
    tuplize(layerWiseNeuronCount)
      .map {
        case (row, col) =>
          w = weights.data.slice(init, init + (row + 1) * col)
          init += (row + 1) * col
          DenseMatrix.create[Double](row + 1, col, w)
      }
  }

  private def apply(layerWiseNeuronCount: Seq[Int], weights: DenseVector[Double]): NeuralNetwork = {
    val weightsArr = UnpackWeights(layerWiseNeuronCount, weights)
    new NeuralNetwork(weightsArr)
  }

  def splitOutputMat(y: DenseVector[Double]) = {
    val result = y.asDenseMatrix
    //val y1 = (result + getSameDimMat(result,1.0)) :* getSameDimMat(result, 0.5)
    val invertedY = (result + getSameDimMat(result, -1.0)) :* getSameDimMat(result, -1.0)
    //val y2 = invertedY + getSameDimMat(invertedY,1.0) :* getSameDimMat(invertedY, 0.5)
    DenseMatrix.horzcat(result.t, invertedY.t)
  }

  def getSameDimMat(mat: DenseMatrix[Double], value: Double) = {
    DenseMatrix.tabulate[Double](mat.rows, mat.cols)((_, _) => value)
  }
}

class NeuralNetwork private(var weights: Seq[DenseMatrix[Double]], learningRate: Double = 0.01, convergencePrecision: Double = 0.01) {

  def classify(input: DenseMatrix[Double]) = {
    weights.foldLeft(input)((x, w) => {
      sigmoid(Util.prependOnesColumn(x) * w)
    })
  }

  def forwardPropagation(input: DenseMatrix[Double]) = {
    var activationQueue: mutable.Queue[DenseMatrix[Double]] = new mutable.Queue[DenseMatrix[Double]]()
    val actual = weights.foldLeft(input)((x, w) => {
      val a = sigmoid(Util.prependOnesColumn(x) * w)
      activationQueue.enqueue(a)
      a
    })
    activationQueue.toArray
  }

  def calculateDeltas(activations: Array[DenseMatrix[Double]], y: DenseMatrix[Double]) = {
    val newActivations = activations.take(activations.length - 1)
    val newWeights = weights.drop(1)
    val activationf = activations(activations.length - 1)
    val deltaf = (activationf - y) :* (activationf :* (NeuralNetwork.getSameDimMat(activationf, 1.0) - activationf))
    val queueOfDelta = mutable.Queue(deltaf)
    (newActivations, newWeights).zipped.foldRight(deltaf)((wa, d) => {
      val mat = d * wa._2.t
      val nextDelta = mat(*, 1 to mat.cols - 1).underlying :* (wa._1 :* (NeuralNetwork.getSameDimMat(wa._1, 1.0) - wa._1))
      queueOfDelta.enqueue(nextDelta)
      nextDelta
    })
    queueOfDelta.toArray.reverse
  }

  def backPropagation(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]): Boolean = {
    val activations = forwardPropagation(input)
    val deltas = calculateDeltas(activations, expectedOutput)
    val gradient = (Seq(input) ++ activations, deltas).zipped.map((activation, delta) => {
      val del = delta.t * activation
      val biasColumn = DenseMatrix.ones[Double](1, delta.rows) * delta
      val gradDelta = DenseMatrix.create(del.rows, del.cols + 1, biasColumn.data ++ del.data).t
      NeuralNetwork.getSameDimMat(gradDelta, (1.0 / input.rows)) :* gradDelta
    })
    //println(weights(3))
    println(gradient(3))

    //println("=============")
    val newWeights = (weights, gradient).zipped.map((w, g) => w :- (NeuralNetwork.getSameDimMat(g, learningRate) :* g))
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
      NeuralNetwork.getSameDimMat(gradDelta, (1.0 / input.rows)) :* gradDelta
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