package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.{abs, sigmoid}
import breeze.optimize.StochasticGradientDescent
import sk.breeze.ml.Util.{TrainingData, isConverged, prependOnesColumn}

import scala.collection.mutable
import scala.collection.mutable.ArraySeq

object NeuralNetwork {
  def main(args: Array[String]): Unit = {
    val nn = NeuralNetwork(2, Seq(4, 5, 3), 1)
    val (pos, neg) = Util.prepareTrainingDataNoBias(1000)
    println("DG")
    val td = TrainingData(pos, neg)
    while(!nn.backPropagation(td.x, DenseMatrix.create(td.y.length, 1, td.y.data))){
      //println(nn.weights(0))
    }
    val trainingData = Util.prependOnesColumn(td.trainingData)
    //Util.plotXY(trainingData)
    //scala.io.StdIn.readLine()
    val input = Util.prepareClassificationDataNoBias(10000)
    val output = nn.classify(input)
    val graphableInput = Util.prependOnesColumn(input)
    //Util.plotXY(DenseMatrix.create(graphableInput.rows,graphableInput.cols+1,graphableInput.data ++ output.data))
  }

  def tuplize(seq: Seq[Int]) = {
    var mutableSeq = Seq.empty[(Int, Int)]
    seq.foldLeft(Seq.empty[Int])((seq, x) => {
      if (seq.nonEmpty) mutableSeq ++= ArraySeq((seq.head, x))
      Seq(x)
    })
    mutableSeq
  }

  def apply(inputCount: Int, layerWiseNeuronCount: Seq[Int], outputCount: Int): NeuralNetwork = {
    println(tuplize(Seq(inputCount) ++ layerWiseNeuronCount ++ Seq(outputCount)))
    val weightsArr = for ((row, col) <- tuplize(Seq(inputCount) ++ layerWiseNeuronCount ++ Seq(outputCount)))
      yield DenseMatrix.rand[Double](row + 1, col)

    new NeuralNetwork(weightsArr,1,12.5)
  }
}

class NeuralNetwork private(var weights: Seq[DenseMatrix[Double]], learningRate:Double, convergencePrecision:Double) {

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

  private def getSameDimMat(mat:DenseMatrix[Double], value:Double) = {
    DenseMatrix.tabulate[Double](mat.rows, mat.cols)((_,_) => value)
  }

  def backPropagation(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]):Boolean = {
    val activations = forwardPropagation(input)
    val deltas = calculateDeltas(activations, expectedOutput)
    val gradient = (Seq(input) ++ activations, deltas).zipped.map((activation, delta) => {
      val del = delta.t * activation
      val biasColumn = DenseMatrix.ones[Double](1,delta.rows) * delta
      val gradDelta = DenseMatrix.create(del.rows,del.cols+1, biasColumn.data ++ del.data).t
      getSameDimMat(gradDelta, (1/input.rows)) :* gradDelta
    })
//println(weights(3))
    //println(gradient(3))

    //println("=============")
    val newWeights = (weights, gradient).zipped.map((w,g) => w :- (getSameDimMat(g,learningRate) :* g))
//println(newWeights(0))
    //scala.io.StdIn.readLine()
    val isConverged = hasConverged(weights,newWeights)
    weights = newWeights
    isConverged
  }

  private def hasConverged(w1:Seq[DenseMatrix[Double]], w2:Seq[DenseMatrix[Double]]) = {
    println("_____"+(w1,w2).zipped.map((x,y)=> (x - y).data.map(math.abs(_)).sum).sum)
    (w1,w2).zipped.map((x,y)=> (x - y).data.map(math.abs(_)).sum).sum <= convergencePrecision
  }

  private def gradientSum(g:Seq[DenseMatrix[Double]]) = {
    g.map(_.data.map(math.abs).sum).sum
  }

  //private val layerWiseWeightArray:Seq[DenseMatrix[Double]]
}