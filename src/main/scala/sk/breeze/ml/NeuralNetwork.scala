package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import sk.breeze.ml.Util.TrainingData

import scala.collection.mutable
import scala.collection.mutable.ArraySeq

object NeuralNetwork {
  def main(args: Array[String]): Unit = {
    val nn = NeuralNetwork(2, Seq(4, 5, 3), 1)
    val (pos, neg) = Util.prepareTrainingDataNoBias(100)
    val td = TrainingData(pos, neg)
    println(nn.classify(td.x))
    nn.backPropagation(td.x, DenseMatrix.create(td.y.length, 1, td.y.data))
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

    new NeuralNetwork(weightsArr)
  }
}

class NeuralNetwork private(weights: Seq[DenseMatrix[Double]]) {

  def classify(input: DenseMatrix[Double]) = {
    weights.foldLeft(input)((x, w) => {
      println(s"${x.rows}x${x.cols + 1}......${w.rows},${w.cols}")
      sigmoid(Util.prependOnesColumn(x) * w)
    })
  }

  private def forwardPropagation(input: DenseMatrix[Double]) = {
    var activationQueue: mutable.Queue[DenseMatrix[Double]] = new mutable.Queue[DenseMatrix[Double]]()
    val actual = weights.foldLeft(input)((x, w) => {
      println(s"${x.rows}x${x.cols + 1}......${w.rows},${w.cols}")
      val a = sigmoid(Util.prependOnesColumn(x) * w)
      activationQueue.enqueue(a)
      a
    })
    activationQueue.toArray
  }

  private def calculateDeltas(activations: Array[DenseMatrix[Double]], y: DenseMatrix[Double]) = {
    val newActivations = activations.take(activations.length - 1)
    val newWeights = weights.drop(1)
    val deltaf = activations(activations.length - 1) - y
    val queueOfDelta = mutable.Queue(deltaf)
    (newActivations, newWeights).zipped.foldRight(deltaf)((wa, d) => {
      println(s"ACT: ${wa._1.rows}x${wa._1.cols}")
      println(s"WEI: ${wa._2.rows}x${wa._2.cols}")
      println(s"DEL: ${d.rows}x${d.cols}")
      val mat = d * wa._2.t
      //println(mat(*, 1 to mat.cols-1))
      val nextDelta = mat(*, 1 to mat.cols-1).underlying :* (wa._1 :* (DenseMatrix.ones[Double](wa._1.rows, wa._1.cols) - wa._1))
      queueOfDelta.enqueue(nextDelta)
      nextDelta
      //deltaf
    })
    queueOfDelta.toArray.reverse.foreach(x=> println(s"Dim: ${x.rows}x${x.cols}"))
    queueOfDelta.toArray.reverse
  }

  def backPropagation(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]) = {
    val activations = forwardPropagation(input)
    //activations.foreach(x => println(s"ACTIVATIONS: ${x.rows}x${x.cols}"))
    val deltas = calculateDeltas(activations, expectedOutput)
    println(deltas(2).t * activations(1))

    println(s"\n\n\n D - ${deltas(1).rows}x${deltas(1).cols}\nA - ${activations(0).rows}x${activations(0).cols}")
    println(s"${weights(0).rows}x${weights(0).cols}")
    println(weights.length)
  }

  //private val layerWiseWeightArray:Seq[DenseMatrix[Double]]
}