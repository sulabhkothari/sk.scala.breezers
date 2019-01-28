package sk.breeze.ml

import breeze.linalg.{DenseMatrix, DenseVector}
import sk.breeze.ml.Util.TrainingData

import scala.collection.mutable
import scala.collection.mutable.ArraySeq

object NeuralNetwork {
  def main(args: Array[String]): Unit = {
    apply(1, Seq(2), 2)
  }

  def tuplize(seq: Seq[Int]) = {
    var mutableSeq = Seq.empty[(Int, Int)]
    seq./:(Seq.empty[Int])((seq, x) => {
      if (seq.nonEmpty) mutableSeq ++= ArraySeq((seq.head, x))
      Seq (x)
    })
    mutableSeq
  }

  def apply(inputCount: Int, layerWiseNeuronCount: Seq[Int], outputCount: Int): NeuralNetwork = {
    println(tuplize(Seq(inputCount) ++ layerWiseNeuronCount ++ Seq(outputCount)))
    val weightsArr = for ((row, col) <- tuplize(Seq(inputCount) ++ layerWiseNeuronCount ++ Seq(outputCount)))
      yield DenseMatrix.rand[Double](row + 1, col)

    new NeuralNetwork(layerWiseNeuronCount)
  }
}

class NeuralNetwork(layerWiseNeuronCount: Seq[Int]) {
  //private val layerWiseWeightArray:Seq[DenseMatrix[Double]]
}