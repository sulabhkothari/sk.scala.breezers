package sk.breeze.ml

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import org.scalatest.FunSuite

class NeuralNetworkTests extends FunSuite {
  test("NN classify") {
    val input = DenseMatrix.create(1, 2, Array(2.0, 3.5))
    val expectedOutput = DenseMatrix.create(1, 1, Array(-9.0))
    val weights = Array(
      DenseMatrix.create(3, 2, Array(1.5, 1.0, 1.9, 5.5, 3.0, 2.9)),
      DenseMatrix.create(3, 1, Array(2.0, 2.0, 7.0))
    )

    val l1Output = sigmoid(Util.prependOnesColumn(input) * weights(0))
    val l2Output = sigmoid(Util.prependOnesColumn(l1Output) * weights(1))

    val nn = NeuralNetwork(weights)

    val g = nn.gradient(input, expectedOutput)
    val y = nn.classify(input)
    assert(y == l2Output)
  }

  test("NN forward propagation") {
    val input = DenseMatrix.create(1, 2, Array(2.0, 3.5))
    val expectedOutput = DenseMatrix.create(1, 1, Array(-9.0))
    val weights = Array(
      DenseMatrix.create(3, 2, Array(1.5, 1.0, 1.9, 5.5, 3.0, 2.9)),
      DenseMatrix.create(3, 1, Array(2.0, 2.0, 7.0))
    )

    val l1Output = sigmoid(Util.prependOnesColumn(input) * weights(0))
    val l2Output = sigmoid(Util.prependOnesColumn(l1Output) * weights(1))
    println(weights(0))
    val l1NoSigmoid = Util.prependOnesColumn(input) * weights(0)
    println(sigmoid(l1NoSigmoid(0, 0)))
    val nn = NeuralNetwork(weights)
    println(l1Output)

    val y = nn.forwardPropagation(input)
    assert(y(0) == l1Output)
    assert(y(1) == l2Output)
  }

  test("NN Deltas") {
    val input = DenseMatrix.create(1, 1, Array(2.0))
    val expectedOutput = DenseMatrix.create(1, 1, Array(-10.0))
    val weights = Array(
      DenseMatrix.create(2, 2, Array(1.0, 2.0, 3.0, -5.0)),
      DenseMatrix.create(3, 1, Array(2.0, 2.0, 7.0))
    )

    val l1Output = sigmoid(Util.prependOnesColumn(input) * weights(0))
    val l2Output = sigmoid(Util.prependOnesColumn(l1Output) * weights(1))
    println(weights(0))
    val l1NoSigmoid = Util.prependOnesColumn(input) * weights(0)
    println(l1NoSigmoid)
    println(sigmoid(l1NoSigmoid(0, 0)))
    val nn = NeuralNetwork(weights)
    println(l1Output)
    println(NeuralNetwork.getSameDimMat(weights(0), 88.99))
    val activations = nn.forwardPropagation(input)
    assert(activations(0) == l1Output)
    assert(activations(1) == l2Output)

    val deltas = nn.calculateDeltas(activations, expectedOutput)
    println("===========")
    println(deltas(1))
    println((activations(1) - expectedOutput) :* (activations(1) :* (NeuralNetwork.getSameDimMat(activations(1), 1.0) - activations(1))))
  }

  test("split resultant") {
    val y = DenseVector(Array(1.0,1.0,0,1.0,0,0,0,1))
    println(NeuralNetwork.splitOutputMat(y))
  }
}
