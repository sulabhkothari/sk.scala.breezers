package sk.breeze.ml

import breeze.linalg.{DenseVector, norm, sum}
import breeze.optimize.{DiffFunction, LBFGS}
import sk.breeze.ml.LogisticRegression.{TrainingData, cost, f, gradient}

object ConvexOptimization {
  def main(args: Array[String]): Unit = {
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 10000, m = 3)
    val minima = lbfgs.minimize(f2, Util.randomVector(1))
    println(minima)
    println(f2(minima))
    val state = lbfgs.minimizeAndReturnState(f2, Util.randomVector(1))
    val convergence = lbfgs.convergenceCheck(state, state.convergenceInfo)
    println(convergence.get.reason)

    val minima1 = lbfgs.minimize(f, Util.randomVector(1))
    println(minima1)
    println(f(minima1))
    val state1 = lbfgs.minimizeAndReturnState(f, Util.randomVector(1))
    val convergence1 = lbfgs.convergenceCheck(state1, state1.convergenceInfo)
    println(convergence1.get.reason)

  }


  val f = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      (sum(((x - 45d) :^ 2d) - x :* 34.89d), (x :* 2d) - 90d - 34.89d)
    }
  }

  val f2 = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      (norm((x - 3d) :^ 2d, 1d), (x * 2d) - 6d);
    }
  }
}
