package sk.breeze.ml

import breeze.linalg.DenseVector
import breeze.plot.{Figure, plot}

object ProbabilityDistribution {
  val n = 50
  val c = 20
  val b = 3

  def main(args: Array[String]): Unit = {
    val x = (0 until 2 * n - c map (i => i + c/1.0)).toVector
    val y = (0 until 2 * n - c map (i => getProbability(i + c))).toVector
    x zip y foreach{case (u,v) => println(s"(${u},${v})")}
    val f = Figure()
    val p = f.subplot(0)

    p+=plot(x,y, '.')
  }

  def getProbability(numOfTrials: Int): Double = {
    var probability = 1.0
    for (s <- 0 until numOfTrials) {
      val x:Double = s
      val d:Double = if (x < c)
        (n + c - x) / (n * b + c - x)
      else
        ((b - 1)* n - x + c) / (n * b + c - x)
      probability *= d
      println(d)
    }
    probability
  }
}
