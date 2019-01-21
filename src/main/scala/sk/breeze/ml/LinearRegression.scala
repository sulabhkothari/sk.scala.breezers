package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, inv, linspace, sum}
import breeze.plot.{Figure, plot}
import co.theasi.plotly.{Plot, ThreeDPlot, draw}
import sk.breeze.ml.Util.Line

object LinearRegression {
  def main(args: Array[String]): Unit = {
    compareNormalEquationsAndGradientDescent
  }

  def compareNormalEquationsAndGradientDescent = {
    val rg = new scala.util.Random
    val denseMat = DenseMatrix.tabulate[Double](200, 2)((i, j) => if (j == 0) i + 1 else 2 + i + 30 * rg.nextDouble())

    val X = denseMat(*, 0).underlying
    val Y = denseMat(*, 1).underlying

    val theta = normalEquations(denseMat)
    val line = Line(theta(1), theta(0))
    println(s"Normal Equation solution: (${line.c},${line.slope})")

    val theta2 = Util.gradientDescent(denseMat)

    val line2 = Line(theta2(1), theta2(0))
    val f = Figure()
    val p = f.subplot(0)
    p += plot(X, Y, '.')
    println(s"Gradient Descent solution: (${line2.c},${line2.slope})")
    val ls = linspace(0, X(X.length - 1))
    p += plot(ls, ls :* line.slope :+ line.c)
    p += plot(ls, ls :* line2.slope :+ line2.c, '+')

  }

  def showRegressionWithNormalEquations = {
    val rg = new scala.util.Random
    val denseMat = DenseMatrix.tabulate[Double](60, 2)((i, j) => if (j == 0) i + 1 else 2 + i + 30 * rg.nextDouble())

    val X = denseMat(*, 0).underlying
    val Y = denseMat(*, 1).underlying

    val theta = normalEquations(denseMat)
    val line = Line(theta(1), theta(0))
    val f = Figure()
    val p = f.subplot(0)
    p += plot(X, Y, '.')
    println(line.slope)
    println(line.c)
    val ls = linspace(0, X(X.length - 1))
    println(ls)
    p += plot(ls, ls :* line.slope :+ line.c)
  }

  def showSampleRegressionInXY = {
    val rg = new scala.util.Random
    val denseMat = DenseMatrix.tabulate[Double](60, 2)((i, j) => if (j == 0) i + 1 else 2 + i + 30 * rg.nextDouble())

    val X = denseMat(*, 0).underlying
    val Y = denseMat(*, 1).underlying

    val line = Util.gradientDescentInXY(X, Y)
    val f = Figure()
    val p = f.subplot(0)
    p += plot(X, Y, '.')
    println(line.slope)
    println(line.c)
    val ls = linspace(0, X(X.length - 1))
    println(ls)
    p += plot(ls, ls :* line.slope :+ line.c)
  }

  def showSampleRegressionInXYZ = {
    val rg = new scala.util.Random
    //    val denseMat = DenseMatrix.tabulate[Double](100, 3)((i, j) => j match {
    //      case 0 => (i + 1)*0.1
    //      case 1 => (i + 1)*0.1 + 11.11 * rg.nextDouble()
    //      case 2 => 2 + i + 30 * rg.nextDouble()
    //    })
    //
    //    val X = denseMat(*, 0).underlying
    //    val Y = denseMat(*, 1).underlying
    //    val Z = denseMat(*, 2).underlying
    val xs: Vector[Double] = (-8.0 to 8.0 by 0.1).toVector
    val ys: Vector[Double] = (-8.0 to 8.0 by 0.1).toVector
    val zs = xs zip ys map {
      case (x, y) => x * rg.nextDouble() * 2000 - y * rg.nextDouble() * 1717
    }
    val plane = Util.gradientDescentInXYZ(xs, ys, zs)

    println(plane.a)
    println(plane.b)
    println(plane.c)


    def getScatterPoint(x: Double, y: Double) = x * rg.nextDouble() * 2000 - y * rg.nextDouble() * 1717

    def getRegressionPlane(x: Double, y: Double): Double = plane.a * x + plane.b * y + plane.c

    val zsScatter = xs.map { x => ys.map { y => getScatterPoint(x, y) } }
    val zsPlane = xs.map { x => ys.map { y => getRegressionPlane(x, y) } }

    var p = ThreeDPlot().withSurface(xs, ys, zsPlane)
    draw(p, "regressive-plane")
    val psc = ThreeDPlot().withSurface(xs, ys, zsScatter)
    draw(p, "scatter-plane")


  }

  def normalEquations(data: DenseMatrix[Double]) = {
    val featuresWithoutOnes = data(*, 0 to data.cols - 2).underlying
    val features = Util.prependOnesColumn(featuresWithoutOnes)
    val y = data(*, data.cols - 1).underlying
    val theta = inv(features.t * features) * features.t * y
    theta
  }

}
