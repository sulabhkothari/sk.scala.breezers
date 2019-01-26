package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.plot.{Figure, plot}

object Util {

  def randomVector(size:Int) = {
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseVector.rand(size, normal01)
  }

  case class Line(slope: Double, c: Double)

  case class Plane(a: Double, b: Double, c: Double)

  def prependOnesColumn(original: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones = DenseVector.ones[Double](original.rows)
    val dataWithOnes = ones.data ++ original.data
    DenseMatrix.create(original.rows, original.cols + 1, dataWithOnes)
  }

  def gradientDescent(data: DenseMatrix[Double]): DenseVector[Double] = {
    var theta = DenseVector.ones[Double](data.cols)
    var thetaTemp = DenseVector.zeros[Double](data.cols)
    val y = data(*, data.cols - 1).underlying
    val features = prependOnesColumn(data(*, 0 to data.cols - 2).underlying)
    val count: Double = data.rows
    val alpha: Double = 0.05 / count
    val precision = 0.000001 // count

    while (!isConverged(theta, thetaTemp, precision)) {
      theta = thetaTemp
      val diff = (features * theta - y)
      val result = diff.t * features
      thetaTemp = theta - (result.t :* alpha :/ (2 * count))
    }
    thetaTemp
  }

  def isConverged(theta: DenseVector[Double], thetaTemp: DenseVector[Double], precision: Double) = {
    (theta - thetaTemp).data.map(_.abs).forall(_ < precision)
  }

  def gradientDescentInXYZ(X: Seq[Double], Y: Seq[Double], Z: Seq[Double]): Plane = {
    var a: Double = 1.0
    var b: Double = 1.0
    var c: Double = 1.0

    var aTemp: Double = 0.0
    var bTemp: Double = 0.0
    var cTemp: Double = 0.0

    val count = X.length
    val alpha: Double = 0.00005 / count

    val precision = 0.0001 // count

    while ((a - aTemp).abs > precision || (b - bTemp).abs > precision
      || (c - cTemp).abs > precision) {
      a = aTemp
      b = bTemp
      c = cTemp

      val sum = (X zip Y zip Z map {
        case ((x, y), z) => ((a * x + b * y + c - z) * x, (a * x + b * y + c - z) * y, (a * x + b * y + c - z))
      }).foldLeft((0.0, 0.0, 0.0))((accum, item) =>
        (accum._1 + item._1, accum._2 + item._2, accum._3 + item._3)
      )

      aTemp = a - alpha * sum._1 / (2 * count)
      bTemp = b - alpha * sum._2 / (2 * count)
      cTemp = c - alpha * sum._3 / (2 * count)

      println(s"($a,$b,$c)")

      println(s"($aTemp,$bTemp,$cTemp)")
      scala.io.StdIn.readLine()
    }

    Plane(a, b, c)
  }

  def gradientDescentInXY(X: DenseVector[Double], Y: DenseVector[Double]): Line = {
    var m: Double = 1.0
    var c: Double = 1.0
    var mTemp: Double = 0.0
    var cTemp: Double = 0.0

    val count = X.length
    val alpha: Double = 0.005 / count

    val precision = 0.0001 / count

    while ((m - mTemp).abs > precision || (c - cTemp).abs > precision) {
      m = mTemp
      c = cTemp

      val sumM = X.toArray zip Y.toArray map {
        case (x, y) => (m * x + c - y) * x
      } sum

      val sumC = X.toArray zip Y.toArray map {
        case (x, y) => (m * x + c - y)
      } sum

      mTemp = m - alpha * sumM / (2 * count)
      cTemp = c - alpha * sumC / (2 * count)

      //println(s"($sumM,$sumC)")
      //println(s"($mTemp,$cTemp)")
      //scala.io.StdIn.readLine()
    }

    Line(m, c)
  }

  def plotXY(xy: DenseMatrix[Double]): Unit = {
    val xyPositives = for {
      i <- 0 to xy.rows - 1 if (xy(i, *).underlying(3) > 0.5)
    }
      yield xy(i, *)

    val dm1 = DenseMatrix.tabulate[Double](xyPositives.length, 2)((i, j) => xyPositives(i).underlying(j+1))


    val xyNegatives = for {
      i <- 0 to xy.rows - 1 if (xy(i, *).underlying(3) < 0.5)
    }
      yield xy(i, *)
    val dm2 = DenseMatrix.tabulate[Double](xyNegatives.length, 2)((i, j) => xyNegatives(i).underlying(j+1))

    val f = Figure()
    val p = f.subplot(0)
    p += plot(dm1(*, 0).underlying, dm1(*, 1).underlying, '.')
    p += plot(dm2(*, 0).underlying, dm2(*, 1).underlying, '+')

  }

  def randomVector(size:Int) = {
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseVector.rand(size, normal01)
  }
}
