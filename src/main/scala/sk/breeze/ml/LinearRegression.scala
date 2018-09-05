package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector, linspace}
import breeze.plot.{Figure, plot}
import co.theasi.plotly.{Plot, ThreeDPlot, draw}

object LinearRegression {

  case class Line(slope: Double, c: Double)

  case class Plane(a: Double, b: Double, c: Double)

  def main(args: Array[String]): Unit = {
    showSampleRegressionInXYZ
  }

  def showSampleRegressionInXY = {
    val rg = new scala.util.Random
    val denseMat = DenseMatrix.tabulate[Double](60, 2)((i, j) => if (j == 0) i + 1 else 2 + i + 30 * rg.nextDouble())

    val X = denseMat(*, 0).underlying
    val Y = denseMat(*, 1).underlying

    val line = gradientDescentInXY(X, Y)
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
    val xs:Vector[Double] = (-8.0 to 8.0 by 0.1).toVector
    val ys:Vector[Double] = (-8.0 to 8.0 by 0.1).toVector
    val zs = xs zip ys map {
      case (x, y) => x * rg.nextDouble() * 2000 - y * rg.nextDouble() * 1717
    }
    val plane = gradientDescentInXYZ(xs, ys, zs)

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

  def gradientDescentInXY(X: DenseVector[Double], Y: DenseVector[Double]): Line = {
    var m: Double = 1.0
    var c: Double = 1.0
    var mTemp: Double = 0.0
    var cTemp: Double = 0.0

    val count = X.length
    val alpha: Double = 0.00005 / count

    val precision = 0.000001 / count

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

      //      println(s"($sumM,$sumC)")
      //      println(s"($mTemp,$cTemp)")
      //      scala.io.StdIn.readLine()
    }

    Line(m, c)
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
      //scala.io.StdIn.readLine()
    }

    Plane(a, b, c)
  }
}