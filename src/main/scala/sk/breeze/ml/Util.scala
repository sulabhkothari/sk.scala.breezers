package sk.breeze.ml

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.plot.{Figure, plot}

object Util {

  case class TrainingData(trainingDataPos: DenseMatrix[Double], trainingDataNeg: DenseMatrix[Double]) {
    val xPos = trainingDataPos(*, 0 to trainingDataPos.cols - 2).underlying
    val yPos = trainingDataPos(*, trainingDataPos.cols - 1).underlying
    val xNeg = trainingDataNeg(*, 0 to trainingDataNeg.cols - 2).underlying
    val yNeg = trainingDataNeg(*, trainingDataNeg.cols - 1).underlying

    lazy val posOnes = DenseVector.ones[Double](yPos.length)
    lazy val negOnes = DenseVector.ones[Double](yNeg.length)

    lazy val trainingData = DenseMatrix.vertcat(trainingDataPos, trainingDataNeg)

    def x = trainingData(*, 0 to trainingData.cols - 2).underlying

    def y = trainingData(*, trainingData.cols - 1).underlying

    lazy val randomTheta = {
      val normal01 = breeze.stats.distributions.Gaussian(0, 1)
      DenseVector.rand(trainingDataPos.cols - 1, normal01)
    }
  }

  case class Line(slope: Double, c: Double)

  case class Plane(a: Double, b: Double, c: Double)

  def prependOnesColumn(original: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones = DenseVector.ones[Double](original.rows)
    val dataWithOnes = ones.data ++ original.data
    DenseMatrix.create(original.rows, original.cols + 1, dataWithOnes)
  }

  def prependOnesRow(original: DenseMatrix[Double]): DenseMatrix[Double] = {
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

    val dm1 = DenseMatrix.tabulate[Double](xyPositives.length, 2)((i, j) => xyPositives(i).underlying(j + 1))


    val xyNegatives = for {
      i <- 0 to xy.rows - 1 if (xy(i, *).underlying(3) < 0.5)
    }
      yield xy(i, *)
    val dm2 = DenseMatrix.tabulate[Double](xyNegatives.length, 2)((i, j) => xyNegatives(i).underlying(j + 1))

    val f = Figure()
    val p = f.subplot(0)
    p += plot(dm1(*, 0).underlying, dm1(*, 1).underlying, '.')
    p += plot(dm2(*, 0).underlying, dm2(*, 1).underlying, '+')


  }

  def randomVector(size: Int) = {
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseVector.rand(size, normal01)
  }

  def prepareTrainingData(dataSize: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val rg = new scala.util.Random
    (
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i
          case 1 => i * rg.nextDouble + 200.99
          case 2 => 1
        })),
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i - dataSize.asInstanceOf[Double] / 2
          case 1 => (i - dataSize.asInstanceOf[Double] / 2) * rg.nextDouble + 48.49
          case 2 => if (i - dataSize.asInstanceOf[Double] / 2 < 0) 0 else 1
        })))
  }

  def prepareTrainingDataNoBias(dataSize: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val rg = new scala.util.Random

    val posPoints = for {
      x <- dataSize to 2 * dataSize by 1.9
      y <- dataSize to 2 * dataSize by 1.9
    }
      yield (x*x * rg.nextDouble() * 10, y*y * rg.nextDouble() * 10, 1.0)

    val negPoints = for {
      x <- -dataSize to 0 by 11.3
      y <- -dataSize to 0 by 21.1
    }
      yield (-x*x * rg.nextDouble() * 9, -y*y * rg.nextDouble() * 10, 0.0)

    (DenseMatrix(posPoints:_*), DenseMatrix(negPoints:_*))
  }

  def prepareTrainingData(dataSize: Int, pos: Double, neg: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val rg = new scala.util.Random
    (
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i
          case 1 => i * rg.nextDouble + 200.99
          case 2 => pos
        })),
      Util.prependOnesColumn(DenseMatrix.tabulate[Double](dataSize / 2, 3)((i, j) =>
        j match {
          case 0 => i - dataSize.asInstanceOf[Double] / 2
          case 1 => (i - dataSize.asInstanceOf[Double] / 2) * rg.nextDouble + 48.49
          case 2 => if (i - dataSize.asInstanceOf[Double] / 2 < 0) neg else pos
        })))
  }

  def prepareClassificationData(dataSize: Double): DenseMatrix[Double] = {
    val points = for {
      x <- -dataSize to dataSize by 200
      y <- -dataSize to dataSize by 200
    }
      yield (x, y)

    prependOnesColumn(DenseMatrix(points: _*))
  }

  def prepareClassificationDataNoBias(dataSize: Double): DenseMatrix[Double] = {
    val points = for {
      x <- -dataSize to dataSize by 200
      y <- -dataSize to dataSize by 200
    }
      yield (x, y)

    DenseMatrix(points: _*)
  }
}
