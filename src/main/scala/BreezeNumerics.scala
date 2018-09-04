import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.optimize.linear.LinearProgram

object BreezeNumerics{
  def main(args: Array[String]): Unit = {
    val lp = new LinearProgram
    import lp._
    val x = DenseVector.fill[Double](5,16.6)
    println(x)
    println(x*Transpose(x))
    println(inv(DenseMatrix.fill[Double](3,3)(0)))
    //println(det(x))
  }
}
