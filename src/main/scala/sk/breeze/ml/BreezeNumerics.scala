package sk.breeze.ml

import breeze.linalg._
import breeze.numerics.sigmoid
import breeze.plot._
//import breeze.numerics._
import breeze.optimize.linear.LinearProgram
import breeze.plot.Figure
import breeze.linalg._
import breeze.plot._
import math._

object BreezeNumerics{
  var i:Double = 10
  def geti = {
    i = i + sin(i)
    i
  }
  def main(args: Array[String]): Unit = {
    val lp = new LinearProgram
    val denseVec = DenseVector.fill[Double](5,16.6)
    //println(denseVec)
    //println(denseVec*Transpose(denseVec))
    //println(inv(DenseMatrix.fill[Double](3,3)(0)))
    //println(det(x))
//val f = Figure()
    //val p = f.subplot(0)
    val x = linspace(0.0,1.0)
//    p += plot(x, x :^ 2.0)
//    p += plot(x, x :^ 3.0, '.')
//    println(DenseMatrix.tabulate[Double](3,3)((i,j)=>i*j))
//    println(denseMat(*,0).underlying)
//    println(denseMat(0,0))
//    println(denseMat(1,0))
//    println(denseMat(0,1))
//    val x = linspace(0.0,1.0)
//    println(x)


    //val result = inv(Transpose(X).inner*X) //* Transpose(X) * Y

//    val result1 = DenseMatrix.tabulate[Double](3,3)((i,j)=>i*j)
//    val result = DenseMatrix.tabulate[Double](1,1)((i,j)=>i*j+100)




    //println(Y)

    //p += plot(denseMat(*,0).underlying,denseMat(*,1).underlying)
//
//    p.xlabel = "x axis"
//    p.ylabel = "y axis"
    //f.saveas("lines.png")
  }
}
