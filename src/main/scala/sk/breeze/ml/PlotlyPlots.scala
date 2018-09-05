package sk.breeze.ml

import co.theasi.plotly.{Plot, ThreeDPlot}
import co.theasi.plotly._
import util.Random

object PlotlyPlots {
  def main(args: Array[String]): Unit = {
    val xs = (-3.0 to 3.0 by 0.1).toVector
    val ys = (-3.0 to 3.0 by 0.1).toVector

    def gaussian2D(x: Double, y: Double) = x+y
    val zs = xs.map { x => ys.map { y => gaussian2D(x, y) } }

    val p = ThreeDPlot().withSurface(xs, ys, zs)

    draw(p, "gaussian-surfaces")

    val xs1 = (0.0 to 2.0 by 0.1)
    val ys1 = xs.map { x => x*x }

    val plot = Plot().withScatter(xs1, ys1)

    draw(plot, "my-first-plot")
  }
}
