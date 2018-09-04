import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.util.Success

object QSorter {

  import scala.util.Random

  val size = 5000000
  val rg = new Random

  var ab = ArrayBuffer.empty[Int]

  def apply() = {
    ab = ((ArrayBuffer.empty[Int]) /: (1 to size)) ((accum, i) => accum += rg.nextInt(size))
    println("Generated")
  }

  private def swap(x: Int, y: Int): Unit = {
    val temp = ab(x)
    ab(x) = ab(y)
    ab(y) = temp
  }

  private def partition(start: Int, end: Int, pivot: Int): Int = {
    var p = start + 1
    swap(start, pivot)
    for (x <- start + 1 to end) {
      if (ab(start) >= ab(x)) {
        swap(p, x)
        p += 1
      }
    }
    swap(start, p - 1)
    return p - 1
  }

  def qsort: Unit = qsort(0, size - 1)

  private def qsort(start: Int, end: Int): Unit = {
    if (start < end) {
      val p = partition(start, end, start)
      import scala.concurrent.ExecutionContext.Implicits.global
      Future {
        qsort(start, p - 1)
      }.onComplete { case Success(e) => println(start + "," + (p - 1)) }
      Future {
        qsort(p + 1, end)
      }.onComplete { case Success(e) => println(p + 1 + "," + end) }
    }
  }

  def isSorted = {
    ((true, Int.MinValue) /: ab) ((x, y) => if (!x._1 || x._2 > y) (false, y) else (true, y))._1

  }
}
