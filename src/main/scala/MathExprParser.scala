import scala.util.parsing.combinator._

class Expr(val left: Option[Expr], val operator: Option[String], val right: Option[Expr], val value: Option[Float])

object Expr {
  def unapply(arg: Expr): Option[(Option[Expr], Option[String], Option[Expr], Option[Float])] = {
    Some((arg.left, arg.operator, arg.right, arg.value))
  }
}

case class Leaf(v: Float) extends Expr(None, None, None, Some(v))

class MathExpr extends JavaTokenParsers {
  def expr: Parser[Expr] = term ~ rep("+" ~ term | "-" ~ term) ^^ (moreParse(_))

  def term: Parser[Expr] = factor ~ rep("*" ~ factor | "/" ~ factor) ^^ (moreParse(_))

  def factor: Parser[Expr] = floatingPointNumber ^^ (x => Leaf(x.toFloat)) | "(" ~> expr <~ ")"

  def moreParse(p: Any): Expr = p match {
    case x ~ List() if x.isInstanceOf[Expr] => x.asInstanceOf[Expr]
    case Leaf(x) ~ (y :: ys) =>
      val v = getOpAndOperand(y, ys)
      new Expr(Some(Leaf(x)), Some(v._1), Some(v._2), None)
    case x ~ (y :: ys) if x.isInstanceOf[Expr] =>
      val v = getOpAndOperand(y, ys)
      new Expr(Some(x.asInstanceOf[Expr]), Some(v._1), Some(v._2), None)
  }

  def getOpAndOperand(y: Any, ys: Any): (String, Expr) = {
    val (m, n) = y match {
      case a ~ Leaf(b) => (a.toString, Leaf(b))
      case a ~ Expr(l, o, r, v) => (a.toString, new Expr(l, o, r, v))
    }
    if (ys.asInstanceOf[List[Any]].isEmpty) (m, n)
    else {
      val (p, q) = ys match {
        case a :: b => getOpAndOperand(a, b)
      }
      (m, new Expr(Some(n), Some(p), Some(q), None))
    }
  }
}

object MathExprParser extends MathExpr {
  def evaluate(exp: Expr): Leaf = exp match {
    case Leaf(x) => Leaf(x)
    case Expr(left, oper, right, value) => calc(evaluate(left.get), evaluate(right.get), oper.get)
  }

  def calc(left: Leaf, right: Leaf, op: String) = Leaf(op match {
    case "*" => left.value.get * right.value.get
    case "/" => left.value.get / right.value.get
    case "+" => left.value.get + right.value.get
    case "-" => left.value.get - right.value.get
  })

  def main(args: Array[String]): Unit = {
    val exp = "((2+3)-4/(3*(((7+6)-4)/7)/2))"
    println(parse(expr, exp))
    val result = parse(expr, exp)
    println(result)
    println(evaluate(result.get.asInstanceOf[Expr]).value)
    println(parseAll(expr, "((2+3)-4)/2"))
  }
}