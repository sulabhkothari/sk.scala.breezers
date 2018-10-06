package main

import scala.io._
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper

object Diff{

  val baseComparePath = "/Users/sulabhk/Desktop/roles_pcas.json"

  val kc1cd = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_1/policies/category_director_pz.json"
  val kc2cd = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_2/policies/category_director_pz.json"
  val kc3cd = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_3/policies/category_director_pz.json"
  val kc4cd = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_4/policies/category_director_pz.json"

  val kc1md = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_1/policies/marketing_director_pz.json"
  val kc2md = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_2/policies/marketing_director_pz.json"
  val kc3md = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_3/policies/marketing_director_pz.json"
  val kc4md = "/Users/sulabhk/mkyprojects/src/git.periscope-solutions.com/price/price-grocery-client-configurator/clients/client2_4/policies/marketing_director_pz.json"

  def main(args: Array[String]): Unit = {
    // read
    val baseSet = getSetOfStrings(baseComparePath, "result")

    val kc1Cd = getSetOfStrings(kc1cd, "actions")
    val kc2Cd = getSetOfStrings(kc2cd, "actions")
    val kc3Cd = getSetOfStrings(kc3cd, "actions")
    val kc4Cd = getSetOfStrings(kc4cd, "actions")

    val kc1Md = getSetOfStrings(kc1md, "actions")
    val kc2Md = getSetOfStrings(kc2md, "actions")
    val kc3Md = getSetOfStrings(kc3md, "actions")
    val kc4Md = getSetOfStrings(kc4md, "actions")

    println(s"\nDiff1 => ${getJson( baseSet diff kc1Cd )}")
    println(s"Diff2 => ${getJson( baseSet diff kc2Cd )}")
    println(s"Diff3 => ${getJson( baseSet diff kc3Cd )}")
    println(s"Diff4 => ${getJson( baseSet diff kc4Cd )}\n")

    println(s"Kc1 CD => ${getJson( baseSet union kc1Cd)}")
    println(s"Diff => ${getJson( kc1Cd diff baseSet )}\n")
    println(s"Kc2 CD => ${getJson( baseSet union kc2Cd)}")
    println(s"Diff => ${getJson( kc2Cd diff baseSet)}\n")
    println(s"Kc3 CD => ${getJson( baseSet union kc3Cd)}")
    println(s"Diff => ${getJson( kc3Cd diff baseSet)}\n")
    println(s"Kc4 CD => ${getJson( baseSet union kc4Cd)}")
    println(s"Diff => ${getJson( kc4Cd diff baseSet)}\n")

    println(s"Kc1 MD => ${getJson( baseSet union kc1Md)}")
    println(s"Diff => ${getJson( kc1Md diff baseSet)}\n")
    println(s"Kc2 MD => ${getJson( baseSet union kc2Md)}")
    println(s"Diff => ${getJson( kc2Md diff baseSet)}\n")
    println(s"Kc3 MD => ${getJson( baseSet union kc3Md)}")
    println(s"Diff => ${getJson( kc3Md diff baseSet)}\n")
    println(s"Kc4 MD => ${getJson( baseSet union kc4Md)}")
    println(s"Diff => ${getJson( kc4Md diff baseSet)}\n")
  }

  def getJson(set:Set[String]) = {
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    mapper.writeValueAsString(set)
  }
  def getSetOfStrings(file:String, attr:String) = {
    val json = Source.fromFile(file)
    // parse
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    val parsedJson = mapper.readValue[Map[String, Object]](json.reader())
    println(parsedJson)
    parsedJson(attr).asInstanceOf[List[String]].toSet
  }
}


