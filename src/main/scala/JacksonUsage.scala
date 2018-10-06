package main

import scala.io._
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper

object JacksonUsage{

  def main(args: Array[String]): Unit = {
    val a = getSetOfStrings("/Users/sulabhk/Desktop/admin_pz.json", "result")
    val b = getSetOfStrings("/Users/sulabhk/Desktop/category_director_pz.json", "result")
    println(getJson(a diff b))
    println(getJson(b diff a))
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


