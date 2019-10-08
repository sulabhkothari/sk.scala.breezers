package cnn

import java.io.{BufferedReader, File, InputStreamReader}

object RunTesseractOnImage extends App {
  val rt = Runtime.getRuntime()
  val pr = rt.exec("tesseract -l eng /Users/sulabhk/IdeaProjects/sk.scala.breezers/src/main/resources/aadhar.jpg /Users/sulabhk/IdeaProjects/sk.scala.breezers/src/main/resources/outpa")

  val exitVal = pr.waitFor()
  System.out.println("Exited with error code " + exitVal)

}
