package cnn

import java.io.File
import java.util

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.slf4j.{Logger, LoggerFactory}
import java.util.Random

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{MapSchedule, ScheduleType}

object BirdClassifier extends App {
  val height = 28    // height of the picture in px
  val width = 28     // width of the picture in px
  val channels = 1   // single channel for grayscale images
  val outputNum = 200 // 10 digits classification
  val batchSize = 54 // number of samples that will be propagated through the network in each iteration
  val nEpochs = 10    // number of training epochs

  val seed = 1234    // number used to initialize a pseudorandom number generator.
  val randNumGen = new Random(seed)
  println("Data vectorization...")
  // vectorization of train data
  val trainData = new File("/Users/sulabhk/Downloads/images")
  val trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
  val labelMaker = new ParentPathLabelGenerator() // use parent directory name as the image label
  val trainRR = new ImageRecordReader(height, width, channels, labelMaker)
  trainRR.initialize(trainSplit)
  val trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)

  val imageScaler = new ImagePreProcessingScaler()
  imageScaler.fit(trainIter)
  trainIter.setPreProcessor(imageScaler)

  // vectorization of test data
  val testData = new File("/Users/sulabhk/Downloads/images")
  val testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
  val testRR = new ImageRecordReader(height, width, channels, labelMaker)
  testRR.initialize(testSplit)
  val testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
  testIter.setPreProcessor(imageScaler)

  println("Network configuration and training...")

  // reduce the learning rate as the number of training epochs increases
  // iteration #, learning rate
  var learningRateSchedule: util.Map[Integer, java.lang.Double] = new util.HashMap
  learningRateSchedule.put(0, 0.06)
  learningRateSchedule.put(200, 0.05)
  learningRateSchedule.put(600, 0.028)
  learningRateSchedule.put(800, 0.0060)
  learningRateSchedule.put(1000, 0.001)


  //Initialize the user interface backend
//  UIServer uiServer = UIServer.getInstance();
//
//  //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//  val statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
//
//  //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//  uiServer.attach(statsStorage);
//
//  //Then add the StatsListener to collect this information from the network, as it trains
//  net.setListeners(new StatsListener(statsStorage));
  val conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .l2(0.0005) // ridge regression value
    .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
    .weightInit(WeightInit.XAVIER)
    .list()
    .layer(new ConvolutionLayer.Builder(5, 5)
      .nIn(channels)
      .stride(1, 1)
      .nOut(20)
      .activation(Activation.IDENTITY)
      .build())
    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build())
    .layer(new ConvolutionLayer.Builder(5, 5)
      .stride(1, 1) // nIn need not specified in later layers
      .nOut(50)
      .activation(Activation.IDENTITY)
      .build())
    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build())
    .layer(new DenseLayer.Builder().activation(Activation.RELU)
      .nOut(500)
      .build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .activation(Activation.SOFTMAX)
      .build())
    .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
    .build()


  val net = new MultiLayerNetwork(conf)
  net.init()
  net.setListeners(new ScoreIterationListener(10))
  println(s"Total num of params: ${net.numParams()}")

  // evaluation while training (the score should go down)
  for(i <- 0 to nEpochs - 1){
    net.fit(trainIter)
    println(s"Completed epoch $i")
    val eval = net.evaluate[Evaluation](testIter)
    //println(eval.stats(false,true))
    println(eval.stats)

    trainIter.reset()
    testIter.reset()
  }
}
