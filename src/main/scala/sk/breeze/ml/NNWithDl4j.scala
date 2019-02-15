package sk.breeze.ml

import java.util

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd

object NNWithDl4j {
  def main(args: Array[String]): Unit = {
    import org.datavec.api.records.reader.RecordReader
    import org.datavec.api.records.reader.impl.csv.CSVRecordReader
    import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
    import org.deeplearning4j.nn.conf.MultiLayerConfiguration
    import org.deeplearning4j.nn.conf.NeuralNetConfiguration
    import org.deeplearning4j.nn.conf.layers.DenseLayer
    import org.deeplearning4j.nn.conf.layers.OutputLayer
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
    import org.deeplearning4j.optimize.listeners.ScoreIterationListener
    import org.nd4j.linalg.activations.Activation
    import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
    import org.nd4j.linalg.learning.config.Nesterovs
    import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
    val seed = 123
    val learningRate = 0.01
    val batchSize = 50
    val nEpochs = 30

    val numInputs = 2
    val numOutputs = 1
    val numHiddenNodes = 20

    //val filenameTrain = new Nothing("/classification/linear_data_train.csv").getFile.getPath
    //val filenameTest = new Nothing("/classification/linear_data_eval.csv").getFile.getPath

    //Load the training data:
    //val rr = new CSVRecordReader
    //        rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
    //rr.initialize(new Nothing(new Nothing(filenameTrain)))
    //val trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 1)

    //Load the test/evaluation data:
    //val rrTest = new CSVRecordReader
    //rrTest.initialize(new Nothing(new Nothing(filenameTest)))
    //val testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 1)

    val conf = new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(learningRate, 0.9))
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.RELU).build)
      .layer(1, new OutputLayer.Builder(LossFunction.XENT).activation(Activation.SIGMOID).nIn(numHiddenNodes).nOut(numOutputs).build)
      .build
//new DataSet(new INDArray())
    val input = Nd4j.create(Array(1.0,9.0,-1.9,-9.9), Array(2,2))
    val output = Nd4j.create(Array(1.0,0.0), Array(2,1))
    val model = new MultiLayerNetwork(conf)

    model.init()
    model.setListeners(new ScoreIterationListener(10)) //Print score every 10 parameter updates
    model.fit(input,output)

  }
}
