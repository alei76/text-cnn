package lstm;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTM {

	private MultiLayerNetwork model;
	
	public LSTM(int numClasses, int vectorLength, int textLength){
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .iterations(1)
	            .updater(Updater.RMSPROP)
	            .regularization(true).l2(0.00002)
	            .weightInit(WeightInit.XAVIER)
	            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
	            .gradientNormalizationThreshold(1.0)
	            .learningRate(0.002)
	            .list()
	            .layer(0, new GravesLSTM.Builder()
	            		.nIn(vectorLength)
	            		.nOut(200)
	                    .activation("softsign").build())
	            .layer(1, new RnnOutputLayer.Builder()
	            		.activation("softmax")
	                    .lossFunction(LossFunctions.LossFunction.MCXENT)
	                    .nIn(200)
	                    .nOut(numClasses).build())
	            .pretrain(false)
	            .backprop(true)
	            .build();
		
	    model = new MultiLayerNetwork(conf);
	    model.init();
	    model.setListeners(new ScoreIterationListener(100));	    
	}
    
	public MultiLayerNetwork getModel(){
		return model;
	}

//    //DataSetIterators for training and testing respectively
//    //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
//    WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);
//    DataSetIterator train = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,batchSize,truncateReviewsToLength,true),1);
//    DataSetIterator test = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,100,truncateReviewsToLength,false),1);
//
//    System.out.println("Starting training");
//    for( int i=0; i<nEpochs; i++ ){
//        net.fit(train);
//        train.reset();
//        System.out.println("Epoch " + i + " complete. Starting evaluation:");
//
//        //Run evaluation. This is on 25k reviews, so can take some time
//        Evaluation evaluation = new Evaluation();
//        while(test.hasNext()){
//            DataSet t = test.next();
//            INDArray features = t.getFeatureMatrix();
//            INDArray lables = t.getLabels();
//            INDArray inMask = t.getFeaturesMaskArray();
//            INDArray outMask = t.getLabelsMaskArray();
//            INDArray predicted = net.output(features,false,inMask,outMask);
//
//            evaluation.evalTimeSeries(lables,predicted,outMask);
//        }
//        test.reset();
//
//        System.out.println(evaluation.stats());
//    }
//
//
//    System.out.println("----- Example complete -----");
}
