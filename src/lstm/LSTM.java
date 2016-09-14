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
}
