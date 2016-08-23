package cnn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CNN {
	
	private MultiLayerNetwork model;
	
	public CNN(int numClasses, int vectorLength, int textLength){
		int nChannels = 1;
        int iterations = 1;
        int seed = 42;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, vectorLength - 1)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(64)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,1)
                        .stride(2,1)
                        .build())
//                .layer(2, new ConvolutionLayer.Builder(5, vectorLength / 2)
//                        .nIn(nChannels)
//                        .stride(1, 1)
//                        .nOut(64)
//                        .activation("identity")
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .build())
                .layer(2, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,vectorLength,textLength,nChannels);

        MultiLayerConfiguration conf = builder.build();
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
	}
	
	public MultiLayerNetwork getModel(){
		return model;
	}
	
	
}
