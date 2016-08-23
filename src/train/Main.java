package train;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import cnn.CNN;
import preprocessing.DatabaseInterface;
import preprocessing.Word2VecModeler;

public class Main {

	static int vectorLength = 100;
	static int numLabels = 3;
	static int batchsize = 32;
	static int maxLength = 100;
	
	static String file = "objektart_mit_header.csv";
	static String result = "soll";
	static String[] columns = new String[] { 
			"OBJEKTTYP_FREITEXT", 
			"OBJTYP_WOHNUNG_TEXT", 
			"TITEL_FREITEXT", 
			"BESCHREIBUNG" };

	public static void main(String[] args) throws Exception {

		DatabaseInterface db = new DatabaseInterface(new File(file), columns, result);

		// second instance for testing
		DatabaseInterface db2 = new DatabaseInterface(new File(file), columns, result);
		
		// create Word2Vec model
		db.writeSenteceFile(columns);
		File f = db.getSentenceFile();
		Word2VecModeler m = new Word2VecModeler(f);
		Word2Vec vec = m.getModel();

		// get Data
		DataSetIterator iteratorTrain = new CNNIterator(db, numLabels, vec, batchsize, maxLength);
		DataSetIterator iteratorTest = new CNNIterator(db2, numLabels, vec, batchsize, maxLength);

		// train and test model
		int folds = 10;
		int iterations = 10;
		Evaluation evaluation = new Evaluation();

		for (int i = 0; i < folds; i++) {
			System.out.println("Iteration " + Integer.toString(i + 1));
			
			// create Network
			CNN builder = new CNN(numLabels, vectorLength, maxLength);
			MultiLayerNetwork model = builder.getModel();

			DataSetIterator kFoldTrain = new KFoldIterator(iteratorTrain, folds, i, true);
			DataSetIterator kFoldTest = new KFoldIterator(iteratorTest, folds, i, false);

			train(iterations, kFoldTrain, model);
			testCNN(kFoldTest, model, evaluation);
			System.out.println(evaluation.stats() + "\n" + evaluation.getConfusionMatrix().toString());
		}
	}

	public static void train(int nEpochs, DataSetIterator iterator, MultiLayerNetwork model) {
		model.setListeners(new ScoreIterationListener(100));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(iterator);
		}
	}

	public static void testCNN(DataSetIterator iterator, MultiLayerNetwork model, Evaluation evaluation) {
        while(iterator.hasNext()){
            DataSet t = iterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);

            evaluation.eval(labels,predicted);
        }
	}
	
	public static void testLSTM(DataSetIterator iterator, MultiLayerNetwork model, Evaluation evaluation) {
        while(iterator.hasNext()){
            DataSet t = iterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray inMask = t.getFeaturesMaskArray();
            INDArray outMask = t.getLabelsMaskArray();
            INDArray predicted = model.output(features,false,inMask,outMask);

            evaluation.evalTimeSeries(lables,predicted,outMask);
        }
	}
}
