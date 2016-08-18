package train;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
	static int numLabels = 2;
	static int batchsize = 32;
	static int maxLength = 100;

	public static void main(String[] args) throws Exception {

		DatabaseInterface db = new DatabaseInterface(new File("anzart_gesuch_janein_mit_header.csv"),
				new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT", "BESCHREIBUNG" }, "soll");

		// second instance for testing
		DatabaseInterface db2 = new DatabaseInterface(new File("anzart_gesuch_janein_mit_header.csv"),
				new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT", "BESCHREIBUNG" }, "soll");

		// create Word2Vec model
		db.writeSenteceFile(new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT", "BESCHREIBUNG" });
		File f = db.getSentenceFile();
		Word2VecModeler m = new Word2VecModeler(f);
		Word2Vec vec = m.getModel();

		// get Data
		DataSetIterator iteratorTrain = new EntryIterator(db, numLabels, vec, batchsize, maxLength);
		DataSetIterator iteratorTest = new EntryIterator(db2, numLabels, vec, batchsize, maxLength);

		// train and test model
		int folds = 10;
		int iterations = 10;
		int[][][] scores = new int[folds][][];

		for (int i = 0; i < folds; i++) {
			System.out.println("Iteration " + Integer.toString(i + 1));
			
			// create Network
			CNN builder = new CNN(numLabels, vectorLength, maxLength);
			MultiLayerNetwork model = builder.getModel();

			DataSetIterator kFoldTrain = new KFoldIterator(iteratorTrain, folds, i, true);
			DataSetIterator kFoldTest = new KFoldIterator(iteratorTest, folds, i, false);

			train(iterations, kFoldTrain, model);
			scores[i] = test(kFoldTest, model);
		}

		// print out confusion matrices
		for (int[][] test : scores) {
			for (int[] row : test) {
				for (int value : row) {
					System.out.print(value);
					System.out.print(" ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}

	public static void train(int nEpochs, DataSetIterator iterator, MultiLayerNetwork model) {
		model.setListeners(new ScoreIterationListener(100));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(iterator);
		}
	}

	/**
	 * 
	 * @param iterator
	 * @param model
	 * @return confusion matrix for the test
	 */
	public static int[][] test(DataSetIterator iterator, MultiLayerNetwork model) {
		int[][] res = new int[numLabels][];
		for (int i = 0; i < numLabels; i++) {
			res[i] = new int[numLabels];
			for (int j = 0; j < numLabels; j++) {
				res[i][j] = 0;
			}
		}
		while (iterator.hasNext()) {
			DataSet d = iterator.next();
			INDArray out = model.output(d.getFeatureMatrix());
			INDArray labels = d.getLabels();
			for (int i = 0; i < out.rows(); i++) {
				res[getClass(out.getRow(i))][getClass(labels.getRow(i))]++;
			}
		}
		return res;
	}
	
	/**
	 * 
	 * @param vector of the activations for the classes
	 * @return class index
	 */
	public static int getClass(INDArray vector) {
		int r = 0;
		double max = 0;
		for (int i = 0; i < vector.length(); i++) {
			if (vector.getDouble(i) > max) {
				max = vector.getDouble(i);
				r = i;
			}
		}
		return r;
	}

}
