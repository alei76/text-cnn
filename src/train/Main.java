package train;

import java.io.File;
import java.util.ArrayList;

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

	public static void main(String[] args) throws Exception {
		int vectorLength = 100;
		int outputs = 2;
		int numLabels = 2;
		int batchsize = 32;
		int maxLength = 100;
		
		DatabaseInterface db = new DatabaseInterface(
				new File("anzart_gesuch_janein_mit_header.csv"),
				new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT" , "BESCHREIBUNG"},
				"soll");
		DatabaseInterface db2 = new DatabaseInterface(
				new File("anzart_gesuch_janein_mit_header.csv"),
				new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT" , "BESCHREIBUNG"},
				"soll");
		

		// create Network
		CNN builder = new CNN(outputs, vectorLength, maxLength);
		MultiLayerNetwork model = builder.getModel();
		
		// create Word2Vec model
		db.writeSenteceFile(new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT" , "BESCHREIBUNG"});
		File f = db.getSentenceFile();
		Word2VecModeler m = new Word2VecModeler(f);
		Word2Vec vec = m.getModel();
			
		
		// get Data
		DataSetIterator iteratorTrain = new EntryIterator(db, numLabels, vec, batchsize, maxLength);
		DataSetIterator iteratorTest = new EntryIterator(db2, numLabels, vec, batchsize, maxLength);
//		DataSet t = iterator.next();
//		System.out.println(Arrays.toString(t.getFeatures().shape()));
//		System.out.println(Arrays.toString(t.getLabels().shape()));
//		System.out.println(model.numLabels());
//		System.out.println(Arrays.toString(model.getLabels().shape()));
		
		// train and test model
		int folds = 10;
		int iterations = 10;
		double[] scores = new double[folds];
		for (int i = 0; i < folds; i++){
			DataSetIterator kFoldTrain = new KFoldIterator(iteratorTrain, folds, i, true);
			DataSetIterator kFoldTest = new KFoldIterator(iteratorTest, folds, i, false);
			train(iterations, kFoldTrain, model);
			ArrayList<Boolean> test = test(kFoldTest, model);
			int correct = 0;
			for (int j = 0; j < test.size(); j++){
				if (test.get(j))
					correct++;
			}
			scores[i] = ((double) correct) / ((double) test.size());
		}
		for (double score : scores) {
			System.out.print(score);
		}
	}
	
	public static void train(int nEpochs, DataSetIterator iterator, MultiLayerNetwork model){
		model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(iterator);
        }
	}
	
	public static ArrayList<Boolean> test(DataSetIterator iterator, MultiLayerNetwork model) {
		ArrayList<Boolean> res = new ArrayList<>();
		while(iterator.hasNext()) {
			DataSet d = iterator.next();
			INDArray out = model.output(d.getFeatureMatrix());
			INDArray labels = d.getLabels();
			for (int i = 0; i < out.rows(); i++) {
				boolean b = getClass(out.getRow(i)) == getClass(labels.getRow(i));
				res.add(b);
			}
		}
		return res;
	}
	
	public static int getClass(INDArray vector) {
		int r = 0;
		double max = 0;
		for (int i = 0; i < vector.length(); i++){			
			if (vector.getDouble(i) > max) {
				max = vector.getDouble(i);
				r = i;
			}
		}
		return r;
	}
	

}
