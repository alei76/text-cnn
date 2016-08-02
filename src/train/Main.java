package train;

import java.io.File;
import java.util.Arrays;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;

import cnn.CNN;
import train.EntryIterator;
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
		

		// create Network
		CNN builder = new CNN(outputs, vectorLength, maxLength);
		MultiLayerNetwork model = builder.getModel();
		
		// create Word2Vec model
		db.writeSenteceFile(new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT" , "BESCHREIBUNG"});
		File f = db.getSentenceFile();
		Word2VecModeler m = new Word2VecModeler(f);
		Word2Vec vec = m.getModel();
			
		
		// get Data
		DataSetIterator iterator = new EntryIterator(db, numLabels, vec, batchsize, maxLength, false);
		DataSet t = iterator.next();
		System.out.println(t.getFeatures());
		System.out.println(Arrays.toString(t.getFeatures().shape()));
		System.out.println(Arrays.toString(t.getLabels().shape()));
//		System.out.println(model.numLabels());
//		System.out.println(Arrays.toString(model.getLabels().shape()));
//		System.out.println(model.getLabels());
		
		// train model
		train(10, iterator, model);
		
	}
	
	public static void train(int nEpochs, DataSetIterator iterator, MultiLayerNetwork model){
		model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(iterator);
        }
	}

}
