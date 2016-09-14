package train;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import cnn.CNN;
import lstm.LSTM;
import preprocessing.DatabaseInterface;
import preprocessing.Word2VecModeler;

public class Main {

	static final int VECTOR_LENGTH = 300;
	static final int NUM_LABELS = 2;
	static final int BATCHSIZE = 32;
	static final int MAX_LENGTH = 300;
	
	// T1 dataset
	static final String FILE = "anzart_gesuch_janein_mit_header_ZUF_und_OP.csv";
	static final String LABEL_COLUMN = "soll";
	static final String[] COLUMNS = new String[] { 
			"BREADCRUMP", 
			"NACHFRAGEART_TEXT", 
			"TITEL_FREITEXT", 
			"BESCHREIBUNG" };

	// T2 dataset
//	static final String FILE = "objektart_mit_header_ZUF_und_OP.csv";
//	static final String LABEL_COLUMN = "soll";
//	static final String[] COLUMNS = new String[] { 
//			"BREADCRUMP",
//			"OBJEKTTYP_FREITEXT", 
//			"OBJTYP_WOHNUNG_TEXT", 
//			"TITEL_FREITEXT", 
//			"BESCHREIBUNG" };
	
	public static void main(String[] args) throws Exception {

		DatabaseInterface db = new DatabaseInterface(new File(FILE), COLUMNS, LABEL_COLUMN);

		// second instance for testing
		DatabaseInterface db2 = new DatabaseInterface(new File(FILE), COLUMNS, LABEL_COLUMN);
		
		// create Word2Vec model
		db.writeSenteceFile(COLUMNS);
		File f = db.getSentenceFile();
		Word2VecModeler m = new Word2VecModeler(f, VECTOR_LENGTH);
		Word2Vec vec = m.getModel();

		// get Data
		DataSetIterator iteratorTrain = new LSTMIterator(db, NUM_LABELS, vec, BATCHSIZE, MAX_LENGTH);
		DataSetIterator iteratorTest = new LSTMIterator(db2, NUM_LABELS, vec, BATCHSIZE, MAX_LENGTH);

		// train and test model
		int folds = 10;
		int iterations = 10;

		for (int i = 0; i < folds; i++) {
			Evaluation evaluation = new Evaluation();
			System.out.println("Iteration " + Integer.toString(i + 1));
			
			// create Network
			LSTM builder = new LSTM(NUM_LABELS, VECTOR_LENGTH, MAX_LENGTH);
			MultiLayerNetwork model = builder.getModel();

			DataSetIterator kFoldTrain = new KFoldIterator(iteratorTrain, folds, i, true);
			DataSetIterator kFoldTest = new KFoldIterator(iteratorTest, folds, i, false);

			train(iterations, kFoldTrain, model);
			testLSTM(kFoldTest, model, evaluation);
			System.out.println(evaluation.stats() + "\n" + evaluation.getConfusionMatrix().toString());
		}
	}

	public static void train(int nEpochs, DataSetIterator iterator, MultiLayerNetwork model) {
		model.setListeners(new ScoreIterationListener(100));
//      model.setListeners(new HistogramIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(iterator);
		}
	}

	public static void testCNN(DataSetIterator iterator, MultiLayerNetwork model, Evaluation evaluation) {
		boolean set = false;
		int test = 0;
		while(iterator.hasNext()){
            DataSet t = iterator.next();
            if (!set) {
            	test = iterator.cursor();
            	set = true;
            }
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);
            
            // outputs wrongly predicted ads
            for (int i = 0; i < labels.rows(); i++){
            	if (getLabel(labels.getRow(i)) != getLabel(predicted.getRow(i)))
            		System.out.println((test+2) + ": " 
            				+ getLabel(labels.getRow(i)) + " " 
            				+ getLabel(predicted.getRow(i)));
            	test++;
            }
            
            evaluation.eval(labels,predicted);
        }
	}
	
	public static void testLSTM(DataSetIterator iterator, MultiLayerNetwork model, Evaluation evaluation) {
		boolean set = false;
		int test = 0;
		while(iterator.hasNext()){
            DataSet t = iterator.next();
            if (!set) {
            	test = iterator.cursor();
            	set = true;
            }
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray inMask = t.getFeaturesMaskArray();
            INDArray outMask = t.getLabelsMaskArray();
            INDArray predicted = model.output(features,false,inMask,outMask);
            
            int totalOutputExamples = outMask.sumNumber().intValue();
            int outSize = labels.size(1);

            // outputs wrongly predicted ads
            INDArray labels2d = Nd4j.create(totalOutputExamples, outSize);
            INDArray predicted2d = Nd4j.create(totalOutputExamples, outSize);

            int rowCount = 0;
            for (int ex = 0; ex < outMask.size(0); ex++) {
                for (int i = 0; i < outMask.size(1); i++) {
                    if (outMask.getDouble(ex, i) == 0.0) continue;

                    labels2d.putRow(rowCount, labels.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(i)));
                    predicted2d.putRow(rowCount, predicted.get(NDArrayIndex.point(ex), NDArrayIndex.all(), NDArrayIndex.point(i)));

                    rowCount++;
                }
            }
            
            for (int i = 0; i < labels2d.rows(); i++){
            	if (getLabel(labels2d.getRow(i)) != getLabel(predicted2d.getRow(i)))
            		System.out.println((test+2) + ": " 
            				+ getLabel(labels2d.getRow(i)) + " " 
            				+ getLabel(predicted2d.getRow(i)));
            	test++;
            }
            
            evaluation.evalTimeSeries(labels,predicted,outMask);
        }
	}
	
	/**
	 * Gets predicted label from output; only works for cnn and for 2d-ized lstm outputs
	 * @param labels
	 * @return predicted label
	 */
	public static int getLabel(INDArray labels){
		int r = 0;
		double max = 0;
		for (int i=0; i < labels.columns(); i++){
			if (labels.getDouble(i) > max){
				r = i;
				max = labels.getDouble(i);
			}	
		}
		return r;
	}
}
