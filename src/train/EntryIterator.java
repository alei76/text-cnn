package train;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import preprocessing.DatabaseInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

@SuppressWarnings("serial")
public class EntryIterator implements DataSetIterator {
	private final WordVectors wordVectors;
	private final int batchSize;
	private final int vectorSize;
	private final int truncateLength;
	private final int numLabels;

	private int cursor = 0;
	private final DatabaseInterface databaseInterface;
	private final TokenizerFactory tokenizerFactory;

	public EntryIterator(DatabaseInterface databaseInterface, int numLabels, WordVectors wordVectors, int batchSize,
			int truncateLength) throws IOException {
		this.batchSize = batchSize;
		this.vectorSize = wordVectors.lookupTable().layerSize();

		this.numLabels = numLabels;
		this.databaseInterface = databaseInterface;

		this.wordVectors = wordVectors;
		this.truncateLength = truncateLength;

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
	}

	@Override
	public DataSet next(int num) {
		if (cursor >= totalExamples())
			throw new NoSuchElementException();
		try {
			return nextDataSet(num);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private DataSet nextDataSet(int num) throws IOException {
//		System.out.println(cursor);
		// First: load reviews to String. Alternate positive and negative
		// reviews
		List<String> texts = new ArrayList<>(num);
		int[] label = new int[num];
		for (int i = 0; i < num && cursor < totalExamples(); i++) {
			String text = databaseInterface.getNextEntry();
			texts.add(text);
			label[i] = databaseInterface.getCurrentLabel();
			cursor++;
		}

		// Second: tokenize reviews and filter out unknown words
		List<List<String>> allTokens = new ArrayList<>(texts.size());
		for (String s : texts) {
			List<String> tokens = tokenizerFactory.create(s).getTokens();
			List<String> tokensFiltered = new ArrayList<>();
			for (String t : tokens) {
				if (wordVectors.hasWord(t))
					tokensFiltered.add(t);
			}
			allTokens.add(tokensFiltered);
		}

		// Create data for training
		// Here: we have texts.size() examples of varying lengths
		// Features have to be 2dim array for cnn to work: batchsize, vectorsize
		// x numWords
		INDArray features = Nd4j.create(texts.size(), vectorSize * truncateLength);
		INDArray labels = Nd4j.create(texts.size(), numLabels); 

		for (int i = 0; i < texts.size(); i++) {
			List<String> tokens = allTokens.get(i);
			// Get word vectors for each word in review, and put them in the
			// training data
			for (int j = 0; j < tokens.size() && j < truncateLength; j++) {
				String token = tokens.get(j);
				INDArray vector = wordVectors.getWordVectorMatrix(token);
				features.put(new INDArrayIndex[] { 
						NDArrayIndex.point(i),
						NDArrayIndex.interval(j * vectorSize, (j + 1) * vectorSize) }, 
						vector);
			}

			int idx = label[i];
			// Set label: [0,1] for negative, [1,0] for positive
			labels.putScalar(new int[] { i, idx }, 1.0);
		}

		return new DataSet(features, labels);
	}

	@Override
	public int totalExamples() {
		return databaseInterface.size();
	}

	@Override
	public int inputColumns() {
		return vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return numLabels;
	}

	@Override
	public void reset() {
		cursor = 0;
		try {
			databaseInterface.newParser();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList("positive", "negative");
	}

	@Override
	public boolean hasNext() {
		return cursor < (numExamples() - batchSize);
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public void remove() {

	}
}
