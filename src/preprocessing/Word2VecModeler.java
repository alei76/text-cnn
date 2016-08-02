package preprocessing;

import java.io.File;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Word2VecModeler {
	private Word2Vec vec;

	public Word2VecModeler(File f) throws Exception {
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(f);
		// Split on white spaces in the line to get words
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		vec = new Word2Vec.Builder()
				.minWordFrequency(5)
				.iterations(1)
				.layerSize(100)
				.seed(42)
				.windowSize(5)
				.iterate(iter)
				.tokenizerFactory(t)
				.build();

		vec.fit();
	}

	public Word2VecModeler(File f, String saveModel) throws Exception {
		this(f);

		// Write word vectors
		WordVectorSerializer.writeWordVectors(vec, saveModel);
	}
	
	public Word2VecModeler(String loadModel) throws Exception {
		WordVectorSerializer.loadFullModel(loadModel);
	}

	public Word2Vec getModel() {
		return vec;
	}
}
