package train;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.IOException;
import java.util.List;

@SuppressWarnings("serial")
public class KFoldIterator implements DataSetIterator {
	private final DataSetIterator iter;
	private final int folds, iteration;
	private final int lowerBound, upperBound;
	private final boolean train;

	private int cursor;

	public KFoldIterator(DataSetIterator iter, int folds, int iteration, boolean train) throws IOException {
		this.iter = iter;
		iter.reset();

		this.train = train;
		this.folds = folds;
		this.iteration = iteration;

		int range = iter.numExamples();
		lowerBound = iteration * range / folds;
		upperBound = (iteration + 1) * range / folds;
		cursor = 0;
	}

	@Override
	public DataSet next(int num) {
		if (train) {
			while (cursor > lowerBound && cursor < upperBound) {
				cursor += num;
				iter.next(num);
			}
			cursor += num;
			return iter.next(num);
		} else {
			while (cursor < lowerBound) {
				cursor += num;
				iter.next(num);
			}
			cursor += num;
			return iter.next(num);
		}
	}

	@Override
	public int totalExamples() {
		if (train)
			return iter.totalExamples() - (upperBound - lowerBound);
		else
			return (upperBound - lowerBound);
	}

	@Override
	public int inputColumns() {
		return iter.inputColumns();
	}

	@Override
	public int totalOutcomes() {
		return iter.totalOutcomes();
	}

	@Override
	public void reset() {
		cursor = 0;
		iter.reset();
	}

	@Override
	public int batch() {
		return iter.batch();
	}

	@Override
	public int cursor() {
		return cursor();
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		iter.setPreProcessor(preProcessor);
	}

	@Override
	public List<String> getLabels() {
		return iter.getLabels();
	}

	@Override
	public boolean hasNext() {
		return iter.hasNext() 
				// skip last elements if training and last iteration
				&& !(train && cursor > lowerBound && folds == iteration + 1)
				// stop if testing is done
				&& !(!train && cursor > upperBound);
	}

	@Override
	public DataSet next() {
		return next(iter.batch());
	}

	@Override
	public void remove() {
		iter.remove();
	}
}
