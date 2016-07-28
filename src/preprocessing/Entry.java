package preprocessing;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

import org.deeplearning4j.models.word2vec.Word2Vec;

public class Entry {

	public static void main(String[] args) throws Exception {
		DatabaseInterface i = new DatabaseInterface(new File("anzart_gesuch_janein_mit_header.csv"));
		try {
			i.writeSenteceFile(new String[] { "NACHFRAGEART_TEXT", "TITEL_FREITEXT" , "BESCHREIBUNG"});
		} catch (IOException e) {
		}

		File f = i.getSentenceFile();
		
		Word2VecModeler m = new Word2VecModeler(f);
		
		Word2Vec vec = m.getModel();

		Collection<String> lst = vec.wordsNearest("suche", 10);
		System.out.println(lst);
		
	}

}