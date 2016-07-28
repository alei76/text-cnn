package preprocessing;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.apache.commons.csv.*;


public class DatabaseInterface {

	private File sentences;
	private CSVParser p;
	private Iterator<CSVRecord> iter;
	
	public DatabaseInterface(File f) throws IOException{
		p = CSVParser.parse(f, 
				StandardCharsets.UTF_8, 
				CSVFormat.TDF.withHeader().withQuote(null));
		iter = p.iterator();
	}
	
	public void writeSenteceFile(String[] columns) throws Exception{
		sentences = new File("sentences.temp");
		PrintWriter writer;
		writer = new PrintWriter(sentences, "UTF-8");
		
		while(iter.hasNext()){
			CSVRecord r = iter.next();
			for (String c : columns){
				writer.println(StringProcessor.stringToSentences(r.get(c)));
			}
        }
		
		writer.close();
	}

	public File getSentenceFile(){
		return sentences;
	}
	
	public void removeSentenceFile(){
		sentences.delete();
	}
	
}
