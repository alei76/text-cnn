package preprocessing;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.apache.commons.csv.*;


public class DatabaseInterface {

	private File f;
	private File sentences;
	
	public DatabaseInterface(File f){
		this.f = f;
	}
	
	public void writeSenteceFile(String[] columns) throws IOException{
		sentences = new File("sentences.temp");
		PrintWriter writer;
		try {
			writer = new PrintWriter(sentences, "UTF-8");
		} catch (UnsupportedEncodingException e) {
			writer = new PrintWriter(sentences);
		}
		
		CSVParser p = CSVParser.parse(f, StandardCharsets.UTF_8, CSVFormat.TDF.withHeader().withQuote(null));
		System.out.println(p.getHeaderMap());
		Iterator<CSVRecord> iter = p.iterator();
		
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
