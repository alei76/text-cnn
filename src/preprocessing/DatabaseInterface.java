package preprocessing;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.apache.commons.csv.*;
import preprocessing.StringProcessor;

public class DatabaseInterface {

	private File sentences, db;
	private CSVParser p;
	private String[] columns;
	private String labelColumn;
	private int size;
	
	Iterator<CSVRecord> iter;
	CSVRecord currentRecord;
	
	public DatabaseInterface(File f, String[] columns, String labelColumn) throws IOException{
		this.db = f;
		this.columns = columns;
		this.labelColumn = labelColumn;
		size = 0;
		newParser();
		while (iter.hasNext()){
			size++;
			iter.next();
		}
//		System.out.println(size);
		newParser();
	}
	
	public void newParser() throws IOException {
		p = CSVParser.parse(db, 
				StandardCharsets.UTF_8, 
				CSVFormat.TDF.withHeader().withQuote(null));
		iter = p.iterator();
	}

	public String getNextEntry() throws IOException {
		String s = "";
		if (iter.hasNext()){
			currentRecord = iter.next();
			for (String c : columns){
				s = s + " " + StringProcessor.stringForCNN(currentRecord.get(c));
			}
		} else {
			throw new IOException();
		}
		return s;
	}
	
	public int getCurrentLabel(){ 
		String s = currentRecord.get(labelColumn);
		
		if (s.equals("false")) {
			return 0;
		} else if (s.equals("true")) {
			return 1;
		}
		
		return Integer.parseInt(s);
	}
	
	public void writeSenteceFile(String[] columns) throws Exception{
		sentences = new File("sentences.temp");
		PrintWriter writer = new PrintWriter(sentences, "UTF-8");
		
		while(iter.hasNext()){
			CSVRecord r = iter.next();
			for (String c : columns){
				writer.println(StringProcessor.stringToSentences(r.get(c)));
			}
        }
		
		writer.close();
		newParser();
	}

	public File getSentenceFile(){
		return sentences;
	}
	
	public void removeSentenceFile(){
		sentences.delete();
	}

	public int size() {
		return size;
	}

}
