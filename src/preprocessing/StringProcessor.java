package preprocessing;

public class StringProcessor {

	static public String stringToSentences(String string) {
		string = seperatePunctuation(string);
		// comment in the next line to apply splitting of word compounds
//		string = additionalReplacements(string);
		return string;
	}
	
	static public String stringForCNN(String string) {
		return stringToSentences(string);
	}

	public static String newlineSentences(String string) {
		string = string.replaceAll("\\.", "\\.\n");
		return string;
	}

	public static String additionalReplacements(String string) {
		string = string.replaceAll("haus ", " haus ");
		string = string.replaceAll(" haus", " haus ");
		string = string.replaceAll("wohnung ", " wohnung ");
		string = string.replaceAll("apartment ", " apartment ");
		string = string.replaceAll("garten ", " garten ");
		string = string.replaceAll("gebäude ", " gebäude ");
		string = string.replaceAll("zimmer ", " zimmer ");
		string = string.replaceAll("raum ", " raum ");
		return string;
	}

	public static String seperatePunctuation(String string) {
		string = string.replaceAll("!!", " ");
		string = string.replaceAll("(?<=\\S)(?:(?<=\\p{Punct})|(?=\\p{Punct}))(?=\\S)", " ");
		return string;
	}
	
}
