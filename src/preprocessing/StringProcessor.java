package preprocessing;

public class StringProcessor {

	static public String stringToSentences(String string) {
		string = seperatePunctuation(string);
		string = newlineSentences(string);
		return string;
	}

	public static String newlineSentences(String string) {
		string = string.replaceAll("\\.", "\\.\n");
		return string;
	}

	public static String seperatePunctuation(String string) {
		string = string.replaceAll("!!", " ");
		string = string.replaceAll("(?<=\\S)(?:(?<=\\p{Punct})|(?=\\p{Punct}))(?=\\S)", " ");
		return string;
	}
	
}
