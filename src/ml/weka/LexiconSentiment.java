import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

import com.opencsv.CSVReader;

import javax.json.Json;
import javax.json.JsonNumber;
import javax.json.JsonReader;
import javax.json.JsonStructure;
import javax.json.JsonObject;
import javax.json.JsonValue;

import weka.core.*;
import weka.core.converters.*;
import weka.core.stemmers.SnowballStemmer;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.*;
import weka.core.tokenizers.NGramTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;

public class LexiconSentiment {
	private Set<String> negativeLexicon = null;
	private Set<String> positiveLexicon = null;
	private Set<String> stopwords = null;
	private Vector<String> testLines = null;
	double sentimentHour[] = new double[24];
	double sentimentDay[] = new double[7];
	    double topicDetectionCorrect=0;
		String topicDetected="";
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
			// TODO Auto-generated method stub
			if(args.length != 4){
				System.err.println("example command: java -cp workspace path-to-jars:path-to-workspace " +
//						"lexicon-pos-file " +
//						"lexicon-neg-file " +
						"stopwords "		+
						"training-file "	+
						"tweet-location\n"	+
						"testing-file");
				return ;
			}
		// set preprocess to false to save time in preprocessing
		boolean preprocess = true;
		boolean removestopword = false;
		// available topics - apple, google, microsoft, twitter, all
		// set to all to train/test for all topics. Try train for domain specific (e.g. twitter for twitter) 
		// and all then test on specific topic. Record the difference in result in the report
		String trainTopic = "all";
		String testTopic = "all";
		boolean useExtra = true;
		boolean useHour = true;
		boolean useDay = true;
		long startTrainTime =0; 
		long endTrainTime=0;
		long startTestTime =0; 
		long endTestTime=0;
		LexiconSentiment sentimenter = new LexiconSentiment(args[0], removestopword);
		
		if (preprocess == true) {
			System.out.println("Preprocessing start...");
		//	sentimenter.preprocessTweets(args[1], args[2], "wekaDataDir");
			sentimenter.preprocessTweets(args[3], args[2], "wekaTestDir");
			System.out.println("Preprocessing end.");
		}
		
		System.out.println("Loading training tweets text.");
		TextDirectoryLoader loader = new TextDirectoryLoader();
		loader.setOutputFilename(true);		
	    loader.setDirectory(new File("wekaDataDir" + File.separator + trainTopic));
	    Instances dataRaw = loader.getDataSet();
	    
	    System.out.println("Generating training tweet list.");
	    // Get list of all tweets accessed in order
	    Vector<String> tweetList = getListOfTweets(dataRaw);
	    
	    // Remove filenames from data
	    dataRaw.deleteAttributeAt(1);
	    
	    // Set tokenizer to uni & bigrams
	    // Play around with this, set it to 1,1 for unigram.  2,2 bigrams. 3,3 for trigram
	    // Write the different results and compare in the report.
	    NGramTokenizer bigram = new NGramTokenizer();
	    bigram.setNGramMinSize(1);
	    bigram.setNGramMaxSize(2);
	    
	    System.out.println("Converting training tweets to vector.");
	    // Convert dataset to vector
	    StringToWordVector filter = new StringToWordVector();
	    filter.setMinTermFreq(1);
	    filter.setTokenizer(bigram);
	    filter.setInputFormat(dataRaw);
	    Instances dataFiltered = Filter.useFilter(dataRaw, filter);
	    
	    // Set use extra to true if using extra attributes
	    if(useExtra == true) {
	    	System.out.println("Calculating extra training attributes.");
	    	// first boolean use hour, 2nd boolean use day
		    Instances extra = sentimenter.extractAdditionalAttributes(tweetList, args[2], args[1], useHour, useDay, false);
		    if(extra != null) {
		    	dataFiltered = Instances.mergeInstances(dataFiltered, extra);
		    }
	    }
	    
	    PrintWriter ou = new PrintWriter("filtered.txt");
	    ou.println("\n\nFiltered data:\n\n" + dataFiltered);
	    ou.close();
	    //System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
	
	    System.out.println("Training classifier...");
	    // train NB and output model
	    dataFiltered.setClassIndex(0);
	    startTrainTime=System.currentTimeMillis();
	    //construct various of classifier
	    //  NaiveBayes classifier = new NaiveBayes();
	  //Classifier classifier = new IBk(3);
	   
	    //	RandomForest classifier = new RandomForest();
     	 LibSVM classifier = new LibSVM();
   	 classifier.setCacheSize(512); // MB
   	 classifier.setNormalize(true);
   	 classifier.setShrinking(true);
   	 classifier.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
   	 classifier.setDegree(3);
   	 classifier.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
   	classifier.buildClassifier(dataFiltered);
//	    Vote vote =new Vote();
//	    Classifier[] classifier={new NaiveBayes(), new RandomForest(), new IBk()};
//	    vote.setClassifiers(classifier);
//	    vote.buildClassifier(dataFiltered);
//	    Stacking stacking=new Stacking();
//	    stacking.setMetaClassifier(new RandomForest());
//	    Classifier[] classifier={new IBk()};
//	    stacking.setClassifiers(classifier);
//	    stacking.buildClassifier(dataFiltered);
//	    Bagging bagging=new Bagging();
//	    bagging.setClassifier(new LibSVM());
//	    bagging.setNumIterations(10);
//	    bagging.buildClassifier(dataFiltered);
	    endTrainTime=System.currentTimeMillis();
	 //   System.out.println("\n\nClassifier model:\n\n" + classifier);
	    PrintWriter out = new PrintWriter("outputtest.txt");
	 //   out.println("\n\nClassifier model:\n\n" + classifier);
	    out.close();
	    System.out.println("Done training classifier.");
	    
	    // Test trained model
	    // Load test set
	    System.out.println("Loading test tweets text.");
	    TextDirectoryLoader testloader = new TextDirectoryLoader();
	    testloader.setOutputFilename(true);
	    testloader.setDirectory(new File("wekaTestDir" + File.separator + testTopic));
	    Instances testdataRaw = testloader.getDataSet();
	    
	    System.out.println("Generating test tweet list.");
	    Vector<String> tweetListTest = getListOfTweets(testdataRaw);
	    
	 // Remove filenames from data
	    testdataRaw.deleteAttributeAt(1);
	    
	    // Pass it through the filter
	    System.out.println("Converting test tweets to vector based on training tweets.");
	    Instances testdataFiltered = Filter.useFilter(testdataRaw, filter);
	    
	    if(useExtra == true) {
	    	System.out.println("Calculating extra test attributes.");
	    	// first boolean use hour, 2nd boolean use day
		    Instances extraTest = sentimenter.extractAdditionalAttributes(tweetListTest, args[2], args[3], useHour, useDay, true);
		    if(extraTest != null) {
		    	testdataFiltered = Instances.mergeInstances(testdataFiltered, extraTest);
		    }
	    }
	    
	    PrintWriter te = new PrintWriter("filteredtest.txt");
	    te.println("\n\nFiltered data:\n\n" + testdataFiltered);
	    te.close();
	    
	    System.out.println("Testing classifier...");
	    // Evaluate trained classifier
	    testdataFiltered.setClassIndex(0);
	    startTestTime=System.currentTimeMillis();
	    Evaluation eTest = new Evaluation(testdataFiltered);
	    eTest.evaluateModel(classifier, testdataFiltered);
	    endTestTime=System.currentTimeMillis();
	    System.out.println("Done testing classifier.");
	    System.out.println("TrainTime is:"+(endTrainTime-startTrainTime)/1000.0+"s");
	    System.out.println("TestTime is:"+(endTestTime-startTestTime)/1000.0+"s");
	    System.out.println ("weighted avg precision is:"+eTest.weightedPrecision());
	    System.out.println ("weighted avg recall is:"+eTest.weightedRecall());
	    System.out.println ("category prediction is:"+sentimenter.topicDetectionCorrect+"/"+ testdataFiltered.numInstances()+"="+sentimenter.topicDetectionCorrect/testdataFiltered.numInstances());
	    
	    String strSummary = eTest.toClassDetailsString()+"\n"+eTest.toSummaryString();
	    PrintWriter res = new PrintWriter("result.txt");
	    res.println(strSummary);
	    res.close();
	    
	    System.out.println("\n" + strSummary);
	    if(!useExtra)
	    {
	     double successCount=0; //if prediction is correct
	   	 double successRelevantCount=0; //if prediction is correct
	   	 double precisionCount=0; //count of prediction not irrelevant (pos neg neutral of prediction) precision=successCount/precisionCount
	   	 double recallCount=0; //count of acutal relevant count (pos neg neutral of actual) precision=successCount/recallCount
	   	 int totalCount=testdataFiltered.numInstances();
	   	 for (int i = 0; i < testdataFiltered.numInstances(); i++) {
	   		   double pred = classifier.classifyInstance(testdataFiltered.instance(i));
	   		   String actualClass=testdataFiltered.classAttribute().value((int) testdataFiltered.instance(i).classValue());
	   		   String predictedClass=testdataFiltered.classAttribute().value((int) pred);
	   		   if(!actualClass.equals("irrelevant"))
	   		   {
	   			   recallCount++;
	   		   }
	   		   if(!predictedClass.equals("irrelevant"))
	   		   {
	   			   precisionCount++;
	   		   }
	   		   if(actualClass.equals(predictedClass))
	   		   {
	   			   successCount++;
	   			   if(!predictedClass.equals("irrelevant"))
	   			   {
	   				   successRelevantCount++;
	   			   }
	   		   }
	   		 //  System.out.print("ID: " + testdataFiltered.instance(i).value(0));
	   		   System.out.print("actual: " + actualClass);
	   		   System.out.println(", predicted: " + predictedClass);
	   		 }
	   	 System.out.println("accuracy count/Total Count "+successCount+"/"+totalCount);
	   	 System.out.println("accuracy: "+successCount/totalCount);
	   	 System.out.println("precision: "+successRelevantCount/precisionCount);// how many selected items are relevant?
	   	 System.out.println("recall: "+successRelevantCount/recallCount); // how many relevant items are selected?
	    }
  	}
	
	// For Enhanced Classifiers requirement
	Instances extractAdditionalAttributes(Vector<String> tweetList, String tweetDir, String csvFName, boolean useHour, boolean useDay, boolean isTest) throws IOException {
		FastVector attriList = new FastVector();
		Attribute sentHour = new Attribute("sentimentHour");
		Attribute sentDay = new Attribute("sentimentDay");
		if (useHour = true) {
			attriList.addElement(sentHour);
		}
		if (useHour = true) {
			attriList.addElement(sentDay);
		}
		
		Instances extras = new Instances("extraAttribute", attriList, tweetList.size());
		
		TreeMap<String, Integer> userTweetCount = new TreeMap<String, Integer>();
		TreeMap<String, String> tweetPolarity = new TreeMap<String, String>();
		
		//System.out.println(csvFName);
		CSVReader reader = new CSVReader(new FileReader(csvFName));
		String[] tokens;
		
		int index = 0;
		while((tokens = reader.readNext()) != null){
			if(tokens.length != 3){
				System.err.println("unexpect csv element of length 3 with index: " + index);
				reader.close();
			}
			if(index != 0) {
				tweetPolarity.put(tokens[2], tokens[1]);
			}
			index++;
		}
		reader.close();
		//System.out.println(tweetPolarity);
		
		int totalTweets = tweetList.size();
		
		Vector<Vector<Integer>> extraData = new Vector<Vector<Integer>>();
		
		int countHour[] = new int[24];
		int countDay[] = new int[7];
		
		for(int i = 0; i < tweetList.size(); i++) {
			
			// Read in tweet;
			String tweetID = tweetList.get(i);
			JsonReader jReader = Json.createReader(new FileReader(tweetDir + "\\" + tweetID + ".json"));
			JsonStructure js = jReader.read();
			JsonObject object = (JsonObject) js;

			// Extract temporal information
			
			// Author ID
			String userid = object.getJsonObject("user").getString("id_str");
			//System.out.println("user id: " + userid);
			
			// Find the author sentiment ( How positive/negative is the author ) [ Only for author with high tweet counts ]
			/*
			String userDir = tweetDir.replace("tweets", "users");
			JsonReader jReaderUsder = Json.createReader(new FileReader(userDir + "\\" + userid + ".json"));
			
			JsonStructure jsuser = jReaderUsder.read();
			JsonObject objectuser = (JsonObject) jsuser;
			
			Integer count = userTweetCount.get(userid);
			if(count != null) {
				userTweetCount.put(userid, count + 1);
				if(count + 1 > 2) {
					System.out.println(count+1);
				}
			} else {
				userTweetCount.put(userid, 1);
			}
			// Do something with the tweet counts, count the total sentiment of the authors post
			// Not in use, dataset too small to give reliable prediction with authors tweet
			//End of author ID
			*/
			
			// Time
			// Find time of the day
			String timestr = object.getString("created_at");
			//System.out.println(timestr);
			
			JsonObject user = object.getJsonObject("user");
			String zoneStr = user.get("utc_offset").toString();
			int timezone = 0;
			if(zoneStr == null || zoneStr.contentEquals("null")) {
				timezone = 0;
			} else {
				timezone = Integer.parseInt(zoneStr) / (60*60);
			}
		//	System.out.println(timezone);
			
			// Find local time
		//	System.out.println(timestr.substring(11, 13));
			int nextDay = 0;
			int localHour = (Integer.parseInt(timestr.substring(11, 13)) + timezone);
			String globalDay = timestr.substring(0,3);
			
			//System.out.println(globalDay);
			if(localHour < 0) {
				localHour = 24 + localHour;
				nextDay  = -1;
			}
			if(localHour >= 24 ) {
				localHour = localHour - 24;
				nextDay = 1;
			}
		//	System.out.println(localHour);
			String polarity = tweetPolarity.get(tweetID);
			double increment = 0;
			if(polarity != null) {
				if (polarity.contentEquals("positive")) {
					increment = 1;
				} else if (polarity.contentEquals("negative")) {
					increment = -1;
				} else {
					increment = 0;
				}
			} else {
				//System.out.println("Error, cannot find tweet! " + tweetID);
			}
			
			int localDay = getDayID(globalDay);
			if(localDay == -1) {
				System.out.println("Error, day code can't be found.");
				return null;
			} else {
				localDay += nextDay;
				if(localDay >= 7) {
					localDay = 0;
				}
				if(localDay < 0) {
					localDay = 6;
				}
			}
			
			// If not a test, store the sentiment
			if(!isTest) {
				sentimentHour[localHour] += increment;
				countHour[localHour]++;
				sentimentDay[localDay] += increment;
				countDay[localDay]++;
			}
			
			Vector<Integer> tmp = new Vector<Integer>();
			tmp.add(localHour);
			tmp.add(localDay);
			extraData.add(tmp);
		}
		
		// Check if isTest, if it is not process the sentiments into the bins
		if (!isTest) {
			//System.out.println(userTweetCount.toString());
			for (int i = 0; i < 24; i++) {
				if(countHour[i] != 0) {
					sentimentHour[i] /= (double)countHour[i];
				}
				//System.out.print(sentimentHour[i]+ " ");
			}
			//System.out.println();
			for (int i = 0; i < 7; i++) {
				if(countDay[i] != 0) {
					sentimentDay[i] /= (double)countDay[i];
				}
				//System.out.print(sentimentDay[i] + " ");
			}
		}

		for(int i = 0; i < tweetList.size(); i++) {
			Instance inst = new Instance(2);
			if (useHour = true) {
				inst.setValue(sentHour, sentimentHour[extraData.get(i).get(0)]);
			}
			if (useDay = true) {
				inst.setValue(sentDay, sentimentDay[extraData.get(i).get(1)]);
			}
			extras.add(inst);
		}
		
		//System.out.println(extras);
		return extras;
		
	}
	
	static int getDayID(String day) {
		if (day.contentEquals("Mon")) {
			return 0;
		}
		if (day.contentEquals("Tue")) {
			return 1;
		}
		if (day.contentEquals("Wed")) {
			return 2;
		}
		if (day.contentEquals("Thu")) {
			return 3;
		}
		if (day.contentEquals("Fri")) {
			return 4;
		}
		if (day.contentEquals("Sat")) {
			return 5;
		}
		if (day.contentEquals("Sun")) {
			return 6;
		}
		return -1;
	}
	
	// Get the list of tweets used in the arff, it is in order. Use this to against the result to see which classify correctly or not.
	static Vector<String> getListOfTweets(Instances data) {
		
		Vector<String> tweetList = new Vector<String>();
		
		for(int i = 0; i < data.numInstances(); i++) {
			//System.out.println(data.instance(i).stringValue(1));
			String str = data.instance(i).stringValue(1);
			str = str.replaceAll("[^\\d]", "");
			//System.out.println(str);
			tweetList.add(str);
		}
		
		tweetList.get(0);
		
		return tweetList;
	}
	
	public boolean preprocessTweets(String trainFname, String tweetDir, String rootDir) throws IOException{
		@SuppressWarnings("resource")
		CSVReader reader = new CSVReader(new FileReader(trainFname));
		String[] tokens;
		//csv format

		int index = 0;
		while((tokens = reader.readNext()) != null){
			if(tokens.length != 3){
				System.err.println("unexpect csv element of length 3 with index: " + index);
				reader.close();
				return false;
			}
			if(index != 0) {
			//	System.out.println(tokens[0] + " " + tokens[1] + " " + tokens[2]);				
				processTweetJson(tokens[0], tokens[1], tokens[2], tweetDir, rootDir);
			}
			index++;
		}
		reader.close();
		return true;
	}
	
//	public boolean isStopword(String word) {
//		return stopwords.contains(word);
//	}
	
	public void processForWeka(String topic, String sentiment, String tweetID, String text , String rootDir) throws IOException {
	
		BufferedWriter output = null;
		BufferedWriter outputall = null;
		boolean all = true;
	    try {
	    	
	    	File directory = new File(rootDir + File.separator + topic + File.separator + sentiment);
	    	directory.mkdirs();
	        File file = new File(rootDir + File.separator + topic + File.separator + sentiment + File.separator + tweetID + ".txt");
	        output = new BufferedWriter(new FileWriter(file));
	        output.write(text);
	        
	        if(all) {
	        	File directoryall = new File(rootDir + File.separator + "all" + File.separator + sentiment);
	        	directoryall.mkdirs();
		        File fileall = new File(rootDir + File.separator + "all" + File.separator + sentiment + File.separator + tweetID + ".txt");
		        outputall = new BufferedWriter(new FileWriter(fileall));
		        outputall.write(text);
	        }
	        
	    } catch ( IOException e ) {
	        e.printStackTrace();
	    } finally {
	        if ( output != null ) output.close();
	        if ( outputall != null ) outputall.close();
	    }
	}
	
	public boolean processTweetJson(String topic, String sentiment, String tweetID, String tweetDir, String rootDir) throws IOException {
		
		JsonReader jReader = Json.createReader(new FileReader(tweetDir + "\\" + tweetID + ".json"));
		JsonStructure js = jReader.read();
		
		JsonObject object = (JsonObject) js;
	//	System.out.println(topic + " " + sentiment + " ");
		
	//	System.out.println(object.getString("text"));
		
		String text = object.getString("text").toLowerCase();
		StringReader sr = new StringReader(text);
		if(text.contains("#apple")||text.contains("@apple"))
		{
			topicDetected="apple";
		}
		else if(text.contains("#google")||text.contains("@google"))
		{
				topicDetected="google";
		}
		else if(text.contains("#microsoft")||text.contains("@microsoft"))
		{
			topicDetected="microsoft";
		}
		else if(text.contains("#twitter")||text.contains("@twitter"))
		{
				topicDetected="twitter";
		}
		if(topicDetected.equals(topic))
		{
			topicDetectionCorrect=topicDetectionCorrect+1;
		}
		topic=topicDetected;
		PTBTokenizer ptbt = new PTBTokenizer(sr,  new CoreLabelTokenFactory(), "");
		
		String processed = "";
		
		for (CoreLabel label; ptbt.hasNext(); ) {
			label = (CoreLabel) ptbt.next();
			String token = label.toString();
			
				if(stopwords.contains(token))
				{
					continue;
				}
			
		    if(token.contains("http:")) {
				token = "URL";
			} else if(token.startsWith("@")) {
				token = "USERNAME";
			} else if(token.startsWith("#")) {
				continue;
			}
			
			token = token.replaceAll("[^A-Za-z]", "");	//Removes special characters
			
			
			
			if (token.trim().length() > 0) {
				processed += token  + " ";
			}
		}
		
		processed = processed.trim();
		
		//System.out.println(processed);
		processForWeka(topic, sentiment, tweetID, processed, rootDir);
		
		return true;
	}
	
	public static boolean isAllUpperCase(String str){
		for(int i=0; i<str.length(); i++){
			char c = str.charAt(i);
			if(!Character.isUpperCase(c)) {
				return false;
			}
		}
		//str.charAt(index)
		return true;
	}

	public LexiconSentiment(String stopFname, boolean remvoeStopwords) throws IOException {
		// TODO Auto-generated constructor stub
//		negativeLexicon = new TreeSet<String>();
//		initLexicon(negLexFname, negativeLexicon);
//		positiveLexicon = new TreeSet<String>();
//		initLexicon(posLexFname, positiveLexicon);
		stopwords = new TreeSet<String>();
		if(remvoeStopwords)
		{
		initLexicon(stopFname, stopwords);
		}
	}
	
	private void initLexicon(String fname, Set<String> lexicon) throws IOException{
		Vector<String> lines = new Vector<String>();
		LinesReader.readin(fname, lines);
		for(String line : lines){
			lexicon.add(line);
		}
	}
	private boolean readInTest(String fname, Vector<String> testTexts) throws IOException{
		CSVReader reader = new CSVReader(new FileReader(fname));
		String[] tokens;
		//csv format
		//index,press,date,url,title,description
		int index = 0;
		while((tokens = reader.readNext()) != null){
			if(tokens.length != 4){
				System.err.println("unexpect csv element with index: " + index);
				reader.close();
				return false;
			}
			if(index != 0)
				testTexts.add(tokens[3]);
			index++;
		}
		reader.close();
		return true;
	}
//	public void test() throws FileNotFoundException{
//		FileOutputStream fout = new FileOutputStream("sentiment.result");
//		PrintWriter p = new PrintWriter(fout);
//		for(String line : testLines){
//			int sentCount = 0;
//			String[] tokens = line.split(" ");
//			for(String token : tokens){
//				if(negativeLexicon.contains(token))
//					sentCount--;
//				if(positiveLexicon.contains(token))
//					sentCount++;
//			}
//			p.println(sentCount + " ||| " + line);
//		}
//		p.close();
//	}
}
