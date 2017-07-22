

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;


public class LinesReader {
	public static boolean readin(String inFname, Vector<String> lines) throws IOException{
		File file = new File(inFname);
		if(file.isFile() && file.exists()){
			InputStreamReader read = new InputStreamReader(new FileInputStream(file));
			BufferedReader bufferedReader = new BufferedReader(read);
			String lineTxt = null;
			while((lineTxt = bufferedReader.readLine()) != null)
				lines.add(lineTxt);
			read.close();
			return true;
		}else{
			System.err.println("can not find input file: " + inFname);
			return false;
		}
	}


}
