package clustering;

import java.io.FileNotFoundException;
import java.io.IOException;

/*
 * This is to be run as a test to observe how it runs
 * 
 * Run it from vscode with the play button I have not figured out
 * yet the correct compile order
 */
public class Test {

    public static void main(String[] args) throws FileNotFoundException, IOException {

        Point[] dataset = DataUtilities.readCsvDataset("data\\clustering\\clustering_dataset.csv", 1000);
        Kmeans km = new Kmeans(10, dataset);


        km.runKMeans();

    }
}