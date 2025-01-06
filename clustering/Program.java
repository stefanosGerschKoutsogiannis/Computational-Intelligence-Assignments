package clustering;

import java.io.FileNotFoundException;
import java.io.IOException;


public class Program {
    
    public static void run(int numRuns, int numClusters, String inputFilename,
                String clusterPointDataFileName, int numOfPointsInFile, String varianceFile) throws FileNotFoundException, IOException {

        Point[] dataset = DataUtilities.readCsvDataset(inputFilename, numOfPointsInFile);
        double bestRun = Double.MAX_VALUE; 
        Cluster[] bestClusters = null;
        Cluster[] currentRun = null;

        for (int i = 0; i < numRuns; i++) {
            Kmeans km = new Kmeans(numClusters, dataset);
            currentRun = km.runKMeans();
            double currentRunVariance = km.calculateTotalVariance(currentRun);

            if (currentRunVariance < bestRun) {
                bestRun = currentRunVariance;
                bestClusters = currentRun;
            }
        }
        DataUtilities.storeData(bestClusters, clusterPointDataFileName);
        DataUtilities.storeClusterTotalVariance(numClusters, bestRun, varianceFile);
        
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {

        int NUM_RUNS = 20;
        int NUM_OF_POINTS = 1000;
        String FILENAME = "data\\clustering\\clustering_dataset.csv";
        String[] outputFileNames = {
            "best_4_clusters.csv",
            "best_6_clusters.csv",
            "best_8_clusters.csv",
            "best_10_clusters.csv",
            "best_12_clusters.csv"
        };

        String VARIANCE_FILENAME = "clusters_variance.csv";
        int[] numberOfClusters = {4,6,8,10,12};
        for (int i = 0; i < numberOfClusters.length; i++) {
            run(NUM_RUNS, numberOfClusters[i], FILENAME, "output\\clustering\\"+outputFileNames[i], NUM_OF_POINTS, "output\\clustering\\"+VARIANCE_FILENAME);
        }
    }
}


