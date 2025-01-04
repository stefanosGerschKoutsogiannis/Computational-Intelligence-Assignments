package clustering;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class DataUtilities {

    public static Point[] readCsvDataset(String filename, int size) throws FileNotFoundException, IOException  {
        Point[] output = new Point[size];
        Point p;
        int cnt = 0;
        try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                p = new Point(Double.parseDouble(values[0]), Double.parseDouble(values[1]));
                output[cnt] = p;
                cnt++;
            }
        }
        return output;
    }

    public static void storeData(Cluster[] data, String filepath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
        for (Cluster c: data) {
            for (Point p: c.points) {
                String point = p.x1 + "," +p.x2 + "," + c.clusterId + "," +c.centroid.x1+","+c.centroid.x2;
                bw.write(point);
                bw.write('\n');
            }
        }
        bw.close();
    }

    public static void storeClusterTotalVariance(int numberOfClusters, double variance, String filepath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
        bw.append(numberOfClusters+","+variance);
        bw.close();
    }
}
