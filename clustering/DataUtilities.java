package clustering;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
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
        bw.write("x1,x2,cluster_id,c_x1,c_x2\n");
        for (Cluster c: data) {
            for (Point p: c.points) {
                String point = p.x1 + "," +p.x2 + "," +c.clusterId+","+c.centroid.x1+","+c.centroid.x2;
                bw.write(point);
                bw.write('\n');
            }
        }
        bw.close();
    }



    public static void storeClusterTotalVariance(int numberOfClusters, double variance, String filepath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath, true));
        File file = new File(filepath);

        if (!file.exists()) {
            bw.write("number_of_clusters,total_variance\n");
        }

        
        bw.append(numberOfClusters+","+variance+"\n");
        bw.close();
    }
}
