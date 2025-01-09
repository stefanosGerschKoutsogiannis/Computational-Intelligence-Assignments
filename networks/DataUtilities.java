package networks;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataUtilities {

        public static double[][] loadCSV(String filename) throws IOException {
        ArrayList<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] values = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    values[i] = Double.parseDouble(parts[i]);
                }
                data.add(values);
            }
        }
        return data.toArray(new double[0][]);
    }

    public static double[][] extractFeatures(double[][] data) {
        double[][] features = new double[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, features[i].length);
        }
        return features;
    }

    public static double[][] extractLabels(double[][] data) {
        int numClasses = 4;
        double[][] labels = new double[data.length][numClasses];
        for (int i = 0; i < data.length; i++) {
            int label = (int) data[i][data[i].length - 1];
            labels[i][label] = 1.0; // One-hot encoding
        }
        return labels;
    }

    public static void storeDatapoint(BufferedWriter bw, double[] datapoint, int correct) throws IOException {
        bw.append(datapoint[0]+","+datapoint[1]+","+correct+"\n");
    }
    
}
