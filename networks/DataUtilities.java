package networks;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
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
    
    public static BufferedWriter createFile(String filepath) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath, true));
        bw.write("learning_rate,batch_size,neurons_H1,neurons_h2,activation_H1,activation_H2,accuracy\n");
        return bw;
    }

    public static void storeModel2Parameters(BufferedWriter bw, double learningRate, int batchSize, int numHeuronsH1, int numHeuronsH2,
        String activationH1, String activationH2, double accuracy, String filepath) throws IOException {
            bw.append(learningRate+","+batchSize+","+numHeuronsH1+","+numHeuronsH2+","+activationH1+","+activationH2+","+accuracy+"\n");
    }

    public static void storeModel3Parameters(BufferedWriter bw, double learningRate, int batchSize, int numHeuronsH1, int numHeuronsH2, int numHeuronsH3,
        String activationH1, String activationH2, String activationH3, double accuracy, String filepath) throws IOException {
        bw.append(learningRate+","+batchSize+","+numHeuronsH1+","+numHeuronsH2+","+numHeuronsH3+","+activationH1+","+activationH2+","+activationH3+","+accuracy+"\n");
    }
}
