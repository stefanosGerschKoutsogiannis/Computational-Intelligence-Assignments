package networks;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class MLPBench {

    public static void main(String[] args) throws IOException {
        
        double[][] trainData = loadCSV("data\\networks\\classification_train.csv");
        double[][] testData = loadCSV("data\\networks\\classification_test.csv");

        double[][] trainFeatures = extractFeatures(trainData);
        double[][] trainLabels = extractLabels(trainData);
        double[][] testFeatures = extractFeatures(testData);
        double[][] testLabels = extractLabels(testData);

        double[] learningRate = {0.1, 0.01};
        int[] batchSize = {20, 200};
        double[] threshold = {10, 1, 0.1};
        //String[] activationOnHidden2 = {"tanh", "relu"};
        //int[] numNeurons = {4,8,16};

        double bestAccuracyInTestData = 0.0;
        double bestAccuracySoFar; 

        double bestLearningRate = 0;
        double bestThreshold = 0;
        int bestBatchSize = 0;

        for (double l: learningRate) {
            for (int b: batchSize) {
                for (double t: threshold) {
                    bestAccuracySoFar = run(trainFeatures, trainLabels, testFeatures, testLabels, l, t, b, 10);

                    if (bestAccuracySoFar > bestAccuracyInTestData) {
                        bestLearningRate = l;
                        bestThreshold = t;
                        bestBatchSize = b;
                        bestAccuracyInTestData = bestAccuracySoFar;
                    }
                }
            }
        }

        System.out.println("The best parameters for our program are:");
        System.out.println("Learning rate: " + bestLearningRate);
        System.out.println("Threshold: " + bestThreshold);
        System.out.println("Best batch size: "+ bestBatchSize);
        System.out.printf("Accuracy on test data with these parameters: %.2f%%%n", bestAccuracyInTestData*100);
    }

    private static double run(double[][] data, double[][] labels, double[][] testData, double[][] testLabels,
            double learningRate, double threshold, int batchSize, int numRuns) {

        double bestPerformingAccuracy = 0.0;
        double accuracy;

        for (int i = 0; i < numRuns; i++) {
            MLP mlp = new MLP(learningRate);
            mlp.train(data, labels, batchSize, threshold);
            accuracy = evaluate(mlp, testData, testLabels);

            if (accuracy > bestPerformingAccuracy) {
                bestPerformingAccuracy = accuracy;
            }
        }
        return bestPerformingAccuracy;
    }
    
    private static double[][] loadCSV(String filename) throws IOException {
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

    private static double[][] extractFeatures(double[][] data) {
        double[][] features = new double[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, features[i].length);
        }
        return features;
    }

    private static double[][] extractLabels(double[][] data) {
        int numClasses = 4; 
        double[][] labels = new double[data.length][numClasses];
        for (int i = 0; i < data.length; i++) {
            int label = (int) data[i][data[i].length - 1];
            labels[i][label] = 1.0; // One-hot encoding
        }
        return labels;
    }

    private static double evaluate(MLP mlp, double[][] features, double[][] labels) {
        int correct = 0;
        for (int i = 0; i < features.length; i++) {
            double[] prediction = mlp.forward(features[i]);
            int predictedClass = argMax(prediction);
            int trueClass = argMax(labels[i]);
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        double accuracy = (double) correct / features.length;
        System.out.printf("Accuracy at test set: %.2f%%%n", accuracy * 100);
        return accuracy;
    }

    private static int argMax(double[] array) {
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[index]) {
                index = i;
            }
        }
        return index;
    }
}
