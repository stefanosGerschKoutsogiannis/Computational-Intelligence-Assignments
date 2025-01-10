package networks;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Model3Data {
    
    public static void main(String[] args) throws IOException {

        double[][] trainData = DataUtilities.loadCSV("data\\networks\\classification_train.csv");
        double[][] testData = DataUtilities.loadCSV("data\\networks\\classification_test.csv");

        double[][] trainFeatures = DataUtilities.extractFeatures(trainData);
        double[][] trainLabels = DataUtilities.extractLabels(trainData);
        double[][] testFeatures = DataUtilities.extractFeatures(testData);
        double[][] testLabels = DataUtilities.extractLabels(testData);

        double LEARNING_RATE = 0.1;
        double THRESHOLD = 0.1;

        int[] batchSize = { 20, 200 };
        int[] numNeurons = { 4, 8, 16 };
        String[] activation = { "tanh", "relu" };

        String FILENAME = "output\\networks\\model3_parameters_performance.csv";

        MLP3 model;
        double modelAccuracy;
        BufferedWriter bw = new BufferedWriter(new FileWriter(FILENAME, true));

        for (int batch: batchSize) {
            for (int neuronsH1: numNeurons) {
                for (int neuronsH2: numNeurons) {
                    for (int neurons_H3: numNeurons) {
                        for (String activationH3: activation) {
                            model = new MLP3(LEARNING_RATE, neuronsH1, neuronsH2, neurons_H3, "tanh", "tanh", activationH3);
                            model.train(trainFeatures, trainLabels, batch, THRESHOLD, 1);
                            modelAccuracy = evaluate(model, testFeatures, testLabels);
                            DataUtilities.storeModel3Parameters(bw, LEARNING_RATE, batch, neuronsH1, neuronsH2, neurons_H3, "tanh", "tanh", activationH3, modelAccuracy, FILENAME);
                        }

                    }
                }
            }
        }
        bw.close();
    }

    private static double evaluate(MLP3 mlp, double[][] features, double[][] labels) {
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

