package networks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class BestParametersModel3 {
    
    public static void main(String[] args) throws IOException {

        double[][] trainData = DataUtilities.loadCSV("data\\networks\\classification_train.csv");
        double[][] testData = DataUtilities.loadCSV("data\\networks\\classification_test.csv");

        double[][] trainFeatures = DataUtilities.extractFeatures(trainData);
        double[][] trainLabels = DataUtilities.extractLabels(trainData);
        double[][] testFeatures = DataUtilities.extractFeatures(testData);
        double[][] testLabels = DataUtilities.extractLabels(testData);

        MLP3 model = new MLP3(0.1, 16, 16, 8, "tanh", "tanh", "tanh");
        model.train(trainFeatures, trainLabels, 200, 0.1, 1);
        
        evaluateModel(model, testFeatures, testLabels, "output\\networks\\model3_predictions.csv");
    }

    private static double evaluateModel(MLP3 mlp, double[][] features, double[][] labels, String filepath) throws IOException {
        int correct = 0;
        File file = new File(filepath);
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath, true));

        if (!file.exists()) {
            bw.write("x1,x2,correct\n");
        }

        for (int i = 0; i < features.length; i++) {
            double[] prediction = mlp.forward(features[i]);
            int predictedClass = argMax(prediction);
            int trueClass = argMax(labels[i]);
            if (predictedClass == trueClass) {
                correct++;
                DataUtilities.storeDatapoint(bw, features[i], 1);
            } else {
                DataUtilities.storeDatapoint(bw, features[i], 0);
            }
        }
        double accuracy = (double) correct / features.length;
        System.out.printf("Accuracy at test set: %.2f%%%n", accuracy * 100);
                
        bw.close();
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

