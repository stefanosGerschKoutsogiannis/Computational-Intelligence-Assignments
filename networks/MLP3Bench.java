package networks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;



public class MLP3Bench {

    private static final double LEARNING_RATE = 0.1;
    private static final double THRESHOLD = 0.1;
    private static final String FILEPATH = "output\\networks\\prediction_results_model3.csv";
    private static final int NUM_RUMS = 5;

    public static void main(String[] args) throws IOException {

        double[][] trainData = DataUtilities.loadCSV("data\\networks\\classification_train.csv");
        double[][] testData = DataUtilities.loadCSV("data\\networks\\classification_test.csv");

        double[][] trainFeatures = DataUtilities.extractFeatures(trainData);
        double[][] trainLabels = DataUtilities.extractLabels(trainData);
        double[][] testFeatures = DataUtilities.extractFeatures(testData);
        double[][] testLabels = DataUtilities.extractLabels(testData);

        int[] batchSize = { 20, 200 };
        int[] numNeurons = { 4, 8, 16 };
        String[] activation = { "tanh", "relu" };

        double bestAccuracyInTestData = 0.0;
        double bestAccuracySoFar;

        int bestBatchSize = 0;
        int bestNumberOfNeuronsH1 = 0;
        int bestNumberOfNeuronsH2 = 0;
        int bestNumberOfNeuronsH3 = 0;
        String bestActivationOnH3 = "";

        for (int batch : batchSize) {
            for (int neuronsH1 : numNeurons) {
                for (int neuronsH2: numNeurons) {
                    for (int neuronsH3: numNeurons) {
                        for (String activationH3 : activation) {
                            bestAccuracySoFar = run(trainFeatures, trainLabels, testFeatures, testLabels, LEARNING_RATE, THRESHOLD,
                            batch, neuronsH1, neuronsH2, neuronsH3, "tanh", "tanh", activationH3, NUM_RUMS);
        
                            if (bestAccuracySoFar > bestAccuracyInTestData) {
                                bestBatchSize = batch;
                                bestNumberOfNeuronsH1 = neuronsH1;
                                bestNumberOfNeuronsH2 = neuronsH2;
                                bestNumberOfNeuronsH3 = neuronsH3;
                                bestActivationOnH3 = activationH3;
                                bestAccuracyInTestData = bestAccuracySoFar;
                            }
                        }

                    }
                }
            }
        }

        System.out.println("============================================================");        
        System.out.println("The best parameters for our program are:");
        System.out.println("Best batch size: "+bestBatchSize);
        System.out.println("Best number of neurons on layer hidden layer 1: "+bestNumberOfNeuronsH1);
        System.out.println("Best number of neurons on layer hidden layer 2: "+bestNumberOfNeuronsH2);
        System.out.println("Best number of neurons on hidden layer 3: " + bestNumberOfNeuronsH3);
        System.out.println("Best activation functions: ");
        System.out.println("    Hidden layer 1: tanh");
        System.out.println("    Hidden layer 2: tanh");
        System.out.println("    Hidden layer 3: " + bestActivationOnH3);
        System.out.println("============================================================");
        System.out.printf("Accuracy on test data with these parameters: %.2f%%%n",bestAccuracyInTestData*100);
        System.out.println("Retraining a model with these parameters to store its predictions");
        MLP3 mlp = new MLP3(LEARNING_RATE, bestNumberOfNeuronsH1, bestNumberOfNeuronsH2, bestNumberOfNeuronsH3, "tanh", "tanh", bestActivationOnH3);
        mlp.train(trainFeatures, trainLabels, bestBatchSize, THRESHOLD, 0);
        double accuracyOfModel = evaluateModel(mlp, testFeatures, testLabels, FILEPATH);
        System.out.printf("Accuracy of model: %.2f%%%n", accuracyOfModel*100);
    }

    private static double run(double[][] data, double[][] labels, double[][] testData, double[][] testLabels,
            double learningRate, double threshold, int batchSize, int numNeuronsH1, int numNeuronsH2, int numNeuronsH3,
            String activationH1, String activationH2, String activationH3, int numRuns) throws IOException {

        double bestPerformingAccuracy = 0.0;
        double accuracy;

        for (int i = 0; i < numRuns; i++) {
            MLP3 mlp = new MLP3(learningRate, numNeuronsH1, numNeuronsH2, numNeuronsH3, activationH1, activationH2, activationH3);
            mlp.train(data, labels, batchSize, threshold, 1);
            accuracy = evaluate(mlp, testData, testLabels);

            if (accuracy > bestPerformingAccuracy) {
                bestPerformingAccuracy = accuracy;
            }
        }
        return bestPerformingAccuracy;
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

