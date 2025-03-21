package networks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class MLP3 {

    private static final int INPUT_SIZE = 2; 
    private static final int OUTPUT_SIZE = 4; 
    private static final int HIDDEN1_SIZE = 16; 
    private static final int HIDDEN2_SIZE = 16; 
    private static final int HIDDEN3_SIZE = 16; 
    private static final String ACTIVATION_FUNCTION_H1_H2 = "tanh"; 
    private static final String ACTIVATION_FUNCTION_H3 = "tanh";

    private static final String FILEPATH = "output\\networks\\prediction_results_3_layers.csv";

    private int inputSize, hidden1Size, hidden2Size, hidden3Size, outputSize;
    private double[][] weights1, weights2, weights3, weights4; 
    private double[] biases1, biases2, biases3, biases4;       
    private double learningRate;
    private String activationH1;
    private String activationH2;
    private String activationH3;

    public MLP3(double learningRate) {
        this.inputSize = INPUT_SIZE;
        this.hidden1Size = HIDDEN1_SIZE;
        this.hidden2Size = HIDDEN2_SIZE;
        this.hidden3Size = HIDDEN3_SIZE; 
        this.outputSize = OUTPUT_SIZE;
        this.learningRate = learningRate;
        this.activationH1 = ACTIVATION_FUNCTION_H1_H2;
        this.activationH2 = ACTIVATION_FUNCTION_H1_H2;
        this.activationH3 = ACTIVATION_FUNCTION_H3;

        // Initialize weights and biases
        this.weights1 = initializeWeights(hidden1Size, inputSize);
        this.biases1 = initializeBiases(hidden1Size);
        this.weights2 = initializeWeights(hidden2Size, hidden1Size);
        this.biases2 = initializeBiases(hidden2Size);
        this.weights3 = initializeWeights(hidden3Size, hidden2Size); 
        this.biases3 = initializeBiases(hidden3Size);
        this.weights4 = initializeWeights(outputSize, hidden3Size);  
        this.biases4 = initializeBiases(outputSize);
    }

    public MLP3(double learningRate, int hidden1Size, int hidden2Size, int hidden3Size,
            String activationH1, String activationH2, String activationH3) {
        this.inputSize = INPUT_SIZE;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.hidden3Size = hidden3Size; 
        this.outputSize = OUTPUT_SIZE;
        this.learningRate = learningRate;
        this.activationH1 = activationH1;
        this.activationH2 = activationH2;
        this.activationH3 = activationH3;

        // Initialize weights and biases
        this.weights1 = initializeWeights(hidden1Size, inputSize);
        this.biases1 = initializeBiases(hidden1Size);
        this.weights2 = initializeWeights(hidden2Size, hidden1Size);
        this.biases2 = initializeBiases(hidden2Size);
        this.weights3 = initializeWeights(hidden3Size, hidden2Size); 
        this.biases3 = initializeBiases(hidden3Size);
        this.weights4 = initializeWeights(outputSize, hidden3Size);  
        this.biases4 = initializeBiases(outputSize);
    }

    private double[][] initializeWeights(int rows, int cols) {
        Random random = new Random();
        double[][] weights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = random.nextDouble() * 2 - 1; // [-1, 1]
            }
        }
        return weights;
    }

    private double[] initializeBiases(int size) {
        Random random = new Random();
        double[] biases = new double[size];
        for (int i = 0; i < size; i++) {
            biases[i] = random.nextDouble() * 2 - 1; // [-1, 1]
        }
        return biases;
    }

    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double tanhDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    public static double sigmoidDerivative(double x) {
        double sigmoidValue = sigmoid(x);
        return sigmoidValue * (1.0 - sigmoidValue);
    }

    public double[] forward(double[] input) {
        double[] hidden1 = activate(layerOutput(weights1, input, biases1), this.activationH1);
        double[] hidden2 = activate(layerOutput(weights2, hidden1, biases2), this.activationH2);
        double[] hidden3 = activate(layerOutput(weights3, hidden2, biases3), this.activationH3); // New layer
        double[] output = activate(layerOutput(weights4, hidden3, biases4), "sigmoid");
        return output;
    }

    private double[] activate(double[] inputs, String activation) {
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            if (activation.equals("tanh")) {
                outputs[i] = tanh(inputs[i]);
            } else if (activation.equals("relu")) {
                outputs[i] = relu(inputs[i]);
            } else {
                outputs[i] = sigmoid(inputs[i]);
            }
        }
        return outputs;
    }

    private double[] layerOutput(double[][] matrix, double[] vector, double[] biases) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = biases[i];
            for (int j = 0; j < matrix[i].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    public void backprop(double[][] batchInputs, double[][] batchTargets) {
        int batchSize = batchInputs.length;

        double[][] weight1Gradients = new double[hidden1Size][inputSize];
        double[][] weight2Gradients = new double[hidden2Size][hidden1Size];
        double[][] weight3Gradients = new double[hidden3Size][hidden2Size];
        double[][] weight4Gradients = new double[outputSize][hidden3Size];
        double[] bias1Gradients = new double[hidden1Size];
        double[] bias2Gradients = new double[hidden2Size];
        double[] bias3Gradients = new double[hidden3Size];
        double[] bias4Gradients = new double[outputSize];

        for (int sample = 0; sample < batchSize; sample++) {
            double[] input = batchInputs[sample];
            double[] target = batchTargets[sample];

            // Forward pass
            double[] hidden1output = layerOutput(weights1, input, biases1);
            double[] hidden1 = activate(hidden1output, this.activationH1);
            double[] hidden2output = layerOutput(weights2, hidden1, biases2);
            double[] hidden2 = activate(hidden2output,  this.activationH2);
            double[] hidden3output = layerOutput(weights3, hidden2, biases3);
            double[] hidden3 = activate(hidden3output, this.activationH3); 
            double[] outputTotal = layerOutput(weights4, hidden3, biases4);
            double[] output = activate(outputTotal, "sigmoid");

            double[] outputErrors = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                outputErrors[i] = output[i] - target[i];
            }

            double[] hidden3Errors = new double[hidden3Size];
            double[] hidden2Errors = new double[hidden2Size];
            double[] hidden1Errors = new double[hidden1Size];

            for (int i = 0; i < hidden3Size; i++) {
                double derivative = this.activationH3.equals("tanh") ? tanhDerivative(hidden3output[i]) : reluDerivative(hidden3output[i]);
                for (int j = 0; j < outputSize; j++) {
                    hidden3Errors[i] += weights4[j][i] * outputErrors[j];
                }
                hidden3Errors[i] *= derivative;
            }

            for (int i = 0; i < hidden2Size; i++) {
                double derivative = this.activationH2.equals("tanh") ? tanhDerivative(hidden2output[i]) : reluDerivative(hidden2output[i]);
                for (int j = 0; j < hidden3Size; j++) {
                    hidden2Errors[i] += weights3[j][i] * hidden3Errors[j];
                }
                hidden2Errors[i] *= derivative;
            }

            for (int i = 0; i < hidden1Size; i++) {
                double derivative = this.activationH1.equals("tanh") ? tanhDerivative(hidden1output[i]) : reluDerivative(hidden1output[i]);
                for (int j = 0; j < hidden2Size; j++) {
                    hidden1Errors[i] += weights2[j][i] * hidden2Errors[j];
                }
                hidden1Errors[i] *= derivative;
            }

            accumulateGradients(weight4Gradients, bias4Gradients, hidden3, outputErrors);
            accumulateGradients(weight3Gradients, bias3Gradients, hidden2, hidden3Errors);
            accumulateGradients(weight2Gradients, bias2Gradients, hidden1, hidden2Errors);
            accumulateGradients(weight1Gradients, bias1Gradients, input, hidden1Errors);
        }

        updateWeights(weights4, biases4, weight4Gradients, bias4Gradients, batchSize);
        updateWeights(weights3, biases3, weight3Gradients, bias3Gradients, batchSize);
        updateWeights(weights2, biases2, weight2Gradients, bias2Gradients, batchSize);
        updateWeights(weights1, biases1, weight1Gradients, bias1Gradients, batchSize);
    }

    private void accumulateGradients(double[][] weightGradients, double[] biasGradients, double[] inputs, double[] errors) {
        for (int i = 0; i < errors.length; i++) {
            biasGradients[i] += errors[i];
            for (int j = 0; j < inputs.length; j++) {
                weightGradients[i][j] += errors[i] * inputs[j];
            }
        }
    }

    private void updateWeights(double[][] weights, double[] biases, double[][] weightGradients, double[] biasGradients, int batchSize) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * (biasGradients[i] / batchSize);
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * (weightGradients[i][j] / batchSize);
            }
        }
    }

    public void train(double[][] data, double[][] labels, int batchSize, double threshold, int show) {
        double previousEpochLoss = Double.MAX_VALUE; 
        int minEpochs = 800;
        int epoch = 1;
        while (true) {
            double epochLoss = 0.0;
            for (int i = 0; i < data.length; i += batchSize) {
                int end = Math.min(i + batchSize, data.length);
                double[][] batchInputs = new double[end - i][];
                double[][] batchLabels = new double[end - i][];
                System.arraycopy(data, i, batchInputs, 0, end - i);
                System.arraycopy(labels, i, batchLabels, 0, end - i);
    
                backprop(batchInputs, batchLabels);
    
                for (int j = i; j < end; j++) {
                    double[] output = forward(data[j]);
                    for (int k = 0; k < output.length; k++) {
                        epochLoss += Math.pow(output[k] - labels[j][k], 2);
                    }
                }
            }
    
            epochLoss /= 2;
    
            if (show == 1) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, epochLoss);
            }

            if (epoch >= minEpochs && Math.abs(previousEpochLoss - epochLoss) < threshold) {
                System.out.println("Training stopped early due to small loss difference.");
                break;
            }
    
            previousEpochLoss = epochLoss;
            epoch++;
        }
    }

    public static void main(String[] args) {
        try {
            double[][] trainData = DataUtilities.loadCSV("data\\networks\\classification_train.csv");
            double[][] testData = DataUtilities.loadCSV("data\\networks\\classification_test.csv");
            double[][] trainFeatures = DataUtilities.extractFeatures(trainData);
            double[][] trainLabels = DataUtilities.extractLabels(trainData);
            double[][] testFeatures = DataUtilities.extractFeatures(testData);
            double[][] testLabels = DataUtilities.extractLabels(testData);

            MLP3 mlp3 = new MLP3(0.01);

            int batchSize = 20;
            double threshold = 0.1;
            mlp3.train(trainFeatures, trainLabels, batchSize, threshold, 1);

            evaluate(mlp3, testFeatures, testLabels);
            evaluateModel(null, testFeatures, testLabels, FILEPATH);
        } catch (IOException e) {
            System.err.println("Error while loading the data " + e.getMessage());
        }
    }

    private static void evaluate(MLP3 mlp, double[][] features, double[][] labels) {
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
    }

    private static double evaluateModel(MLP mlp, double[][] features, double[][] labels, String filepath) throws IOException {
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



