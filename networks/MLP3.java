package networks;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class MLP3 {

    private static final int INPUT_SIZE = 2; 
    private static final int OUTPUT_SIZE = 4; 
    private static final int HIDDEN1_SIZE = 16; 
    private static final int HIDDEN2_SIZE = 16; 
    private static final int HIDDEN3_SIZE = 16; 
    private static final String ACTIVATION_FUNCTION_H1_H2 = "tanh"; 
    private static final String ACTIVATION_FUNCTION_H3 = "tanh";

    private int inputSize, hidden1Size, hidden2Size, hidden3Size, outputSize;
    private double[][] weights1, weights2, weights3, weights4; 
    private double[] biases1, biases2, biases3, biases4;       
    private double learningRate;

    public MLP3(double learningRate) {
        this.inputSize = INPUT_SIZE;
        this.hidden1Size = HIDDEN1_SIZE;
        this.hidden2Size = HIDDEN2_SIZE;
        this.hidden3Size = HIDDEN3_SIZE; 
        this.outputSize = OUTPUT_SIZE;
        this.learningRate = learningRate;

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

    public double[] forward(double[] input) {
        double[] hidden1 = activate(layerOutput(weights1, input, biases1), ACTIVATION_FUNCTION_H1_H2);
        double[] hidden2 = activate(layerOutput(weights2, hidden1, biases2), ACTIVATION_FUNCTION_H1_H2);
        double[] hidden3 = activate(layerOutput(weights3, hidden2, biases3), ACTIVATION_FUNCTION_H3); // New layer
        double[] output = activate(layerOutput(weights4, hidden3, biases4), "relu");
        return output;
    }

    private double[] activate(double[] inputs, String activation) {
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            if (activation.equals("tanh")) {
                outputs[i] = tanh(inputs[i]);
            } else if (activation.equals("relu")) {
                outputs[i] = relu(inputs[i]);
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

        for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
            double[] input = batchInputs[sampleIdx];
            double[] target = batchTargets[sampleIdx];

            // Forward pass
            double[] hidden1output = layerOutput(weights1, input, biases1);
            double[] hidden1 = activate(hidden1output, ACTIVATION_FUNCTION_H1_H2);
            double[] hidden2output = layerOutput(weights2, hidden1, biases2);
            double[] hidden2 = activate(hidden2output,  ACTIVATION_FUNCTION_H1_H2);
            double[] hidden3output = layerOutput(weights3, hidden2, biases3);
            double[] hidden3 = activate(hidden3output, ACTIVATION_FUNCTION_H3); // New layer
            double[] outputTotal = layerOutput(weights4, hidden3, biases4);
            double[] output = activate(outputTotal, "relu");

            // Output error
            double[] outputErrors = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                outputErrors[i] = output[i] - target[i];
            }

            // Errors for hidden layers
            double[] hidden3Errors = new double[hidden3Size];
            double[] hidden2Errors = new double[hidden2Size];
            double[] hidden1Errors = new double[hidden1Size];

            for (int i = 0; i < hidden3Size; i++) {
                double derivative = ACTIVATION_FUNCTION_H3.equals("tanh") ? tanhDerivative(hidden3output[i]) : reluDerivative(hidden3output[i]);
                for (int j = 0; j < outputSize; j++) {
                    hidden3Errors[i] += weights4[j][i] * outputErrors[j];
                }
                hidden3Errors[i] *= derivative;
            }

            for (int i = 0; i < hidden2Size; i++) {
                double derivative = ACTIVATION_FUNCTION_H1_H2.equals("tanh") ? tanhDerivative(hidden2output[i]) : reluDerivative(hidden2output[i]);
                for (int j = 0; j < hidden3Size; j++) {
                    hidden2Errors[i] += weights3[j][i] * hidden3Errors[j];
                }
                hidden2Errors[i] *= derivative;
            }

            for (int i = 0; i < hidden1Size; i++) {
                double derivative = ACTIVATION_FUNCTION_H1_H2.equals("tanh") ? tanhDerivative(hidden1output[i]) : reluDerivative(hidden1output[i]);
                for (int j = 0; j < hidden2Size; j++) {
                    hidden1Errors[i] += weights2[j][i] * hidden2Errors[j];
                }
                hidden1Errors[i] *= derivative;
            }

            // Accumulate gradients
            accumulateGradients(weight4Gradients, bias4Gradients, hidden3, outputErrors);
            accumulateGradients(weight3Gradients, bias3Gradients, hidden2, hidden3Errors);
            accumulateGradients(weight2Gradients, bias2Gradients, hidden1, hidden2Errors);
            accumulateGradients(weight1Gradients, bias1Gradients, input, hidden1Errors);
        }

        // Update weights and biases
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

    public void train(double[][] data, double[][] labels, int batchSize, double threshold) {
        double previousEpochLoss = Double.MAX_VALUE; 
        int minEpochs = 800;
        int epoch = 1;
        while (true) {
            double epochLoss = 0.0;
            // for each batch
            for (int i = 0; i < data.length; i += batchSize) {
                int end = Math.min(i + batchSize, data.length);
                double[][] batchInputs = new double[end - i][];
                double[][] batchLabels = new double[end - i][];
                System.arraycopy(data, i, batchInputs, 0, end - i);
                System.arraycopy(labels, i, batchLabels, 0, end - i);
    
                backprop(batchInputs, batchLabels);
    
                // calculate loss for this batch
                for (int j = i; j < end; j++) {
                    double[] output = forward(data[j]);
                    for (int k = 0; k < output.length; k++) {
                        epochLoss += Math.pow(output[k] - labels[j][k], 2); // MSE
                    }
                }
            }
    
            // mean loss for this epoch
            epochLoss /= 2;
    
            // Εprint epoch - loss
            System.out.printf("Epoch %d, Loss: %.6f%n", epoch, epochLoss);

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
            double[][] trainData = loadCSV("data\\networks\\classification_train.csv");
            double[][] testData = loadCSV("data\\networks\\classification_test.csv");
            double[][] trainFeatures = extractFeatures(trainData);
            double[][] trainLabels = extractLabels(trainData);
            double[][] testFeatures = extractFeatures(testData);
            double[][] testLabels = extractLabels(testData);

            MLP3 mlp3 = new MLP3(0.01);

            int batchSize = 20;
            double threshold = 0.1;
            mlp3.train(trainFeatures, trainLabels, batchSize, threshold);

            evaluate(mlp3, testFeatures, testLabels);
        } catch (IOException e) {
            System.err.println("Error while loading the data " + e.getMessage());
        }
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



