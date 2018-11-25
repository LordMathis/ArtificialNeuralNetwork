package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

public class Network {

    private Layer[] layers;
    private ActivationFunction activationFunction;
    private ActivationDerivative activationDerivative;
    private double learningRate;

    private double[][][] batchResults;
    private double[][][] batchDeltas;

    public Network(int[] layerSizes,
                   ActivationFunction activationFunction,
                   ActivationDerivative activationDerivative,
                   double learningRate) {

        this.layers = new Layer[layerSizes.length - 1];
        this.activationFunction = activationFunction;
        this.activationDerivative = activationDerivative;
        this.learningRate = learningRate;


        int prevLayerSize = layerSizes[0];
        int layerSize;
        for (int i = 1; i < layerSizes.length; i++) {

            layerSize = layerSizes[i];
            this.layers[i - 1] = new Layer(layerSize, prevLayerSize);
            prevLayerSize = layerSize;
        }
    }

    public double[] feedForward(double[] inputs, int k) {

        double[] nextInputs = inputs;
        double[] layerResult;

        for (int i = 0; i < layers.length; i++) {
            layerResult = layers[i].compute(nextInputs);
            batchResults[k][i] = layerResult;
            nextInputs = activate(layerResult);
        }

        return nextInputs;
    }

//    public double[][] feedForward(double[][] inputs, int k) {
//
//        double[][] nextInputs = inputs;
//        double[][] layerResult;
//
//        for (int i = 0; i < layers.length; i++) {
//            Layer layer = layers[i];
//            layerResult = layer.compute(nextInputs);
//            batchResults[k][i] = layerResult;
//            nextInputs = activate(layerResult);
//        }
//
//        return nextInputs;
//    }

    public void backPropagate(int k) {

        for (int i = layers.length - 2; i >= 0; i--) {

            double[][] transposedWeights = MatrixMath.transpose(layers[i + 1].getWeights());
            double[] multiple = MatrixMath.multiply(transposedWeights, batchDeltas[k][i + 1]);
            double[] activateDerivation = activateDerivation(batchResults[k][i]);
            this.batchDeltas[k][i] = MatrixMath.hadamard(multiple, activateDerivation);
        }
    }

    public double[] activate(double[] input) {

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = activationFunction.compute(input[i]);
        }

        return result;
    }

    public double[] activateDerivation(double[] input) {

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = activationDerivative.compute(input[i]);
        }

        return result;
    }

//    public double[][] activate(double[][] input) {
//
//        int[] inputDims = Utils.getDimensions(input);
//        double[][] result = new double[inputDims[0]][inputDims[1]];
//
//        for (int i = 0; i < inputDims[0]; i++) {
//            for (int j = 0; j < inputDims[1]; j++) {
//                result[i][j] = activationFunction.compute(input[i][j]);
//            }
//        }
//
//        return result;
//    }

    public void batchTrain(double[][] inputBatch, double[][] actual) {

        this.batchResults = new double[inputBatch.length][layers.length][];
        this.batchDeltas = new double[inputBatch.length][layers.length][];

        for (int i = 0; i < inputBatch.length; i++) {

            double[] prediction = feedForward(inputBatch[i], i);
            double[] errorVector = costDerivative(prediction, actual[i]);
            double[] activationsDer = activateDerivation(prediction);

            batchDeltas[i][layers.length - 1] = MatrixMath.hadamard(errorVector, activationsDer);

            backPropagate(i);
        }

        gradientDescent(inputBatch);
    }

    public void gradientDescent(double[][] inputBatch) {

        for (int i = layers.length - 1; i >= 0; i--) {

            int[] dims = Utils.getDimensions(layers[i].getWeights());

            double[][] avgChange = Utils.initializeMatrix(dims[0], dims[1], false);
            double[] avgChangeBias = Utils.initializeVector(dims[0], null);

            for (int j = 0; j < inputBatch.length; j++) {

                double[] prevLayerResult;

                if (i == 0) {
                    prevLayerResult = activate(inputBatch[j]);
                } else {
                    prevLayerResult = activate(batchResults[j][i - 1]);
                }

                avgChange = MatrixMath.sum(avgChange, MatrixMath.multiply(batchDeltas[j][i], prevLayerResult));
                avgChangeBias = MatrixMath.sum(avgChangeBias, batchDeltas[j][i]);
            }

            double[][] change = MatrixMath.multiply(-this.learningRate / (double) inputBatch.length, avgChange);
            double[][] newWeights = MatrixMath.sum(layers[i].getWeights(), change);
            layers[i].setWeights(newWeights);

            double[] changeBias = MatrixMath.multiply(-this.learningRate / (double) inputBatch.length, avgChangeBias);
            double[] newBias = MatrixMath.sum(layers[i].getBias(), changeBias);
            layers[i].setBias(newBias);

        }
    }

    public double[] softmax(double[] input) {

        double totalInput = 0;

        for (double in : input) {
            totalInput += Math.exp(in);
        }

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i]) / totalInput;
        }

        return result;
    }

    public double quadraticCost(double[] predicted, double[] actual) {

        double cost = 0.0d;

        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        for (int i = 0; i < predicted.length; i++) {
            cost += Math.pow(actual[i] - predicted[i], 2);
        }

        return cost/2.0d;

    }

    public double[] costDerivative(double[] predicted, double[] actual) {
        return MatrixMath.substract(predicted, actual);
    }

}
