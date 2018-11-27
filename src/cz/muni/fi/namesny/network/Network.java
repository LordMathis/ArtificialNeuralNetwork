package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

import java.util.Arrays;

public class Network {

    private Layer[] layers;
    private IActivate activate;
    private double learningRate;

    private double[][][] batchResults;
    private double[][][] batchDeltas;

    public Network(int[] layerSizes,
                   IActivate activate,
                   double learningRate) {

        this.layers = new Layer[layerSizes.length - 1];
        this.activate = activate;
        this.learningRate = learningRate;


        int prevLayerSize = layerSizes[0];
        int layerSize;
        for (int i = 1; i < layerSizes.length; i++) {

            layerSize = layerSizes[i];
            this.getLayers()[i - 1] = new Layer(layerSize, prevLayerSize);
            prevLayerSize = layerSize;
        }
    }

    public double[] guess(double[] input) {
        double[] nextInputs = input;

        for (int i = 0; i < getLayers().length; i++) nextInputs = activate(getLayers()[i].compute(nextInputs));

        return nextInputs;
    }

    private double[] feedForward(double[] inputs, int k) {

        double[] nextInputs = inputs;
        double[] layerResult;

        for (int i = 0; i < getLayers().length; i++) {
            layerResult = getLayers()[i].compute(nextInputs);
            batchResults[k][i] = layerResult;
            nextInputs = activate(layerResult);
        }

        return nextInputs;
    }

    private void backPropagate(int k) {

        for (int i = getLayers().length - 2; i >= 0; i--) {

            batchDeltas[k][i] = MatrixMath.hadamard(
                    MatrixMath.multiply(
                            MatrixMath.transpose(getLayers()[i + 1].getWeights()),
                            batchDeltas[k][i + 1]),
                    activateDerivation(batchResults[k][i])
            );

        }
    }

    private double[] activate(double[] input) {

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = activate.getActivation(input[i]);
        }

        return result;
    }

    private double[] activateDerivation(double[] input) {

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = activate.getDerivative(input[i]);
        }

        return result;
    }

    public void batchTrain(double[][] inputBatch, double[][] actual) {

        this.batchResults = new double[inputBatch.length][getLayers().length][];
        this.batchDeltas = new double[inputBatch.length][getLayers().length][];

        for (int i = 0; i < inputBatch.length; i++) {

            double[] prediction = feedForward(inputBatch[i], i);
            double[] errorVector = costDerivative(prediction, actual[i]);
            System.out.println("Input: " + Arrays.toString(inputBatch[i]));
            System.out.println("Prediction: " + Arrays.toString(prediction));
            System.out.println("Error vector" + Arrays.toString(errorVector));
            double[] activationsDer = activateDerivation(prediction);

            batchDeltas[i][getLayers().length - 1] = MatrixMath.hadamard(errorVector, activationsDer);

            backPropagate(i);
        }

        gradientDescent(inputBatch);
    }

    private void gradientDescent(double[][] inputBatch) {

        for (int i = getLayers().length - 1; i >= 0; i--) {

            int[] dims = Utils.getDimensions(getLayers()[i].getWeights());

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
            double[][] newWeights = MatrixMath.sum(getLayers()[i].getWeights(), change);
            getLayers()[i].setWeights(newWeights);

            double[] changeBias = MatrixMath.multiply(-this.learningRate / (double) inputBatch.length, avgChangeBias);
            double[] newBias = MatrixMath.sum(getLayers()[i].getBias(), changeBias);
            getLayers()[i].setBias(newBias);

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

    public Layer[] getLayers() {
        return layers;
    }
}
