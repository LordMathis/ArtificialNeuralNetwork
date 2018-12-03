package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

public class Network {

    private Layer[] layers;
    private IActivate activate;
    private double learningRate;
    private int networkLength;

    private double[][] layerResults;
    private double[][] layerActivations;
    private double[][][] deltaWeights;
    private double[][] deltaBiases;

    public Network(int[] layerSizes,
                   IActivate activate,
                   double learningRate) {

        this.layers = new Layer[layerSizes.length - 1];
        this.networkLength = this.layers.length;
        this.activate = activate;
        this.learningRate = learningRate;

        for (int i = 1; i < layerSizes.length; i++) {
            this.layers[i - 1] = new Layer(layerSizes[i], layerSizes[i-1]);
        }
    }

    public double[] guess(double[] input) {
        double[] nextInputs = input;

        for (int i = 0; i < getLayers().length; i++) nextInputs = activate(getLayers()[i].compute(nextInputs));

        return nextInputs;
    }

    private double[] feedForward(double[] inputs, int k) {

        double[] nextInputs = inputs;

        for (int i = 0; i < networkLength; i++) {
            layerResults[i] = getLayers()[i].compute(nextInputs);
            layerActivations[i + 1] = activate(layerResults[i]);
            nextInputs = layerActivations[i + 1];
        }

        return nextInputs;
    }

    private void backPropagate(double[] delta) {

        for (int i = networkLength - 2; i >= 0; i--) {

            delta = MatrixMath.hadamard(
                    MatrixMath.multiply(
                            MatrixMath.transpose(layers[i + 1].getWeights()),
                            delta),
                    activateDerivation(layerResults[i])
            );

            deltaBiases[i] = MatrixMath.sum(
                    deltaBiases[i],
                    delta
            );

            deltaWeights[i] = MatrixMath.sum(
                    deltaWeights[i],
                    MatrixMath.multiply(
                            delta,
                            layerActivations[i]
                    )
            );
        }
    }

    private double batchTrain(double[][] inputBatch, double[][] target, int start, int end) {

        // Initialize deltas

        deltaBiases = new double[networkLength][];
        deltaWeights = new double[networkLength][][];

        double totalError = 0;

        for (int i = 0; i < networkLength; i++) {
            deltaBiases[i] = new double[layers[i].getLayerSize()];

            int inputSize;
            if (i == 0) {
                inputSize = inputBatch[0].length;
            } else {
                inputSize = getLayers()[i - 1].getWeights().length;
            }
            deltaWeights[i] = new double[layers[i].getLayerSize()][inputSize];
        }

        // Batch train

        for (int i = start; i < end; i++) {

            double[] input = inputBatch[i];

            layerResults = new double[networkLength][];
            layerActivations = new double[networkLength + 1][];

            layerActivations[0] = input;

            for (int j = 0; j < networkLength; j++) {
                layerResults[j] = new double[layers[j].getLayerSize()];
                layerActivations[j + 1] = new double[layers[j].getLayerSize()];
            }

            double[] prediction = feedForward(input, i);

            double[] delta = MatrixMath.hadamard(
                    costDerivative(prediction, target[i]),
                    activateDerivation(layerResults[layerResults.length - 1]));

            deltaBiases[deltaBiases.length - 1] = MatrixMath.sum(
                    deltaBiases[deltaBiases.length - 1],
                    delta
            );

            deltaWeights[deltaWeights.length - 1] = MatrixMath.sum(
                    deltaWeights[deltaWeights.length - 1],
                    MatrixMath.multiply(
                            delta,
                            layerActivations[layerActivations.length - 2]
                    )
            );

            backPropagate(delta);
        }

        adjustWeights(inputBatch.length);
        return totalError;
    }

    public void train(double[][] dataset, double[][] targets, Integer batchSize) {

        if (batchSize == null || batchSize > dataset.length) {
            batchSize = dataset.length;
        }

        int epoch = 0;

        while (epoch < 100000){

            System.out.println("Epoch " + epoch + ":");

            int i = 0;
            while (i < dataset.length) {
                int batchStart = i;
                int batchEnd = i + batchSize;

                batchTrain(dataset, targets, batchStart, batchEnd);
                i += batchSize;
            }

            epoch++;
        }


    }

    private void adjustWeights(int batchSize) {
        for (int i = getLayers().length - 1; i >= 0; i--) {

            double[][] change = MatrixMath.multiply((-1 * this.learningRate) / (double) batchSize, deltaWeights[i]);
            double[][] newWeights = MatrixMath.sum(getLayers()[i].getWeights(), change);
            getLayers()[i].setWeights(newWeights);

            double[] changeBias = MatrixMath.multiply((-1 * this.learningRate) / (double) batchSize, deltaBiases[i]);
            double[] newBias = MatrixMath.sum(getLayers()[i].getBias(), changeBias);
            getLayers()[i].setBias(newBias);
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

    public void printNetwork() {
        for (int i = 0; i < this.getLayers().length; i++) {

            System.out.println("Layer " + i + ":");
            double[][] layerWeights = this.getLayers()[i].getWeights();
            double[] layerBias = this.getLayers()[i].getBias();

            for (int j = 0; j < layerWeights.length; j++) {
                System.out.print("[");
                for (int k = 0; k < layerWeights[j].length; k++) {
                    System.out.print(layerWeights[j][k] + " ");
                }
                System.out.print("] [" + layerBias[j] + "]");
                System.out.println();
            }
        }
        System.out.println();
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

    public double[] costDerivative(double[] predicted, double[] target) {
        return MatrixMath.subtract(predicted, target);
    }

    public Layer[] getLayers() {
        return layers;
    }
}
