package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.utils.MathUtils;
import cz.muni.fi.namesny.utils.MatrixUtils;

public class Network {

    private Layer[] layers;
    private IActivate activate;
    private ICost cost;
    private double learningRate;
    private int networkLength;

    private double[][] layerResults;
    private double[][] layerActivations;
    private double[][][] deltaWeights;
    private double[][] deltaBiases;
    private double[][][] momentums;

    public Network(int[] layerSizes,
                   IActivate activate,
                   ICost cost) {

        this.layers = new Layer[layerSizes.length - 1];
        this.networkLength = this.layers.length;
        this.activate = activate;
        this.cost = cost;

        for (int i = 1; i < layerSizes.length; i++) {
            this.layers[i - 1] = new Layer(layerSizes[i], layerSizes[i-1]);
        }
    }

    public double[] guess(double[] input) {
        double[] nextInputs = input;

        for (int i = 0; i < getLayers().length; i++) nextInputs = activate.getActivation(getLayers()[i].compute(nextInputs));

        return nextInputs;
    }

    private double[] feedForward(double[] inputs) {

        double[] nextInputs = inputs;

        for (int i = 0; i < networkLength; i++) {
            layerResults[i] = getLayers()[i].compute(nextInputs);
            layerActivations[i + 1] = activate.getActivation(layerResults[i]);
            nextInputs = layerActivations[i + 1];
        }

        return nextInputs;
    }

    private void backPropagate(double[] delta) {

        for (int i = networkLength - 2; i >= 0; i--) {

            delta = MathUtils.hadamard(
                    MathUtils.multiply(
                            MathUtils.transpose(layers[i + 1].getWeights()),
                            delta),
                    activate.getDerivative(layerResults[i])
            );

            deltaBiases[i] = MathUtils.sum(
                    deltaBiases[i],
                    delta
            );

            deltaWeights[i] = MathUtils.sum(
                    deltaWeights[i],
                    MathUtils.multiply(
                            delta,
                            layerActivations[i]
                    )
            );
        }
    }

    private void batchTrain(double[][] inputBatch, double[][] targets, int start, int end, double lambda, double momentum) {

        // Initialize deltas

        deltaBiases = new double[networkLength][];
        deltaWeights = new double[networkLength][][];

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

        for (int i = start; i < Math.min(end, inputBatch.length); i++) {

            double[] input = inputBatch[i];

            layerResults = new double[networkLength][];
            layerActivations = new double[networkLength + 1][];

            layerActivations[0] = input;

            for (int j = 0; j < networkLength; j++) {
                layerResults[j] = new double[layers[j].getLayerSize()];
                layerActivations[j + 1] = new double[layers[j].getLayerSize()];
            }

            double[] prediction = feedForward(input);

            double[] delta = cost.getDelta(prediction,
                    targets[i],
                    layerActivations[layerActivations.length - 1]);

            deltaBiases[deltaBiases.length - 1] = MathUtils.sum(
                    deltaBiases[deltaBiases.length - 1],
                    delta
            );

            deltaWeights[deltaWeights.length - 1] = MathUtils.sum(
                    deltaWeights[deltaWeights.length - 1],
                    MathUtils.multiply(
                            delta,
                            layerActivations[layerActivations.length - 2]
                    )
            );

            backPropagate(delta);
        }

        adjustWeights(inputBatch.length, lambda, momentum);
    }

    public void train(double[][] dataset,
                      double[][] targets,
                      double learningRate,
                      Integer batchSize,
                      Integer epochs,
                      Double lambda,
                      Double momentum,
                      boolean monitorAccuracy) {

        this.learningRate = learningRate;

        if (batchSize == null || batchSize > dataset.length) {
            batchSize = dataset.length;
        }

        if (lambda == null) {
            lambda = 0.0d;
        }

        if (momentum == null) {
            momentum = 0.0d;
        }

        momentums = new double[networkLength][][];
        for (int i = 0; i < networkLength; i++) {
            int[] dims = MatrixUtils.getDimensions(getLayers()[i].getWeights());
            momentums[i] = MatrixUtils.initializeMatrix(dims[0], dims[1], false);
        }

        int epoch = 0;

        while (epoch < epochs){

            int i = 0;
            while (i < dataset.length) {
                int batchStart = i;
                int batchEnd = i + batchSize;
                batchTrain(dataset, targets, batchStart, batchEnd, lambda, momentum);
                i += batchSize;
            }

            if (monitorAccuracy) {
                System.out.print("Epoch " + epoch + " - ");
                System.out.println("training accuracy: " + accuracy(dataset, targets));
            }

            epoch++;
        }
    }

    private void adjustWeights(int batchSize, double lambda, double momentum) {

        for (int i = getLayers().length - 1; i >= 0; i--) {

            // Update momentum
            momentums[i] = MathUtils.sum(
                MathUtils.multiply(
                        momentum,
                        momentums[i]
                ),
                MathUtils.multiply(
                        (-1 * this.learningRate) / (double) batchSize,
                        deltaWeights[i])
            );

            double[][] regularization = MathUtils.multiply(
                    1 - (((-1) * this.learningRate * lambda) / (double) batchSize),
                    getLayers()[i].getWeights()
            );

            double[][] newWeights = MathUtils.sum(
                    regularization,
                    momentums[i]
            );

            getLayers()[i].setWeights(newWeights);

            // Update Bias

            double[] changeBias = MathUtils.multiply(
                    (-1 * this.learningRate) / (double) batchSize,
                    deltaBiases[i]);
            double[] newBias = MathUtils.sum(getLayers()[i].getBias(), changeBias);

            getLayers()[i].setBias(newBias);

            // Update Moment
        }
    }

    public double accuracy(double[][] testData, double[][] testLabels) {

        int totalCorrect = 0;

        for (int i = 0; i < testData.length; i++) {

            if (testLabels[0].length > 1) {

                int prediction = MatrixUtils.argmax(guess(testData[i]));
                int label = MatrixUtils.argmax(testLabels[i]);

                if (prediction == label) {
                    totalCorrect++;
                }
            } else {

                double prediction;

                if (guess(testData[i])[0] < 0.5d) {
                    prediction = 0d;
                } else {
                    prediction = 1d;
                }

                if (prediction == testLabels[i][0]) {
                    totalCorrect++;
                }

            }
        }

        return totalCorrect / (double) testData.length;
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

    public Layer[] getLayers() {
        return layers;
    }

}
