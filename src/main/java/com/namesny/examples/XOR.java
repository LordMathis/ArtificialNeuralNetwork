package com.namesny.examples;

import com.namesny.network.CrossEntropyCost;
import com.namesny.network.Network;
import com.namesny.network.SigmoidActivation;
import com.namesny.utils.MatrixUtils;

public class XOR {

    public static void trainXOR(String[] args) {
        int[] networkLayers = {2, 2, 1};

        double[][] trainingData = {{1.0d, 0.0d}, {1.0d, 1.0d}, {0.0d, 0.0d}, {0.0d, 1.0d}};
        double[][] trainingLabels = {{1.0d}, {0.0d}, {0.0d}, {1.0d}};

        Network network = new Network(networkLayers, new SigmoidActivation(), new CrossEntropyCost());

        final long startTime = System.currentTimeMillis();
        network.train(
                trainingData,
                trainingLabels,
                0.01,
                null,
                 100000,
                0.0d,
                0.01d,
                true);
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");

        double acc;
        acc = network.accuracy(trainingData, trainingLabels);
        System.out.println(acc);

        MatrixUtils.printVector(network.guess(new double[]{0.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{0.0d, 1.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 1.0d}));
    }

}
