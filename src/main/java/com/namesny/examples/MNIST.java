package com.namesny.examples;

import com.namesny.network.*;
import com.namesny.utils.DataWrapper;
import com.namesny.utils.MNISTLoader;

import java.io.File;

public class MNIST {

    public void train() {

        DataWrapper mnist = loadMnistData();

        int[] networkLayers = {784, 50, 10};

        IActivate activate = new SigmoidActivation();
        ICost entropyCost = new CrossEntropyCost();

        Network network = new Network(networkLayers, activate, entropyCost);

        final long startTime = System.currentTimeMillis();

        network.train(
                mnist.getData(),
                mnist.getLabels(),
                2d,
                10,
                60,
                0.1d,
                0.9d,
                true);

        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");

        DataWrapper mnistTest = loadMnistTestData();
        double accuracy = network.accuracy(mnistTest.getData(), mnistTest.getLabels());
        System.out.println(accuracy);

    }

    private DataWrapper loadMnistData() {
        File dataFile = new File("MNIST_DATA/mnist_train_vectors.csv");
        File labelFile = new File("MNIST_DATA/mnist_train_labels.csv");

        MNISTLoader loader = new MNISTLoader();
        loader.load(dataFile, labelFile);

        return new DataWrapper(loader.getData(), loader.getLabels());
    }

    private DataWrapper loadMnistTestData() {
        File testDataFile = new File("MNIST_DATA/mnist_test_vectors.csv");
        File testLabelsFile = new File("MNIST_DATA/mnist_test_labels.csv");

        MNISTLoader loader = new MNISTLoader();
        loader.load(testDataFile, testLabelsFile);

        return new DataWrapper(loader.getData(), loader.getLabels());
    }

}
