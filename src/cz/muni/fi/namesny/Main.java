package cz.muni.fi.namesny;

import cz.muni.fi.namesny.dataloaders.MNISTLoader;
import cz.muni.fi.namesny.matrixutils.Utils;
import cz.muni.fi.namesny.network.*;

import java.io.File;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {784,100,10};

        File dataFile = new File("MNIST_DATA/mnist_train_vectors.csv");
        File labelFile = new File("MNIST_DATA/mnist_train_labels.csv");
        File testDataFile = new File("MNIST_DATA/mnist_test_vectors.csv");
        File testLabelsFile = new File("MNIST_DATA/mnist_test_labels.csv");


        MNISTLoader loader = new MNISTLoader();
        loader.load(dataFile, labelFile);
        double[][] trainingData = loader.getData();
        double[][] trainingLabels = loader.getLabels();

        loader.load(testDataFile, testLabelsFile);
        double[][] testData = loader.getData();
        double[][] testLabels = loader.getLabels();

        IActivate activate = new SigmoidActivation();
        //ICost quadraticCost = new QuadraticCost(activate);
        ICost entropyCost = new CrossEntropyCost();

        Network network = new Network(networkLayers, activate, entropyCost, 0.5d);

        System.out.println();
        System.out.println(network.accuracy(testData, testLabels));

//        double[][] trainingData = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
//        double[][] trainingLabels = {{0.0d, 1.0d}, {1.0d, 0.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}};

//        Utils.printVector(network.guess(testData[0]));
//        Utils.printVector(testLabels[0]);

        final long startTime = System.currentTimeMillis();
        network.train(trainingData, trainingLabels, 100, 30);
        final long endTime = System.currentTimeMillis();

        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");
        System.out.println();
        System.out.println(network.accuracy(testData, testLabels));

    }
}
