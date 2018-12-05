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

        System.out.println(Arrays.toString(Utils.getDimensions(trainingData)));
        System.out.println(Arrays.toString(Utils.getDimensions(trainingLabels)));

        IActivate activate = new SigmoidActivation();
        //ICost cost = new QuadraticCost(activate);
        ICost cost = new CrossEntropyCost();

        Network network = new Network(networkLayers, activate, cost, 0.1d);

//        double[][] batch = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
//        double[][] targets = {{0.0d}, {1.0d}, {1.0d}, {0.0d}};

        network.train(trainingData, trainingLabels, 100);

        System.out.println(Arrays.toString(network.guess(trainingData[0])));
        System.out.println(Arrays.toString(trainingLabels[0]));

        loader.load(testDataFile, testLabelsFile);
        double[][] testData = loader.getData();
        double[][] testLabels = loader.getLabels();
    }
}
