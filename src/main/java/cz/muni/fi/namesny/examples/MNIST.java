package cz.muni.fi.namesny.examples;

import cz.muni.fi.namesny.network.*;
import cz.muni.fi.namesny.utils.DataWrapper;
import cz.muni.fi.namesny.utils.MNISTLoader;
import cz.muni.fi.namesny.utils.MatrixUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

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
                2.0d,
                10,
                60,
                0.09d,
                0.9d,
                true);

        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");

        //exportPredictions(network, mnist.getData(), new File("trainPredictions"));

        DataWrapper mnistTest = loadMnistTestData();
        //exportPredictions(network, mnistTest.getData(), new File("actualTestPredictions"));

        System.out.println("Test accuracy : " + network.accuracy(mnistTest.getData(), mnistTest.getLabels()));


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

    private void exportPredictions(Network network, double[][] data, File file) {

        double[][] results = new double[data.length][10];

        for (int i = 0; i < data.length; i++) {
            results[i] = network.guess(data[i]);
        }

        PrintWriter pw = null;
        try {
            pw = new PrintWriter(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < results.length; i++) {
            int predictedClass = MatrixUtils.argmax(results[i]);
            pw.write(predictedClass + "\n");
        }

        pw.close();

    }

}
