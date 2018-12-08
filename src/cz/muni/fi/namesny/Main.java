package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.*;
import cz.muni.fi.namesny.utils.DataUtils;
import cz.muni.fi.namesny.utils.DataWrapper;
import cz.muni.fi.namesny.utils.MNISTLoader;
import cz.muni.fi.namesny.utils.MatrixUtils;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        trainXOR();

    }

    public static void gridSearch(
            double[][] trainingData,
            double[][] trainingLabels,
            double[][] validationData,
            double[][] validationLabels,
            int[] epochs,
            int[][] networkSizes,
            double[] learningRates,
            int[] batchSizes,
            double[] lambdas,
            double[] momentums,
            File output) throws FileNotFoundException {

        PrintWriter pw = new PrintWriter(output);
        pw.write("Network Size\t Learning Rate\t Batch Size\t Lambda\t Momentum\t Epochs\t Time\t Accuracy\n");

        for (int[] networkSize: networkSizes) {
            for (double learningRate: learningRates) {
                for (int batchSize: batchSizes) {
                    for (double lambda: lambdas) {
                        for (double momentum: momentums) {
                            for (int epoch: epochs) {

                                Network network = new Network(networkSize, new SigmoidActivation(), new CrossEntropyCost(), learningRate);

                                final long startTime = System.currentTimeMillis();
                                network.train(trainingData, trainingLabels, batchSize, epoch, lambda, momentum, false);
                                final long endTime = System.currentTimeMillis();
                                long timeMins = (endTime - startTime) / 60000;

                                double accuracy = network.accuracy(validationData, validationLabels);

                                String res = Arrays.toString(networkSize) + "\t" + learningRate + "\t" + batchSize +
                                        "\t" + lambda + "\t" + momentum + "\t" + epoch + "\t" + timeMins + "\t" + accuracy + "\n";
                                System.out.println(res);
                                pw.append(res);
                                pw.flush();

                            }
                        }
                    }
                }
            }
        }

        pw.close();
    }

    public static void trainXOR() {
        int[] networkLayers = {2, 2, 1};

        double[][] trainingData = {{1.0d, 0.0d}, {1.0d, 1.0d}, {0.0d, 0.0d}, {0.0d, 1.0d}};
        double[][] trainingLabels = {{1.0d}, {0.0d}, {0.0d}, {1.0d}};

        Network network = new Network(networkLayers, new SigmoidActivation(), new CrossEntropyCost(), 0.001d);

        network.train(
                trainingData,
                trainingLabels,
                null,
                100000,
                10d,
                0.0d,
                false
        );

        double acc = network.accuracy(trainingData, trainingLabels);
        System.out.println(acc);

        MatrixUtils.printVector(network.guess(new double[]{0.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{0.0d, 1.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 1.0d}));

    }

    public static void trainMnistSmall() {

        DataWrapper dataWrapper = loadMnistData();
        DataWrapper testDataWrapper = loadMnistTestData();

        // Smaller dataset
        DataWrapper subset = DataUtils.subset(dataWrapper.getData(), dataWrapper.getLabels(), 1000);
        double[][] smallData = subset.getData();
        double[][] smallLabels = subset.getLabels();

        // Smaller test set
        DataWrapper testSubset = DataUtils.subset(testDataWrapper.getData(), testDataWrapper.getLabels(), 100);
        double[][] smallTestData = testSubset.getData();
        double[][] smallTestLabels = testSubset.getLabels();

    }

    public static void trainMNIST01() {

        DataWrapper dataWrapper = loadMnistData();
        DataWrapper testDataWrapper = loadMnistTestData();

        DataWrapper zerosAndOnes = DataUtils.filter(dataWrapper.getData(), dataWrapper.getLabels(), Arrays.asList(0, 1));
        double[][] zerosAndOnesData = zerosAndOnes.getData();
        double[][] zerosAndOnesLabels = zerosAndOnes.getLabels();

        DataWrapper filtered = DataUtils.filter(testDataWrapper.getData(), testDataWrapper.getLabels(), Arrays.asList(0, 1));
        double[][] zerosAndOnesTestData = filtered.getData();
        double[][] zerosAndOnesTestLabels = filtered.getLabels();
    }


    public static DataWrapper loadMnistData() {
        File dataFile = new File("MNIST_DATA/mnist_train_vectors.csv");
        File labelFile = new File("MNIST_DATA/mnist_train_labels.csv");

        MNISTLoader loader = new MNISTLoader();
        loader.load(dataFile, labelFile);

        return new DataWrapper(loader.getData(), loader.getLabels());
    }

    public static DataWrapper loadMnistTestData() {
        File testDataFile = new File("MNIST_DATA/mnist_test_vectors.csv");
        File testLabelsFile = new File("MNIST_DATA/mnist_test_labels.csv");

        MNISTLoader loader = new MNISTLoader();
        loader.load(testDataFile, testLabelsFile);

        return new DataWrapper(loader.getData(), loader.getLabels());
    }

    public static void trainMNIST() {
        int[] networkLayers = {784, 50, 10};


        /*
        Test data
         */

        loader.load(testDataFile, testLabelsFile);
        double[][] testData = loader.getData();
        double[][] testLabels = loader.getLabels();

        // 0s and 1s
        DataWrapper filtered = DataUtils.filter(loader.getData(), loader.getLabels(), Arrays.asList(0, 1));
        double[][] zerosAndOnesTestData = filtered.getData();
        double[][] zerosAndOnesTestLabels = filtered.getLabels();

        // Smaller test set
        DataWrapper testSubset = DataUtils.subset(testData, testLabels, 100);
        double[][] smallTestData = testSubset.getData();
        double[][] smallTestLabels = testSubset.getLabels();


        IActivate activate = new SigmoidActivation();
        ICost quadraticCost = new QuadraticCost(activate);
        ICost entropyCost = new CrossEntropyCost();

        Network network = new Network(networkLayers, activate, entropyCost, 0.5d);

        int[][] networkSizes = {
                {784, 30, 10}, {784, 50, 10}, {784, 100, 10}, {784, 30, 30, 10}, {784, 50, 30, 10},
                {784, 50, 50, 10}, {784, 100, 50, 10}, {784, 100, 30, 10}, {784, 100, 100, 10}
        };

//        try {
//            gridSearch(smallData, smallLabels, smallTestData, smallTestLabels,
//                    new int[] {10, 30},
//                    networkSizes,
//                    new double[] {0.01, 0.05, 0.1, 0.5, 1d, 5d},
//                    new int[] {10, 100, 1000},
//                    new double[] {0, 0.01, 0.1, 1, 10, 100},
//                    new double[] {0, 0.1, 0.5, 0.9},
//                    new File("gridSearch.tsv"));
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }

        final long startTime = System.currentTimeMillis();

        //network.train(trainingData, trainingLabels, 10, 10, 1d, 0.1d, true);
        // network.train(zerosAndOnesData, zerosAndOnesLabels, 10, 10, 1d, 0.0d, true);
        // network.train(smallData, smallLabels, 10, 10, 0.0d, 0.0d, true);

        final long endTime = System.currentTimeMillis();

        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");
//        System.out.println();
        double acc;
        acc = network.accuracy(testData, testLabels);
        // acc = network.accuracy(zerosAndOnesTestData, zerosAndOnesTestLabels);
        //acc = network.accuracy(smallTestData, smallTestLabels);
        System.out.println(acc);

        // Best so far: 30hn, 0.05lr, 1lmbd, 10bs
        // 0.74 30 1 0 10 0
    }
}
