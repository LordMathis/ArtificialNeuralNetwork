package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.*;
import cz.muni.fi.namesny.utils.DataUtils;
import cz.muni.fi.namesny.utils.DataWrapper;
import cz.muni.fi.namesny.utils.MNISTLoader;
import cz.muni.fi.namesny.utils.MatrixUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        //trainXOR();
        trainMnistSmall(false);
        //trainMNIST(false);
    }

    public static void gridSearch(
            DataWrapper trainingData,
            DataWrapper validationData,
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

                                Network network = new Network(networkSize, new SigmoidActivation(), new CrossEntropyCost());

                                final long startTime = System.currentTimeMillis();
                                network.train(trainingData.getData(), trainingData.getLabels(), learningRate, batchSize, epoch, lambda, momentum, false);
                                final long endTime = System.currentTimeMillis();
                                long timeMins = (endTime - startTime) / 60000;

                                double accuracy = network.accuracy(validationData.getData(), validationData.getLabels());

                                String res = Arrays.toString(networkSize) + "\t" + learningRate + "\t" + batchSize +
                                        "\t" + lambda + "\t" + momentum + "\t" + epoch + "\t" + timeMins + "\t" + accuracy + "\n";
                                System.out.print(res);
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

        Network network = new Network(networkLayers, new SigmoidActivation(), new CrossEntropyCost());

        justTrain(
                network,
                new DataWrapper(trainingData, trainingLabels),
                new DataWrapper(trainingData, trainingLabels),
                0.01d,
                100000,
                 null,
                0.0d,
                0.1d,
                false
        );

        MatrixUtils.printVector(network.guess(new double[]{0.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{0.0d, 1.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 0.0d}));
        MatrixUtils.printVector(network.guess(new double[]{1.0d, 1.0d}));

    }

    public static void trainMnistSmall(boolean standard) {

        DataWrapper dataWrapper = loadMnistData();
        DataWrapper testDataWrapper = loadMnistTestData();

        // Smaller dataset
        DataWrapper subset = DataUtils.subset(dataWrapper.getData(), dataWrapper.getLabels(), 1000);

        // Smaller test set
        DataWrapper testSubset = DataUtils.subset(testDataWrapper.getData(), testDataWrapper.getLabels(), 100);

        if (standard) {
            Network network = new Network(new int[]{784, 50, 10}, new SigmoidActivation(), new CrossEntropyCost());
            justTrain(
                    network,
                    subset,
                    testSubset,
                    1,
                    50,
                    10,
                    0,
                    0.5,
                    true);
        } else {

            try {

                int[][] networkSizes = new int[][]{{784, 100, 10}, {784, 150, 10}, {784, 200, 10}, {784, 50, 50, 10}, {784, 100, 100, 10}, {784, 100, 50, 10}};

                gridSearch(
                        subset,
                        testSubset,
                        new int[]{30},
                        networkSizes,
                        new double[]{0.1d},
                        new int[]{10},
                        new double[]{0},
                        new double[]{0},
                        new File("network_sizes.tsv")
                );
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

    }

    public static void trainMNIST01() {

        DataWrapper dataWrapper = loadMnistData();
        DataWrapper testDataWrapper = loadMnistTestData();

        DataWrapper zerosAndOnes = DataUtils.filter(dataWrapper.getData(), dataWrapper.getLabels(), Arrays.asList(0, 1));

        DataWrapper filtered = DataUtils.filter(testDataWrapper.getData(), testDataWrapper.getLabels(), Arrays.asList(0, 1));


        Network network = new Network(new int[]{784, 50, 10}, new SigmoidActivation(), new CrossEntropyCost());
        justTrain(
                network,
                zerosAndOnes,
                filtered,
                0.05,
                20,
                10,
                0.5,
                0.3,
                true);
    }

    public static void trainMNIST(boolean standard) {

        DataWrapper mnist = loadMnistData();
        DataWrapper mnistTest = loadMnistTestData();

        if (standard) {

            int[] networkLayers = {784, 100, 10};

            IActivate activate = new SigmoidActivation();
            ICost entropyCost = new CrossEntropyCost();

            Network network = new Network(networkLayers, activate, entropyCost);

            justTrain(
                    network,
                    mnist,
                    mnistTest,
                    0.1d,
                    20,
                    10,
                    0.0d,
                    0.0d,
                    true
            );
        } else {

            try {

                int[][] networkSizes = new int[][]{{784, 100, 10}, {784, 150, 10}, {784, 200, 10}, {784, 50, 50, 10}, {784, 100, 100, 10}, {784, 100, 50, 10}};

                gridSearch(
                        mnist,
                        mnistTest,
                        new int[]{30},
                        networkSizes,
                        new double[]{0.1d},
                        new int[]{10},
                        new double[]{0},
                        new double[]{0},
                        new File("refined_search.tsv")
                );
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

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

    public static void justTrain(Network network,
                                 DataWrapper data,
                                 DataWrapper testData,
                                 double learningRate,
                                 int epochs,
                                 Integer batchSize,
                                 double lambda,
                                 double momentum,
                                 boolean monitor) {

        final long startTime = System.currentTimeMillis();
        network.train(data.getData(), data.getLabels(), learningRate, batchSize, epochs, lambda, momentum, monitor);
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / 60000 + " minutes");

        double acc;
        acc = network.accuracy(testData.getData(), testData.getLabels());
        System.out.println(acc);

    }
}