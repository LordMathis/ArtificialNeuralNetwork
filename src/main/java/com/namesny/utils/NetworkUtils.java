package com.namesny.utils;

import com.namesny.network.CrossEntropyCost;
import com.namesny.network.Network;
import com.namesny.network.SigmoidActivation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;

public class NetworkUtils {
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
}
