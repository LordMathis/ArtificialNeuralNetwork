package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.IActivate;
import cz.muni.fi.namesny.network.Network;
import cz.muni.fi.namesny.network.SigmoidActivation;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {2,1};

        IActivate activate = new SigmoidActivation();

        Network network = new Network(networkLayers, activate,0.1d);
        network.printNetwork();


        network.getLayers()[0].setWeights(new double[][]{{0.39,0.86}});
        network.printNetwork();

        //double[][] input = {{1.0d, 1.0d, 0.0d, 0.0d}, {1.0d, 0.0d, 1.0d, 0.0d}};

//        double[][][] batch = {{{1.0d, 1.0d}}, {{1.0d, 0.0d}}, {{0.0d, 1.0d}}, {{0.0d, 0.0d}}};
//        double[][][] targets = {{{0.0d}}, {{1.0d}}, {{1.0d}}, {{0.0d}}};
        double[] input = {1.0d, 1.0d};

        double[][] batch = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
        //double[][] targets = {{0.0d}, {1.0d}, {1.0d}, {0.0d}};

        double[][] targets = {{1.0d}, {0.0d}, {0.0d}, {0.0d}};

        //int epochs = Integer.MAX_VALUE;
        int epochs = 1;

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + i + ":");
            network.batchTrain(batch, targets);
            network.printNetwork();
        }

        System.out.println(Arrays.toString(network.guess(new double[]{1.0d, 1.0d})));
    }
}
