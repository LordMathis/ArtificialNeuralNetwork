package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.*;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {2,2,1};

        IActivate activate = new SigmoidActivation();
        //ICost cost = new QuadraticCost(activate);
        ICost cost = new CrossEntropyCost();

        Network network = new Network(networkLayers, activate, cost, 0.1d);
        network.printNetwork();

        double[][] batch = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
        double[][] targets = {{0.0d}, {1.0d}, {1.0d}, {0.0d}};

        //double[][] targets = {{1.0d}, {0.0d}, {0.0d}, {0.0d}};

        network.train(batch, targets, null);
        network.printNetwork();

        System.out.println(Arrays.toString(network.guess(new double[]{1.0d, 1.0d})));
        System.out.println(Arrays.toString(network.guess(new double[]{1.0d, 0.0d})));
        System.out.println(Arrays.toString(network.guess(new double[]{0.0d, 1.0d})));
        System.out.println(Arrays.toString(network.guess(new double[]{0.0d, 0.0d})));
    }
}
