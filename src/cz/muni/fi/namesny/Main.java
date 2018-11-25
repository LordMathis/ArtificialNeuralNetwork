package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.ActivationDerivative;
import cz.muni.fi.namesny.network.ActivationFunction;
import cz.muni.fi.namesny.network.Network;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {2,2,1};

        ActivationFunction activationFunction = (x) -> 1.0d / (1.0d + Math.exp(-x));

        ActivationDerivative activationDerivative = (x) -> {
            double a = activationFunction.compute(x);
            return a * (1.0d - a);
        };

        Network network = new Network(networkLayers,
                activationFunction,
                activationDerivative,
                0.001d);

        //double[][] input = {{1.0d, 1.0d, 0.0d, 0.0d}, {1.0d, 0.0d, 1.0d, 0.0d}};

        double[][] batch = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
        double[][] actual = {{0.0d}, {1.0d}, {1.0d}, {0.0d}};

        network.batchTrain(batch, actual);
        // System.out.println(Arrays.deepToString(result));

//        double[] input = {1.0d, 1.0d};
//        double[] result = network.feedForward(input);
//        System.out.println(Arrays.toString(result));
    }
}
