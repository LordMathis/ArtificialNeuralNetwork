package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.IActivate;
import cz.muni.fi.namesny.network.Layer;
import cz.muni.fi.namesny.network.Network;
import cz.muni.fi.namesny.network.SigmoidActivation;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {2,2,1};

        IActivate activate = new SigmoidActivation();

        Network network = new Network(networkLayers, activate,0.01d);

        //double[][] input = {{1.0d, 1.0d, 0.0d, 0.0d}, {1.0d, 0.0d, 1.0d, 0.0d}};

        double[][] batch = {{1.0d, 1.0d}, {1.0d, 0.0d}, {0.0d, 1.0d}, {0.0d, 0.0d}};
        double[][] actual = {{0.0d}, {1.0d}, {1.0d}, {0.0d}};
        double[] input = {1.0d, 1.0d};

        System.out.println(Arrays.deepToString(batch));
        System.out.println(Arrays.deepToString(actual));

        //int epochs = Integer.MAX_VALUE;
        int epochs = 1;

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + i + ":");
            network.batchTrain(batch, actual);
            System.out.println("1 xor 1 guess: " + Arrays.toString(network.guess(input)));
            System.out.println("Network weights:");
            for (Layer layer :
                    network.getLayers()) {
                System.out.println(Arrays.deepToString(layer.getWeights()));
            }
            System.out.println();
        }
    }
}
