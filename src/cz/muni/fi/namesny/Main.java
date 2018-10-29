package cz.muni.fi.namesny;

import cz.muni.fi.namesny.network.Network;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        int[] networkLayers = {2,2,2,1};
        Network network = new Network(networkLayers);

        double[] input = {1.0d, 0.0d};
        double[] result = network.feedForward(input);

        System.out.println(Arrays.toString(result));
    }
}
