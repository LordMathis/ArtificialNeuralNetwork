package cz.muni.fi.namesny.network;

import java.util.List;

public class Network {

    private Layer[] layers;

    public Network(int[] layerSizes) {

        this.layers = new Layer[layerSizes.length];

        for (int i = 0; i < layerSizes.length; i++) {
            Integer layerSize = layerSizes[i];
            Integer prevLayerSize = layerSize;

            if (i>0){
                prevLayerSize = layerSizes[i-1];
            }

            this.layers[i] = new Layer(layerSize, prevLayerSize);
        }
    }

    public double[] feedForward(double[] inputs) {

        double[] nextInputs = inputs;

        for (Layer layer :
                layers) {
            nextInputs = layer.compute(nextInputs);
        }

        return nextInputs;
    }

    public double[] softmax(double[] input) {

        double totalInput = 0;

        for (double in : input) {
            totalInput += Math.exp(in);
        }

        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i]) / totalInput;
        }

        return result;
    }

}
