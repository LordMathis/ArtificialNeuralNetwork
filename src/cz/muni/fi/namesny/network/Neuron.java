package cz.muni.fi.namesny.network;

import java.util.Random;

public class Neuron {

    private int bias;
    private double[] weights;

    public Neuron(int weightsSize) {
        this.weights = initializeWeights(weightsSize);
    }

    private double[] initializeWeights(int weightsSize) {
        Random r = new Random();
        double[] weights = new double[weightsSize];

        for (int i = 0; i < weightsSize; i++) {
            weights[i] = r.nextDouble();
        }

        return  weights;
    }

    public double compute(double[] inputs) {
        double value = bias;

        for (int i = 0; i < inputs.length; i++) {
            value += weights[i]*inputs[i];
        }

        return activation(value);
    }

    public double activation(double input) {
        return 1.0d / (1.0d + Math.exp(-input));
    }

}
