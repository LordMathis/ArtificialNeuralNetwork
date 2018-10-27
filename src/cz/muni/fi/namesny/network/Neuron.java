package cz.muni.fi.namesny.network;

import java.util.List;

public class Neuron {

    private int bias;
    private List<Double> weights;

    public Neuron() {

    }

    public List<Double> getWeights() {
        return weights;
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public int getBias() {
        return bias;
    }

    public void setBias(int bias) {
        this.bias = bias;
    }
}
