package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.utils.MathUtils;
import cz.muni.fi.namesny.utils.MatrixUtils;

public class Layer {

    private double[][] weights;
    private double[] bias;
    private int layerSize;
    private double[] layerResult;

    public Layer(int numNeurons, int inputSize) {
        this.setWeights(MatrixUtils.initializeMatrix(numNeurons, inputSize, true));
        this.layerSize = numNeurons;
        this.layerResult = MatrixUtils.initializeVector(numNeurons, null);
        this.setBias(MatrixUtils.initializeVector(numNeurons, null));
    }

    public double[] compute(double[] inputs) {
        return MathUtils.sum(
                MathUtils.multiply(getWeights(), inputs),
                getBias());
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public double[] getLayerResult() {
        return layerResult;
    }

}
