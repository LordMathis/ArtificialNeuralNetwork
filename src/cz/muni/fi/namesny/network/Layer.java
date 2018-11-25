package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

public class Layer {

    private double[][] weights;
    private double[] bias;

    public Layer(int numNeurons, int inputSize) {
        this.setWeights(Utils.initializeMatrix(numNeurons, inputSize, true));
        this.setBias(Utils.initializeVector(numNeurons, null));
    }

    public double[] compute(double[] inputs) {
        return MatrixMath.sum(MatrixMath.multiply(getWeights(), inputs), getBias());
    }

    public double[][] compute(double[][] inputs) {
        return MatrixMath.sum(MatrixMath.multiply(getWeights(), inputs), getBias());
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
}
