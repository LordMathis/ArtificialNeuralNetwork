package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

public class Layer {

    private double[][] layer;
    private double[] bias;

    public Layer(int numNeurons, int inputSize) {
        this.setLayer(Utils.initializeMatrix(numNeurons, inputSize, true));
        this.setBias(Utils.initializeVector(numNeurons, null));
    }

    public double[] compute(double[] inputs) {
        return MatrixMath.sum(MatrixMath.multiply(getLayer(), inputs), getBias());
    }

    public double[][] compute(double[][] inputs) {
        return MatrixMath.sum(MatrixMath.multiply(getLayer(), inputs), getBias());
    }

    public double[][] getLayer() {
        return layer;
    }

    public void setLayer(double[][] layer) {
        this.layer = layer;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }
}
