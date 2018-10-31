package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

import java.util.Arrays;

public class Layer {

    private double[][] layer;
    private double[] bias;
    private double[] error;

    public Layer(int numNeurons, int inputSize) {
        this.layer = Utils.initializeMatrix(numNeurons, inputSize, true);
        this.bias = Utils.initializeVector(numNeurons, null);
    }

    public double[] compute(double[] inputs) {

        return MatrixMath.sum(MatrixMath.multiply(layer, inputs), bias);

    }

    public void setError(double[] error) {
        this.error = error;
    }
}
