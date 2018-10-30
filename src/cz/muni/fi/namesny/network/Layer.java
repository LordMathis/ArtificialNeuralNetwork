package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import cz.muni.fi.namesny.matrixutils.Utils;

import java.util.Arrays;
import java.util.Random;

public class Layer {

    private double[][] layer;
    private double[] bias;

    public Layer(int numNeurons, int inputSize) {

        this.layer = Utils.initializeMatrix(numNeurons, inputSize, true);
        this.bias = Utils.initializeVector(numNeurons, null);
    }

    public double[] compute(double[] inputs) {

        double[] result = MatrixMath.sum(MatrixMath.multiply(layer, inputs), bias);

        for (int i = 0; i < result.length; i++) {
            result[i] = activate(result[i]);
        }

        System.out.println(Arrays.toString(result));

        return result;

    }

    private double activate(double input) {

        return 1.0d / (1.0d + Math.exp(-input));
    }

}
