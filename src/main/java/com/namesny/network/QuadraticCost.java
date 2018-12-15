package com.namesny.network;

import com.namesny.utils.MatrixMath;

public class QuadraticCost implements ICost{

    IActivate activate;

    public QuadraticCost(IActivate activate) {
        this.activate = activate;
    }

    @Override
    public double getCost(double prediction[], double[] target) {
        double cost = 0.0d;

        if (prediction.length != target.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        for (int i = 0; i < prediction.length; i++) {
            cost += Math.pow(target[i] - prediction[i], 2);
        }

        return cost/2.0d;
    }

    @Override
    public double[] getDelta(double[] prediction, double[] target, double[] z) {

        double[] delta = MatrixMath.hadamard(
                MatrixMath.subtract(prediction, target),
                activate.getDerivative(z));

        return delta;
    }
}
