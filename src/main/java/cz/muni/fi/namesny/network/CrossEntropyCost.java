package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.utils.MatrixMath;

public class CrossEntropyCost implements ICost{
    @Override
    public double getCost(double[] prediction, double[] target) {

        double cost = 0;

        if (prediction.length != target.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        for (int i = 0; i < prediction.length; i++) {

            double logPred = Math.log(prediction[i]);
            if (Double.isNaN(logPred)) {
                logPred = 0.0d;
            }

            double logMinusPred = Math.log(1 - prediction[i]);
            if (Double.isNaN(logMinusPred)) {
                logMinusPred = 0.0d;
            }
            cost += target[i] * logPred + (1 - target[i]) * logMinusPred;
        }

        return cost;
    }

    @Override
    public double[] getDelta(double[] prediction, double[] target, double[] z) {
        return MatrixMath.subtract(prediction, target);
    }
}
