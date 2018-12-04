package cz.muni.fi.namesny.network;

public class SigmoidActivation implements IActivate{

    @Override
    public double[] getActivation(double[] x) {

        double[] result = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0d / (1.0d + Math.exp(-x[i]));
        }

        return result;
    }

    @Override
    public double[] getDerivative(double[] x) {

        double[] result = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            double a = 1.0d / (1.0d + Math.exp(-x[i]));
            result[i] = a * (1.0d - a);
        }

        return result;
    }
}
