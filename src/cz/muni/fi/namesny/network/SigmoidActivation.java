package cz.muni.fi.namesny.network;

public class SigmoidActivation implements IActivate{

    @Override
    public double getActivation(double x) {
        return 1.0d / (1.0d + Math.exp(-x));
    }

    @Override
    public double getDerivative(double x) {
        double a = getActivation(x);
        return a * (1.0d - a);
    }
}
