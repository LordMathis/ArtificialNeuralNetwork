package cz.muni.fi.namesny.network;

import cz.muni.fi.namesny.matrixutils.MatrixMath;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class CrossEntropyCost implements ICost{
    @Override
    public double getCost(double[] prediction, double[] target) {
        throw new NotImplementedException();
    }

    @Override
    public double[] getDelta(double[] prediction, double[] target, double[] z) {
        return MatrixMath.subtract(prediction, target);
    }
}
