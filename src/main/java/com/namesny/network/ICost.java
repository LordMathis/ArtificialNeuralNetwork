package com.namesny.network;

public interface ICost {
    double getCost(double[] prediction, double[] target);
    double[] getDelta(double[] prediction, double[] target, double[] z);
}
