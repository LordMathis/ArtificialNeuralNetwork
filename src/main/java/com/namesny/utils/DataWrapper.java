package com.namesny.utils;

public class DataWrapper {

    private double[][] data;
    private double[][] labels;

    public DataWrapper(double[][] data, double[][] labels) {
        this.data = data;
        this.labels = labels;
    }

    public double[][] getData() {
        return data;
    }

    public double[][] getLabels() {
        return labels;
    }
}
