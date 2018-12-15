package com.namesny.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataUtils {

    public static DataWrapper shuffle(double[][] data, double[][] labels) {

        return subset(data, labels, data.length);

    }

    public static DataWrapper filter(double[][] data, double[][] labels, List<Integer> allowed) {

        List<double[]> labelList;
        labelList = new ArrayList<>();

        List<double[]> dataList;
        dataList = new ArrayList<>();

        int label;
        for (int i = 0; i < data.length; i++) {
            label = MatrixUtils.argmax(labels[i]);
            if (allowed.contains(label)) {
                dataList.add(data[i]);
                labelList.add(labels[i]);
            }
        }

        double[][] filteredLabels = new double[labelList.size()][9];
        double[][] filteredData = new double[dataList.size()][data[0].length];

        for (int i = 0; i < labelList.size(); i++) {
            filteredLabels[i] = labelList.get(i);
            filteredData[i] = dataList.get(i);

        }

        return new DataWrapper(filteredData, filteredLabels);

    }

    public static DataWrapper subset(double[][] data, double[][] labels, int size) {

        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            indexList.add(i);
        }

        Collections.shuffle(indexList);

        double[][] shuffledData = new double[size][data[0].length];
        double[][] shuffledLabels = new double[size][labels[0].length];

        for (int i = 0; i < size; i++) {
            shuffledData[i] = data[indexList.get(i)];
            shuffledLabels[i] = labels[indexList.get(i)];
        }

        return new DataWrapper(shuffledData, shuffledLabels);

    }



}
