package com.namesny.utils;

import java.io.File;

public interface IDataLoader {
    void load(File dataFile, File labelFile);
    double[][] getData();
    double[][] getLabels();
}
