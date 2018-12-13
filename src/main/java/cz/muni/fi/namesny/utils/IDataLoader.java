package cz.muni.fi.namesny.utils;

import java.io.File;

public interface IDataLoader {
    void load(File dataFile, File labelFile);
    double[][] getData();
    double[][] getLabels();
}
