package cz.muni.fi.namesny.dataloaders;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MNISTLoader {

    private double[][] data;
    private double[][] labels;

    public void load(File dataFile, File labelFile){

        try {

            loadData(dataFile);
            loadLabels(labelFile);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void loadData(File dataFile) throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(dataFile));

        List<double[]> dataList;
        dataList = new ArrayList<>();

        String line;
        while ((line = br.readLine()) != null) {

            String[] row = line.split(",");
            double[] dataRow = new double[row.length];

            for (int i = 0; i < row.length; i++) {
                dataRow[i] = Double.parseDouble(row[i]);
            }

            dataList.add(dataRow);
        }

        this.data = new double[dataList.size()][dataList.get(0).length];
        for (int i = 0; i < dataList.size(); i++) {
            data[i] = dataList.get(i);
        }
        

    }

    private void loadLabels(File labelFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(labelFile));

        List<double[]> labelList;
        labelList = new ArrayList<>();

        String line;
        while ((line = br.readLine()) != null) {

            double[] labelRow = new double[9];
            int label = Integer.parseInt(line);

            for (int i = 0; i < labelRow.length; i++) {
                if (i == label - 1) {
                    labelRow[i] = 1.0d;
                } else {
                    labelRow[i] = 0.0d;
                }
            }

            labelList.add(labelRow);
        }

        this.labels = new double[labelList.size()][9];
        for (int i = 0; i < labelList.size(); i++) {
            labels[i] = labelList.get(i);
        }

    }

    public double[][] getData() {
        return data;
    }

    public double[][] getLabels() {
        return labels;
    }
}