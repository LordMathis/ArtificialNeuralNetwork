package cz.muni.fi.namesny.matrixutils;

import java.util.Random;
import java.util.function.Function;

/**
 *
 */
public class Utils {

    /**
     * Creates and initializes new matrix
     * @param numRows Number of rows in the new matrix
     * @param numCols Number of columns in the new matrix
     * @return New matrix where each element is 0
     */
    public static double[][] initializeMatrix(int numRows, int numCols, Random random) {
        double[][] result = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                if (random != null) {
                    result[i][j] = random.nextDouble();
                } else{
                    result[i][j] = 0.0d;
                }
            }
        }

        return result;
    }

    /**
     * Creates and initializes new vector
     * @param length Size of the vector
     * @return New vector where each element is 0
     */
    public static double[] initializeVector(int length, Random random) {
        double[] result = new double[length];

        for (int i = 0; i < length; i++) {
            if (random != null) {
                result[i] = random.nextDouble();
            } else {
                result[i] = 0.0d;
            }
        }

        return result;
    }

    /**
     * Gets the dimensions of the input matrix
     * @param matrix 2D array representing matrix
     * @return Array with the dimensions of the input matrix
     */
    public static int[] getDimensions(double[][] matrix) {
        int[] dimensions = new int[2];

        dimensions[0] = matrix.length;
        dimensions[1] = matrix[0].length;

        return dimensions;
    }


}
