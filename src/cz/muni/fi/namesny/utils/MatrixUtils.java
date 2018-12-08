package cz.muni.fi.namesny.utils;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 */
public class MatrixUtils {

    /**
     * Creates and initializes new matrix
     * @param numRows Number of rows in the new matrix
     * @param numCols Number of columns in the new matrix
     * @return New matrix where each element is 0
     */
    public static double[][] initializeMatrix(int numRows, int numCols, boolean random) {
        double[][] result = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                if (random) {

                    double range = Math.sqrt(6/((double)numRows + (double)numCols));
                    result[i][j] = getRandomFromRange(range);

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

    public static double getRandomFromRange(double range) {

        return ThreadLocalRandom.current().nextDouble(-range, range);

    }

    public static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            System.out.print("[");

            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }

            System.out.println("]");
        }
    }

    public static void printVector(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            System.out.println("[" + vector[i] + "]");
        }
    }

    public static int argmax(double[] vector) {
        int max = 0;

        for (int i = 0; i < vector.length; i++) {
            if (vector[i] > vector[max]) {
                max = i;
            }
        }

        return max;
    }


}
