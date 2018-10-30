package cz.muni.fi.namesny.matrixutils;

import java.util.Arrays;

/**
 *
 */
public class MatrixMath {

    /**
     * Multiplies matrix with vector
     * @param matrix
     * @param vector
     * @return Vector that is matrix*vector
     */
    public static double[] multiply(double[][] matrix, double[] vector) {

        int[] dimsMatrix = Utils.getDimensions(matrix);

        if (dimsMatrix[1] != vector.length) {
            throw new IllegalArgumentException("Matrix and vector dimensions do not match");
        }

        double[] result = new double[dimsMatrix[0]];

        for (int i = 0; i < dimsMatrix[0]; i++) {
            double value = 0;
            for (int j = 0; j < dimsMatrix[1]; j++) {
                value += matrix[i][j] * vector[j];
            }
            result[i] = value;
        }

        return result;
    }

    /**
     * Multiplies two matrices. The two input matrices have to have compatible dimensions.
     * @param matrixA Two dimensional array representing input matrix A.
     * @param matrixB Two dimensional array representing input matrix B.
     * @return Matrix C such that C = A*B
     */
    public static double[][] multiply(double[][] matrixA, double[][] matrixB) {

        int[] dimsA = Utils.getDimensions(matrixA);
        int[] dimsB = Utils.getDimensions(matrixB);

        if (dimsA[1] != dimsB[0]) {
            throw new IllegalArgumentException("Matrix dimensions do not match!");
        }

        double[][] resultMatrix = Utils.initializeMatrix(dimsA[0], dimsB[1], false);

        for (int i = 0; i < dimsA[0]; i++) {
            for (int j = 0; j < dimsB[1]; j++) {
                for (int k = 0; k < dimsA[1]; k++) {
                    resultMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        return resultMatrix;

    }

    /**
     * Sums two matrices
     * @param matrixA
     * @param matrixB
     * @return The sum of two matrices
     */
    public static double[][] sum(double[][] matrixA, double[][] matrixB) {
        int[] dimsA = Utils.getDimensions(matrixA);
        int[] dimsB = Utils.getDimensions(matrixB);

        if (!Arrays.equals(dimsA, dimsB)) {
            throw new IllegalArgumentException("Matrix dimensions do not match!");
        }

        double[][] resultMatrix = Utils.initializeMatrix(dimsA[0], dimsA[1], false);

        for (int i = 0; i < dimsA[0]; i++) {
            for (int j = 0; j < dimsA[1]; j++) {
                resultMatrix[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }

        return resultMatrix;

    }

    /**
     * Sums two vectors
     * @param vectorA
     * @param vectorB
     * @return the sum of two vectors
     */
    public static double[] sum(double[] vectorA, double[] vectorB) {

        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double[] result = new double[vectorA.length];

        for (int i = 0; i < vectorA.length; i++) {
            result[i] = vectorA[i] + vectorB[i];
        }

        return result;
    }

}
