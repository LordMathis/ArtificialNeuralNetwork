package cz.muni.fi.namesny.utils;

import java.util.Arrays;

/**
 *
 */
public class MathUtils {

    /**
     * Multiplies matrix with vector
     * @param matrix
     * @param vector
     * @return Vector that is matrix*vector
     */
    public static double[] multiply(double[][] matrix, double[] vector) {

        int[] dimsMatrix = MatrixUtils.getDimensions(matrix);

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

        int[] dimsA = MatrixUtils.getDimensions(matrixA);
        int[] dimsB = MatrixUtils.getDimensions(matrixB);

        if (dimsA[1] != dimsB[0]) {
            throw new IllegalArgumentException("Matrix dimensions do not match!");
        }

        double[][] resultMatrix = MatrixUtils.initializeMatrix(dimsA[0], dimsB[1], false);

        for (int i = 0; i < dimsA[0]; i++) {
            for (int j = 0; j < dimsB[1]; j++) {
                for (int k = 0; k < dimsA[1]; k++) {
                    resultMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        return resultMatrix;

    }

    public static double[][] multiply(double[] vectorA, double[] vectorB) {

        double[][] result = new double[vectorA.length][vectorB.length];

        for (int i = 0; i < vectorA.length; i++) {
            for (int j = 0; j < vectorB.length; j++) {
                result[i][j] = vectorA[i] * vectorB[j];
            }
        }

        return  result;
    }

    public static double[][] multiply(double scalar, double[][] matrix) {

        int[] dims = MatrixUtils.getDimensions(matrix);
        double[][] result = new double[dims[0]][dims[1]];

        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }

        return result;
    }

    public static double[] multiply(double scalar, double[] vector) {
        double[] result = new double[vector.length];

        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * scalar;
        }

        return result;
    }

    /**
     * Sums two matrices
     * @param matrixA
     * @param matrixB
     * @return The sum of two matrices
     */
    public static double[][] sum(double[][] matrixA, double[][] matrixB) {
        int[] dimsA = MatrixUtils.getDimensions(matrixA);
        int[] dimsB = MatrixUtils.getDimensions(matrixB);

        if (!Arrays.equals(dimsA, dimsB)) {
            throw new IllegalArgumentException("Matrix dimensions do not match!");
        }

        double[][] resultMatrix = MatrixUtils.initializeMatrix(dimsA[0], dimsA[1], false);

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

    /**
     * Implements elementwise vector multiplication
     * @param vectorA
     * @param vectorB
     * @return vector v such that for each i: v[i]=a[i]*b[i]
     */
    public static double[] hadamard(double[] vectorA, double[] vectorB) {

        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double result[] = new double[vectorA.length];

        for (int i = 0; i < vectorA.length; i++) {
            result[i] = vectorA[i] * vectorB[i];
        }

        return result;
    }

    /**
     * Substracts one vector from the other
     * @param vectorA
     * @param vectorB
     * @return the difference of two vectors
     */
    public static double[] subtract(double[] vectorA, double[] vectorB) {

        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double[] result = new double[vectorA.length];

        for (int i = 0; i < vectorA.length; i++) {
            result[i] = vectorA[i] - vectorB[i];
        }

        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        int[] dimensions = MatrixUtils.getDimensions(matrix);
        double [][] resultMatrix = new double[dimensions[1]][dimensions[0]];

        for (int i = 0; i < dimensions[0]; i++)
            for (int j = 0; j < dimensions[1]; j++)
                resultMatrix[j][i] = matrix[i][j];

        return resultMatrix;
    }
}
