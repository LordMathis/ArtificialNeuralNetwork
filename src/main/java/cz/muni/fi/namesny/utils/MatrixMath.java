package cz.muni.fi.namesny.utils;

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

        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                matrix[i][j] = matrix[i][j] * scalar;
            }
        }

        return matrix;
    }

    public static double[] multiply(double scalar, double[] vector) {

        for (int i = 0; i < vector.length; i++) {
            vector[i] = vector[i] * scalar;
        }

        return vector;
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


        for (int i = 0; i < dimsA[0]; i++) {
            for (int j = 0; j < dimsA[1]; j++) {
                matrixA[i][j] += matrixB[i][j];
            }
        }

        return matrixA;

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

        for (int i = 0; i < vectorA.length; i++) {
            vectorA[i] += vectorB[i];
        }

        return vectorA;
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

        for (int i = 0; i < vectorA.length; i++) {
            vectorA[i] *= vectorB[i];
        }

        return vectorA;
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

        for (int i = 0; i < vectorA.length; i++) {
            vectorA[i] -= vectorB[i];
        }

        return vectorA;
    }

    public static double[][] transpose(double[][] matrix) {
        int[] dimensions = MatrixUtils.getDimensions(matrix);
        double[][] resultMatrix = new double[dimensions[1]][dimensions[0]];

        for (int i = 0; i < dimensions[0]; i++)
            for (int j = 0; j < dimensions[1]; j++)
                resultMatrix[j][i] = matrix[i][j];

        return resultMatrix;
    }
}
