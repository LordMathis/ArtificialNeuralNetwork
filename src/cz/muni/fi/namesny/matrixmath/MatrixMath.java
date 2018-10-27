package cz.muni.fi.namesny.matrixmath;

import java.util.Arrays;

/**
 *
 */
public class MatrixMath {

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

        double[][] resultMatrix = Utils.initializeMatrix(dimsA[0], dimsB[1]);

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
     *
     * @param matrixA
     * @param matrixB
     * @return
     */
    public static double[][] sum(double[][] matrixA, double[][] matrixB) {
        int[] dimsA = Utils.getDimensions(matrixA);
        int[] dimsB = Utils.getDimensions(matrixB);

        if (!Arrays.equals(dimsA, dimsB)) {
            throw new IllegalArgumentException("Matrix dimensions do not match!");
        }

        double[][] resultMatrix = Utils.initializeMatrix(dimsA[0], dimsA[1]);

        for (int i = 0; i < dimsA[0]; i++) {
            for (int j = 0; j < dimsA[1]; j++) {
                resultMatrix[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }

        return resultMatrix;

    }

}
