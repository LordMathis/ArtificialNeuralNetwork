package cz.muni.fi.namesny.matrixmath;

/**
 *
 */
class Utils {

    /**
     * Creates and initializes new matrix
     * @param numRows Number of rows in the new matrix
     * @param numCols Number of columns in the new matrix
     * @return New matrix where each element is 0
     */
    public static double[][] initializeMatrix(int numRows, int numCols) {
        double[][] result = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                result[i][j] = 0.0;
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
