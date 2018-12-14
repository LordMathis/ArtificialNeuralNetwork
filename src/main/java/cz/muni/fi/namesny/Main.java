package cz.muni.fi.namesny;

import cz.muni.fi.namesny.examples.MNIST;

public class Main {

    public static void main(String[] args) {

        MNIST mnist = new MNIST();
        mnist.train();

    }
}
