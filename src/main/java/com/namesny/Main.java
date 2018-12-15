package com.namesny;

import com.namesny.examples.MNIST;

public class Main {

    public static void main(String[] args) {

        MNIST mnist = new MNIST();
        mnist.train();

    }
}
