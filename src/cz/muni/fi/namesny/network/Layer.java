package cz.muni.fi.namesny.network;

public class Layer {

    private Neuron[] neurons;

    public Layer(int numNeurons, int prevLayerSize) {

        this.neurons = new Neuron[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] = new Neuron(prevLayerSize);
        }
    }

    public double[] compute(double[] inputs) {

        double[] outputs = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].compute(inputs);
        }

        return outputs;
    }

}
