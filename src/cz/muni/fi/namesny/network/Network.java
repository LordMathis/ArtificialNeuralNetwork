package cz.muni.fi.namesny.network;

import java.util.List;

public class Network {
    private List<Layer> layers;

    public Network(List<Layer> layers) {
        this.layers = layers;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }
}
