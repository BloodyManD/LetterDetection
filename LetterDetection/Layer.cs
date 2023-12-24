using System;

namespace LetterDetection
{
    [Serializable]
    public class Layer
    {
        public int size;
        public double[] neurons;
        public double[] biases;
        public double[][] weights;

        public Layer(int size, int nextSize) {
            this.size = size;
            neurons = new double[size];
            biases = new double[size];
            weights = new double[size][];
            for (int i = 0; i < size; i++)
            {
                weights[i] = new double[nextSize];
            }
        }
    }
}