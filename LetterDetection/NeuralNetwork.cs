using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace LetterDetection
{
    public class NeuralNetwork
    {
    private double learningRate;
    private Layer[] layers;
    private Func<double, double> activation;
    private Func<double, double> derivative;

    public NeuralNetwork(double learningRate, Func<double, double> activation, Func<double, double> derivative, params int[] sizes)
    {
        this.learningRate = learningRate;
        this.activation = activation;
        this.derivative = derivative;
        layers = new Layer[sizes.Length];
        Random rnd = new Random(42);
        for (int i = 0; i < sizes.Length; i++) {
            int nextSize = 0;
            if(i < sizes.Length - 1) nextSize = sizes[i + 1];
            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++)
            {
                layers[i].biases[j] = rnd.NextDouble() * 2.0 - 1.0;
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = rnd.NextDouble() * 2.0 - 1.0;
                }
            }
        }
    }

    public NeuralNetwork()
    {
    }

    public double[] FeedForward(double[] inputs) {
        System.Array.Copy(inputs, 0, layers[0].neurons, 0, inputs.Length);
        for (int i = 1; i < layers.Length; i++)  {
            Layer l = layers[i - 1];
            Layer l1 = layers[i];
            for (int j = 0; j < l1.size; j++) {
                l1.neurons[j] = 0;
                for (int k = 0; k < l.size; k++) {
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                }
                l1.neurons[j] += l1.biases[j];
                l1.neurons[j] = activation(l1.neurons[j]);
            }
        }
        return layers[layers.Length - 1].neurons;
    }

    public void Backpropagation(double[] targets) {
        double[] errors = new double[layers[layers.Length - 1].size];
        for (int i = 0; i < layers[layers.Length - 1].size; i++) {
            errors[i] = targets[i] - layers[layers.Length - 1].neurons[i];
        }
        for (int k = layers.Length - 2; k >= 0; k--) {
            Layer l = layers[k];
            Layer l1 = layers[k + 1];
            double[] errorsNext = new double[l.size];
            double[] gradients = new double[l1.size];
            for (int i = 0; i < l1.size; i++) {
                gradients[i] = errors[i] * derivative(layers[k + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            double[][] deltas = new double[l1.size][];
            for (int i = 0; i < l1.size; i++)
            {
                deltas[i] = new double[l.size];
            }
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }
            for (int i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }
            errors = new double[l.size];
            System.Array.Copy(errorsNext, 0, errors, 0, l.size);
            double[][] weightsNew = new double[l.weights.Length][];
            for (int i = 0; i < l.weights.Length; i++)
            {
                weightsNew[i] = new double[l.weights[0].Length];
            }
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (int i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
        }
    }

    public void SaveNeuralNetwork(string filename)
    {
        BinaryFormatter formatter = new BinaryFormatter();
        using (FileStream fs = new FileStream("C:\\Users\\ngavr\\RiderProjects\\LetterDetection\\" + filename,
                   FileMode.OpenOrCreate)) 
        {
            formatter.Serialize(fs, layers);
                
            Console.WriteLine("Состояние нейросети сохранено");
        }
    }
    
    public void LoadNeuralNetwork(string filename, double passedLearningRate, Func<double, double> passedActivation, Func<double, double> passedDerivative)
    {
        this.learningRate = passedLearningRate;
        this.activation = passedActivation;
        this.derivative = passedDerivative;
        BinaryFormatter formatter = new BinaryFormatter();
        using (FileStream fs = new FileStream("C:\\Users\\ngavr\\RiderProjects\\LetterDetection\\" + filename,
                   FileMode.OpenOrCreate)) 
        {
            layers = (Layer[])formatter.Deserialize(fs);
                
            Console.WriteLine("Состояние нейросети загружено из файла");
        }
    }
    
    }
}