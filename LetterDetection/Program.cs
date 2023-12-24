using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace LetterDetection
{
    static class Constants
    {
        public const int seed = 42;
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Letters();
            Func<double, double> sigmoid = x => 1 / (1 + Math.Exp(-x));
            Func<double, double> derivativeSigmoid = y => y * (1 - y);
            
            NeuralNetwork myNeuralNetwork = new NeuralNetwork();
            
            // TrainNeuralNetwork(out myNeuralNetwork, 1000);
            // myNeuralNetwork.SaveNeuralNetwork("1000Epochs5Layers25NeuronsEach.dat");
            myNeuralNetwork.LoadNeuralNetwork("1000Epochs.dat", 0.001, sigmoid, derivativeSigmoid);
            TestNeuralNetworkDetectionAbility(myNeuralNetwork);
        }

        private static void Letters()
        { 
            
            
            
        }

        // Не работает из-за невозможности сериализовать делегаты
        private static void SaveNeuralNetwork(NeuralNetwork neuralNetwork, string filename)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream("C:\\Users\\ngavr\\RiderProjects\\LetterDetection\\" + filename,
                       FileMode.OpenOrCreate)) 
            {
                formatter.Serialize(fs, neuralNetwork);
                
                Console.WriteLine("Состояние нейросети сохранено");
            }
        }
        
        // Не работает из-за невозможности сериализовать делегаты
        private static void LoadNeuralNetwork(out NeuralNetwork neuralNetwork, string filename)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream("C:\\Users\\ngavr\\RiderProjects\\LetterDetection\\" + filename,
                       FileMode.OpenOrCreate)) 
            {
                neuralNetwork = (NeuralNetwork)formatter.Deserialize(fs);
                
                Console.WriteLine("Состояние нейросети загружено из файла");
            }
        }
        // Попытка сериализовать делегаты
        // public static double Sigmoid(double x)
        // {
        //     return 1 / (1 + Math.Exp(-x));
        // }
        //
        // public static double DerivativeSigmoid(double x)
        // {
        //     return x * (1 - x);
        // }
        
        private static void TrainNeuralNetwork(out NeuralNetwork neuralNetwork, int epochs)
        {
            Func<double, double> sigmoid = x => 1 / (1 + Math.Exp(-x));
            Func<double, double> derivativeSigmoid = y => y * (1 - y);
            neuralNetwork = new NeuralNetwork(0.001, sigmoid, derivativeSigmoid, 784, 512, 128, 32, 10);
            // neuralNetwork = new NeuralNetwork(0.001, sigmoid, derivativeSigmoid, 784, 25, 25, 25, 25, 25, 10);

            int samples = 60000;
            // int samples = 600;
            Bitmap[] images = new Bitmap[samples];
            int[] digits = new int[samples];
            var dirs = Directory.EnumerateDirectories(
                "C:/Users/ngavr/RiderProjects/LetterDetection/dataset/0-9/train/");
            List<string> imageFiles = new List<string>();
            foreach (var directory in dirs)
            {
                imageFiles.AddRange(Directory.EnumerateFiles(directory, "*.jpg"));
            }
            int ii = 0;
            foreach (var imageFile in imageFiles)
            {
                images[ii] = new Bitmap(imageFile);
                digits[ii] = imageFile[63] - '0';
                ii++;
                // if (ii > 599)
                // {
                //     break;
                // }
            }
            Console.WriteLine("Получили пути к изображениям");
            
            
            double[][] inputs = new double[samples][];
            for (int i = 0; i < samples; i++)
            {
                inputs[i] = new double[784];
            }
            Console.WriteLine("Выделили память для входного массива");
            
            for (int i = 0; i < samples; i++)
            {
                for (int x = 0; x < 28; x++)
                {
                    for (int y = 0; y < 28; y++)
                    {
                        inputs[i][x + y * 28] = (images[i].GetPixel(x, y).ToArgb() & 0xff) / 255.0;
                    }
                }
            }
            Console.WriteLine("Получили числовое представление изображений");

            Console.WriteLine("epoch;correct;error");
            // обучение 
            // epochs = 1000;
            Random rnd = new Random(Constants.seed);
            for (int i = 0; i < epochs; i++)
            {
                int right = 0;
                double errorSum = 0;
                int batchSize = 100;
                for (int j = 0; j < batchSize; j++)
                {
                    int imgIndex = rnd.Next(0, samples);
                    double[] targets = new double[10];
                    int digit = digits[imgIndex];
                    targets[digit] = 1;

                    double[] outputs = neuralNetwork.FeedForward(inputs[imgIndex]);
                    int maxDigit = 0;
                    double maxDigitWeight = -1;
                    for (int k = 0; k < 10; k++)
                    {
                        if(outputs[k] > maxDigitWeight) {
                            maxDigitWeight = outputs[k];
                            maxDigit = k;
                        }
                    }
                    if(digit == maxDigit) right++;
                    for (int k = 0; k < 10; k++) {
                        errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                    }
                    neuralNetwork.Backpropagation(targets);
                }
                // Console.WriteLine("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
                Console.WriteLine(i+";"+right+";"+errorSum);
            }
        }

        private static void TestNeuralNetworkDetectionAbility(NeuralNetwork neuralNetwork)
        {
            Random rnd = new Random(Constants.seed);
            
            int testSamples = 10000;
            Bitmap[] testImages = new Bitmap[testSamples];
            int[] testDigits = new int[testSamples];
            var testDirs = Directory.EnumerateDirectories(
                "C:\\Users\\ngavr\\RiderProjects\\LetterDetection\\dataset\\0-9\\test\\");
            List<string> testImageFiles = new List<string>();
            foreach (var directory in testDirs)
            {
                testImageFiles.AddRange(Directory.EnumerateFiles(directory, "*.jpg"));
            }
            int qq = 0;
            foreach (var imageFile in testImageFiles)
            {
                testImages[qq] = new Bitmap(imageFile);
                testDigits[qq] = imageFile[62] - '0';
                qq++;
            }
            Console.WriteLine("Получили пути к тестовым изображениям");

            double[][] testInputs = new double[testSamples][];
            for (int i = 0; i < testSamples; i++)
            {
                testInputs[i] = new double[784];
            }
            Console.WriteLine("Выделили память для тестового массива");
            
            for (int i = 0; i < testSamples; i++)
            {
                for (int x = 0; x < 28; x++)
                {
                    for (int y = 0; y < 28; y++)
                    {
                        testInputs[i][x + y * 28] = (testImages[i].GetPixel(x, y).ToArgb() & 0xff) / 255.0;
                    }
                }
            }
            Console.WriteLine("Получили числовое представление для тестовых изображений");

            int testBatch = 100;
            for (int i = 0; i < testBatch; i++)
            {
                int testImgIndex = rnd.Next(0, testSamples); // выберем testBatch случайных изображений для проверки
                int digit = testDigits[testImgIndex];
                Console.WriteLine(testImageFiles[testImgIndex]);
                double[] outputs = neuralNetwork.FeedForward(testInputs[testImgIndex]);
                int maxDigit = 0;
                double maxDigitWeight = -1;
                for (int k = 0; k < 10; k++)
                {
                    if(outputs[k] > maxDigitWeight) {
                        maxDigitWeight = outputs[k];
                        maxDigit = k;
                    }
                }
                Console.WriteLine("На изображении "+ digit +". Нейросеть увидела " + maxDigit + ". Вес " + maxDigitWeight);
            }
        }
    }
}