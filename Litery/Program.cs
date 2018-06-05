//przygotować dane spoza zestawy uczącego

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using FANNCSharp;
#if FANN_FIXED
using FANNCSharp.Fixed;
using DataType = System.Int32;
#elif FANN_DOUBLE
using FANNCSharp.Double;
using DataType = System.Double;
#else
using FANNCSharp.Float;
using DataType = System.Single;
#endif
namespace LiteryTrain
{

    class LiteryTrain
    {

        static string literydata = @"LiteryData.data";
        static string literynet = @"LiteryNet.net";
        static string literytest = @"LiteryTest.txt";


        static readonly string filePath = AppDomain.CurrentDomain.BaseDirectory;

        static string newfilenameliterydata = Path.GetFullPath(Path.Combine(filePath, @"..\..\..\")) + literydata;


        static string newfilenameliterynet = Path.GetFullPath(Path.Combine(filePath, @"..\..\..\")) + literynet;


        static string newfilenameliterytest = Path.GetFullPath(Path.Combine(filePath, @"..\..\..\")) + literytest;


        static int PrintCallback(NeuralNet net, TrainingData train, uint max_epochs, uint epochs_between_reports,
            float desired_error, uint epochs, Object user_data)
        {
            Console.WriteLine(String.Format("Epochs     " + String.Format("{0:D}", epochs).PadLeft(8) +
                                            ". Current Error: " + net.MSE));
            return 0;
        }

        private static void LiteryTest()
        {
            Console.WriteLine("\nLitery test started.");

            const float learning_rate = 0.7f;
            const uint num_layers = 6;
            const uint num_input = 225;
            const uint num_hidden1 = 80;
            const uint num_hidden2 = 60;
            const uint num_hidden3 = 40;
            const uint num_hidden4 = 10;
            const uint num_output = 5;
            const float desired_error = 0.0002f;
            const uint max_iterations = 1000000;
            const uint iterations_between_reports = 10000;

            Console.WriteLine("\nCreating network.");


            using (NeuralNet net = new NeuralNet(NetworkType.LAYER, num_layers, num_input, num_hidden1, num_hidden2,
                num_hidden3, num_hidden4, num_output))
            {
                net.LearningRate = learning_rate;

                net.ActivationSteepnessHidden = 1.0F;
                net.ActivationSteepnessOutput = 1.0F;

                net.ActivationFunctionHidden = ActivationFunction.SIGMOID_SYMMETRIC_STEPWISE;
                net.ActivationFunctionOutput = ActivationFunction.SIGMOID_STEPWISE;

                // Output network type and parameters
                Console.Write("\nNetworkType                         :  ");
                switch (net.NetworkType)
                {
                    case NetworkType.LAYER:
                        Console.WriteLine("LAYER");
                        break;
                    case NetworkType.SHORTCUT:
                        Console.WriteLine("SHORTCUT");
                        break;
                    default:
                        Console.WriteLine("UNKNOWN");
                        break;
                }

                net.PrintParameters();

                Console.WriteLine("\nTraining network.");


                using (TrainingData data = new TrainingData())
                {

                    if (data.ReadTrainFromFile(newfilenameliterydata))
                    {
                        // Initialize and train the network with the data
                        net.InitWeights(data);

                        Console.WriteLine("Max Epochs " + max_iterations +
                                          ". Desired Error: " + desired_error);
                        net.SetCallback(PrintCallback, null);
                        net.TrainOnData(data, max_iterations, iterations_between_reports, desired_error);

                        Console.WriteLine("\nTesting network.");

                        Console.WriteLine("\nSaving network.");

                        net.Save(
                            newfilenameliterynet);
                        Console.WriteLine("\n Znaki test completed.");
                        CalculateOutputs(net);
                    }
                }
            }
        }

        public static void CalculateOutputs(NeuralNet net)
        {

            using (TextReader reader = File.OpenText(newfilenameliterytest))
            {
                int n = Array.ConvertAll(reader.ReadLine().Split(' '), Int32.Parse).First();


                for (int i = 0; i < n; i++)
                {
                    var inputsChar = reader.ReadLine().Split(' ');
                    var outputsChar = reader.ReadLine().Split(' ');
                    var inputs = new float[225];
                    var outputs = new float[5];
                    for (int j = 0; j < inputsChar.Length; j++)
                    {
                        inputs[j] = Convert.ToSingle(inputsChar[j]);
                    }
                    for (int j = 0; j < outputsChar.Length; j++)
                    {
                        outputs[j] = Convert.ToSingle(outputsChar[j]);
                    }

                    float[] calc_out = net.Run(inputs);
                    for (int j = 0; j < 15; j++)
                    {
                        for (int k = 0; k < 15; k++)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.Write(inputs[15 * j + k] == 0 ? ' ' : '+');
                            Console.ResetColor();
                        }
                        Console.WriteLine();
                    }

                    Console.WriteLine();

                    Console.WriteLine($"{outputs[0]} => {calc_out[0]} " +
                                      $"\t {outputs[1]} =>{calc_out[1]} " +
                                      $"\t{outputs[2]} => {calc_out[2]} " +
                                      $"\t{outputs[3]} => {calc_out[3]} " +
                                      $"\t{outputs[4]} => {calc_out[4]}");
                    Console.WriteLine("<------------------------------------------------>");
                }
            }
        }

        static int Main(string[] args)
        {

            LiteryTest();

            Console.ReadKey();
            return 0;
        }

        static DataType FannAbs(DataType value)
        {
            return (((value) > 0) ? (value) : -(value));
        }
    }
}


