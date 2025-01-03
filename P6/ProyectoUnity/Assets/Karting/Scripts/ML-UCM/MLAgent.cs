using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows;

public class StandarScaler {
    private float[] mean;
    private float[] std;

    public StandarScaler(string serieliced) {
        string[] lines = serieliced.Split("\n");
        string[] meanStr = lines[0].Split(",");
        string[] stdStr = lines[1].Split(",");
        mean = new float[meanStr.Length];
        std = new float[stdStr.Length];
        for (int i = 0; i < meanStr.Length; i++)
        {
            mean[i] = float.Parse(meanStr[i], System.Globalization.CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < stdStr.Length; i++)
        {
            std[i] = float.Parse(stdStr[i], System.Globalization.CultureInfo.InvariantCulture);
            std[i] = Mathf.Sqrt(std[i]);
        }
    }

    // TODO Implement the standar scaler.
    public float[] Transform(float[] a_input) {
        float[] transformed = new float[a_input.Length];
        for (int i = 0; i < a_input.Length; i++) {
            transformed[i] = (a_input[i] - mean[i]) / std[i];
        }
        return transformed;
    }
}

public class MLPParameters {
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers) {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++) {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++) {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols) {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v) {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff() {
        return coeficients;
    }

    public void CreateIntercept(int i, int row) {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v) {
        intercepts[i][row] = v;
    }

    public List<float[]> GetInter() {
        return intercepts;
    }
}

public class MLPModel {
    MLPParameters mlpParameters;
    int[] indicesToRemove;
    StandarScaler standarScaler;

    public MLPModel(MLPParameters p, int[] itr, StandarScaler ss) {
        mlpParameters = p;
        indicesToRemove = itr;
        standarScaler = ss;
    }

   
    private float sigmoid(float z) {
        return 1f / (1f + Mathf.Exp(-z));
    }


    public bool FeedForwardTest(string csv, float accuracy, float aceptThreshold, out float acc) {
        Tuple<List<Parameters>, List<Labels>> tuple = Record.ReadFromCsv(csv, true);
        List<Parameters> parameters = tuple.Item1;
        List<Labels> labels = tuple.Item2;
        int goals = 0;
        for(int i = 0; i < parameters.Count; i++) {
            float[] input = parameters[i].ConvertToFloatArrat();
            float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
            a_input = standarScaler.Transform(a_input);
            var outputs = FeedForward(a_input);
            if(i == 0)
               Debug.Log("a");
            Labels label = Predict(outputs.Item1);
            if (label == labels[i])
                goals++;
        }

        acc = goals / ((float)parameters.Count);
        
        float diff = Mathf.Abs(acc - accuracy);
        Debug.Log("Accuracy " + acc + " Accuracy espected " + accuracy + " goalds " + goals + " Examples " + parameters.Count + " Difference "+diff);
        return diff < aceptThreshold;
    }

    public float[] ConvertPerceptionToInput(Perception p, Transform transform) {
        Parameters parameters = Record.ReadParameters(9, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
        a_input = standarScaler.Transform(a_input);
        return a_input;
    }

	// TODO Implement FeedForward
	//   public float[] FeedForward(float[] a_input) {
	//       float[] activations = a_input;

	//       // Iterar sobre las capas del modelo

	//       for (int layer = 0; layer < mlpParameters.GetCoeff().Count; layer++) {
	//           float[,] weights = mlpParameters.GetCoeff()[layer];
	//           float[] biases = mlpParameters.GetInter()[layer];
	//           float[] nextActivations = new float[weights.GetLength(1)];

	//           // Calcular activaciones de la capa actual
	//           for (int j = 0; j < nextActivations.Length; j++) {
	//               float z = biases[j];
	//               for (int i = 0; i < weights.GetLength(0); i++) {
	//                   z += activations[i] * weights[i, j];
	//               }
	//               // Aplicar la función de activación (sigmoide)
	//               nextActivations[j] = sigmoid(z);
	//           }
	//           activations = nextActivations;
	//       }
	//       return activations;
	//}
	public (List<float[]>, List<float[]>) FeedForward(float[] input)
	{

		List<float[]> a = new List<float[]>();
		List<float[]> z = new List<float[]>();

        // Agregar la entrada inicial
		a.Add(input);

		for (int i = 0; i < mlpParameters.GetCoeff().Count; i++)
		{
			float[,] weights = mlpParameters.GetCoeff()[i];

			// Añadir el sesgo como un elemento adicional al vector de activaciones
			float[] aWithBias = AddBiasUnit(a[i]);

			// Calcular z = a[i] @ weights.T
			float[] zLayer = new float[weights.GetLength(1)];
			Debug.Log($"Dimensiones de weights: filas={weights.GetLength(0)}, columnas={weights.GetLength(1)}");
			Debug.Log($"Tamaño de aWithBias: {aWithBias.Length}");
			Debug.Log($"Tamaño de zLayer: {zLayer.Length}");
			for (int j = 0; j < zLayer.Length; j++)
			{
                float sum = 0f;
				for (int k = 0; k < aWithBias.Length - 1; k++)
				{
                    float valor1 = aWithBias[k];
                    float valor2 = weights[k, j];

                    Debug.Log("Valor1 = " + valor1);
                    Debug.Log("Valor2 = " + valor2);
					sum +=  valor1 * valor2;
				}
				zLayer[j] = sum;
			}
			z.Add(zLayer);

			// Calcular la activación aplicando la función sigmoide
			float[] nextActivation = Sigmoid(zLayer);
			a.Add(nextActivation);
		}

		return (a, z);
	}
	private float[,] TransposeMatrix(float[,] matrix)
	{
		int rows = matrix.GetLength(0);
		int cols = matrix.GetLength(1);
		float[,] transposed = new float[cols, rows];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				transposed[j, i] = matrix[i, j];
			}
		}

		return transposed;
	}
	private float[,] MultiplyMatrices(float[,] a, float[,] b)
	{
		int rowsA = a.GetLength(0);
		int colsA = a.GetLength(1);
		int rowsB = b.GetLength(0);
		int colsB = b.GetLength(1);

		if (colsA != rowsB)
			throw new InvalidOperationException("El número de columnas de A debe coincidir con el número de filas de B.");

		float[,] result = new float[rowsA, colsB];

		for (int i = 0; i < rowsA; i++)
		{
			for (int j = 0; j < colsB; j++)
			{
				float sum = 0;
				for (int k = 0; k < colsA; k++)
				{
					sum += a[i, k] * b[k, j];
				}
				result[i, j] = sum;
			}
		}

		return result;
	}

	private float[] AddBiasUnit(float[] input)
	{
		float[] biasedInput = new float[input.Length + 1];
		biasedInput[0] = 1f; // Sesgo
		Array.Copy(input, 0, biasedInput, 1, input.Length);
		return biasedInput;
	}
	// Método para aplicar la función sigmoide a cada elemento de un vector
	private float[] Sigmoid(float[] z)
	{
		float[] result = new float[z.Length];
		for (int i = 0; i < z.Length; i++)
		{
			result[i] = 1f / (1f + (float)Math.Exp(-z[i]));
		}
		return result;
	}

	//TODO: implement the conversion from index to actions. You may need to implement several ways of
	//transforming the data if you play in different ways. You must take into account how many classes
	//you have used, and how One Hot Encoder has encoded them and this may vary if you change the training
	//data.
	public Labels ConvertIndexToLabel(int index) {
        switch (index) {
            case 0: return Labels.NONE;
            case 1: return Labels.ACCELERATE;
            case 2: return Labels.BRAKE;
            case 3: return Labels.LEFT_ACCELERATE;
            case 4: return Labels.RIGHT_ACCELERATE;
            case 5: return Labels.LEFT_BRAKE;
            case 6: return Labels.RIGHT_BRAKE;

            default: return Labels.NONE;
        }
    }

   public Labels Predict(List<float[]> activations)
    {
        // Tomar la activación de la última capa
        float[] output = activations[activations.Count - 1];

        // Encontrar el índice del valor máximo (la clase predicha)
        int predictedIndex = Array.IndexOf(output, output.Max());

        // Convertir el índice a una etiqueta
        return (Labels)predictedIndex;
    }


    public int GetIndexMaxValue(float[] output, out float max) {
        max = output[0];
        max = output[0];
        int index = 0;
        for (int i = 1; i < output.Length; i++) {
            if (output[i] > max) {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}

public class MLAgent : MonoBehaviour {
    public enum ModelType { MLP = 0 }
    public TextAsset text;
    public ModelType model;
    public bool agentEnable;
    public int[] indexToRemove;
    public TextAsset standarScaler;
    public bool testFeedForward;
    public float accuracy;
    public TextAsset trainingCsv;


    private MLPParameters mlpParameters;
    private MLPModel mlpModel;
    private Perception perception;

    // Start is called before the first frame update
    void Start() {
        if (agentEnable) {
            string file = text.text;
            if (model == ModelType.MLP) {
                mlpParameters = LoadParameters(file);
                StandarScaler ss = new StandarScaler(standarScaler.text);
                mlpModel = new MLPModel(mlpParameters, indexToRemove, ss);
                if (testFeedForward) {
                    float acc;
                    if(mlpModel.FeedForwardTest(trainingCsv.text, accuracy, 0.025f, out acc)) {
                        Debug.Log("Test Complete!");
                    }
                    else {
                        Debug.LogError("Error: Accuracy is not the same. Accuracy in C# "+acc + " accuracy in sklearn "+ accuracy);
                    }
                }
            }
            Debug.Log("Parameters loaded " + mlpParameters);
            perception = GetComponent<Perception>();
        }
    }


	public KartGame.KartSystems.InputData AgentInput()
	{
		perception = GetComponent<Perception>();

		Labels label = Labels.NONE;
        Debug.Log("Transform = " + transform);
        Debug.Log("perception= " + perception);
        Debug.Log("MLP= " + model); 
        switch (model)
		{
			case ModelType.MLP:
				// Convertir la percepción en entrada para el modelo
				float[] X = this.mlpModel.ConvertPerceptionToInput(perception, this.transform);
				// Ejecutar FeedForward y obtener las activaciones finales
				var (activations, _) = this.mlpModel.FeedForward(X);
				float[] outputs = activations[activations.Count - 1]; // Salida de la última capa

				// Realizar predicción
				label = this.mlpModel.Predict(activations);
				break;
		}

		// Convertir la etiqueta predicha en datos de entrada para el kart
		KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);
		return input;
	}
	public static string TrimpBrackers(string val) {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val) {
        val = val.Trim();
        string[] values = val.Split(",");
        int[] result = new int[values.Length];
        for (int i = 0; i < values.Length; i++) {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length - 1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val) {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++) {
            result[i] = float.Parse(values[i], System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    public static MLPParameters LoadParameters(string file) {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
        bool coefficient = false;
        for (int i = 0; i < lines.Length; i++) {
            string line = lines[i];
            line = line.Trim();
            if (line != "") {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers") {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }
                else {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims") {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name") {
                            if (val.StartsWith("coefficient")) {
                                coefficient = true;
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else {
                                coefficient = false;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[1]);
                            }

                        }
                        else if (name == "values") {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++) {
                                if (coefficient) {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return mlpParameters;
    }
}
