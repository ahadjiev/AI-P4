/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the input layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2, 0.1, 0.1], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.1, 0.5, 0.2], it should return 3. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{
		Double maxOutcome = -1.0;
		int result = -1;

		// Import the attribute values of the instance into the input layer
		// We don't change the bias node
		for (int i = 0; i < inst.attributes.size(); i++){
			this.inputNodes.get(i).setInput(inst.attributes.get(i));
		}

		// Update the input node to the parent lists of hidden nodes
		// We don't include the bias node in the hidden layer
		for (int i = 0; i < this.hiddenNodes.size() - 1; i++){
			ArrayList<NodeWeightPair> currentParent = this.hiddenNodes.get(i).parents;
			for (int k = 0; k < currentParent.size(); k++){
				currentParent.get(k).node = this.inputNodes.get(k);
			}
			// After getting the new parent list, we then update the outcome values
			hiddenNodes.get(i).calculateOutput();
		}

		// Update the parent lists of output nodes
		for (int i = 0; i < this.outputNodes.size(); i++){
			Node currentNode = this.outputNodes.get(i);
			ArrayList<NodeWeightPair> currentParent = currentNode.parents;

			for (int k = 0; k < currentParent.size(); k++){
				currentParent.get(k).node = this.hiddenNodes.get(k);
			}
			// After getting the new parent list, we then update the outcome values
			outputNodes.get(i).calculateOutput();

			double finalOutput = currentNode.getOutput();
			double roundedOutput = (double) Math.round(finalOutput * 10) / 10;


			// Keep tracking the max of the outcome of output nodes
			// Make sure to use the higher digit to break tie
			//*
			if (maxOutcome <= roundedOutput){
				maxOutcome = roundedOutput;
				result = i;
			}
			/*/

			if (maxOutcome <= finalOutput){
				maxOutcome = finalOutput;
				result = i;
			}
			 */
		}
		return result;
	}





	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		// TODO: add code here
		double output;
		double teacher = 0.0;
		double error;
		double value = 0;
		double adjustedWeight = 0;

		// TODO: add code here


		//        //these matrices store the weights from each node in the graph
		Double[] errors = new Double[this.outputNodes.size()];
		Double[] gDerivs = new Double[this.outputNodes.size()];
		Double[][] hiddens = new Double[this.hiddenNodes.size()][this.outputNodes.size()];
		Double[][] inputs = new Double[this.inputNodes.size()][this.hiddenNodes.size()];
		//int g = gPrimesOut.length;

		for (int y = 0; y < this.maxEpoch; y++){

			for (Instance example : this.trainingSet){
				// Forward pass which is updating information from output layer
				this.calculateOutputForInstance(example);
				//calls the calculateOutput on nodes
				// After obtaining the outcomes by the current weights
				// this loop obtains the initial errors in output nodes
				for (int x = 0; x < this.outputNodes.size(); x++){
					errors[x] = (double) example.classValues.get(x) 
							- this.outputNodes.get(x).getOutput();
				}

				// Backward propagation of the error -> hidden layer to output layer
				for (int h = 0; h < this.hiddenNodes.size(); h++){


					Node currentHiddenNode = this.hiddenNodes.get(h);

					double currentOutput = currentHiddenNode.getOutput();
					//iterate through the errors obtained above

					for (int l = 0; l < this.outputNodes.size(); l ++){
						Node currentOutNode = this.outputNodes.get(l);
						double gPrime =(double) (Math.exp(currentOutNode.getSum()))/((1+Math.exp(currentOutNode.getSum())*(1+Math.exp(currentOutNode.getSum()))));//currentOutNode.getSum() <= 0 ? 0 : 1;
						//store this value, see if it decreases runtime
						gDerivs[l] = gPrime;
						// Delta_values are stored in this matrix = initial * derivative * activation * rate
						hiddens[h][l] = errors[l] * this.learningRate * gPrime * currentOutput;
					}
				}

				// Backward updating the accumulatedError (input layer to hidden layer)
				for (int i = 0; i < this.inputNodes.size(); i++){
					// Local variables on the input layer
					Node currentInputNode = this.inputNodes.get(i);
					double currentInput = currentInputNode.getOutput();


					for (int h = 0; h < this.hiddenNodes.size(); h++){
						// Computing the summation
						double summation = 0.0;
						Node currentHiddenNode = this.hiddenNodes.get(h);
						double gPrimeHidden =(double) (Math.exp(currentHiddenNode.getSum()))/((1+Math.exp(currentHiddenNode.getSum())*(1+Math.exp(currentHiddenNode.getSum()))));//currentOutNode.getSum() <= 0 ? 0 : 1;

						for (int o = 0; o < this.outputNodes.size(); o++){
							Node currentOutNode = this.outputNodes.get(o);
							double gPrimeOut = gDerivs[o];
							//double gPrimeOut = (double) (Math.exp(currentOutNode.getSum()))/((1+Math.exp(currentOutNode.getSum())*(1+Math.exp(currentOutNode.getSum()))));//currentOutNode.getSum() <= 0 ? 0 : 1;;
							summation += errors[o] * gPrimeOut 
									* this.outputNodes.get(o).parents.get(h).weight;
						}

						inputs[i][h] = this.learningRate * gPrimeHidden 
								* currentInput * summation;
					}
				}

				// Updating the weights from input to the hidden layer
				// The size - 1 here is to avoid the bias node
				for (int h = 0; h < this.hiddenNodes.size() - 1; h++){
					List<NodeWeightPair> parents = this.hiddenNodes.get(h).parents;

					for (int p = 0; p < parents.size(); p++){
						parents.get(p).weight += inputs[p][h];
					}
				}

				// Updating the weights from hidden to the output layer
				for (int o = 0; o < this.outputNodes.size(); o++){
					List<NodeWeightPair> parents = this.outputNodes.get(o).parents;
					for (int p = 0; p < parents.size(); p++){
						parents.get(p).weight += hiddens[p][o];
					}
				}
			}
		}
	}




}
