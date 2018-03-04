package com.gmail.taikingyo.nn;

public class StackedAutoEncoder {
	private AutoEncoder[] aes;
	
	public StackedAutoEncoder(int[] unitN) {
		this(unitN, 4);
	}

	public StackedAutoEncoder(int[] unitN, int paraN) {
		aes = new AutoEncoder[unitN.length - 1];
		
		for(int i = 0; i < unitN.length - 1; i++) {
			aes[i] = new AutoEncoder(unitN[i], unitN[i + 1], paraN);
		}
	}
	
	public void preTrain(double[][] data, int trainN, int batchSize, double learnRate, double noiseRate, double weightDecay) {
		double[][] trainData = new double[data.length][];
		
		for(int i = 0; i < aes.length; i++) {
			if(i == 0) trainData = data;
			else for(int j = 0; j < data.length; j++) trainData[j] = aes[i].test(trainData[j]);
			aes[i].train(trainData, trainN, batchSize, learnRate, noiseRate, weightDecay);
		}
	}
	
	public double[][][] getWeight() {
		double[][][] weight = new double[aes.length][][];
		for(int l = 0; l < weight.length; l++) weight[l] = aes[l].getWeight();
		
		return weight;
	}
}
