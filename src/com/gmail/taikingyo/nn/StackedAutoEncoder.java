package com.gmail.taikingyo.nn;

public class StackedAutoEncoder {
	private AutoEncoder[] aes;

	public StackedAutoEncoder(int[] unitN) {
		aes = new AutoEncoder[unitN.length - 1];
		
		for(int i = 0; i < unitN.length - 1; i++) {
			aes[i] = new AutoEncoder(unitN[i], unitN[i + 1]);
		}
	}
	
	public void preTrain(float[][] data, int epoch, int batchSize, float learnRate, float noiseRate, float weightDecay) {
		float[][] trainData = data.clone();
		
		for(int i = 0; i < aes.length; i++) {
			aes[i].train(trainData, epoch, batchSize, learnRate, noiseRate, weightDecay);
			for(int j = 0; j < data.length; j++) trainData[j] = aes[i].encode(trainData[j]);
		}
	}
	
	public float[][][] getWeight() {
		float[][][] weight = new float[aes.length][][];
		for(int l = 0; l < weight.length; l++) weight[l] = aes[l].getWeight();
		
		return weight;
	}
}
