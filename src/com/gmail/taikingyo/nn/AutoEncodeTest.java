package com.gmail.taikingyo.nn;

public class AutoEncodeTest {
	int inputN = 784;
	int middleN = 400;
	int epoch = 10;
	int batchSize = 20;
	float learnRate = 0.01f;
	float noiseRate = 0.5f;
	float weightDecay = 0.0002f;
	
	AutoEncoder ae;
	int trained = 0;	//トレーニングの済んだイメージ数
	
	public AutoEncodeTest() {
		ae = new AutoEncoder(inputN, middleN);
	}
	
	void run() {
		float[][] testData = MnistData.readImage(MnistData.TEST_IMAGE, 0, 100);
		new MnistView("input").view(testData);
		
		int size = 10000;	//開発PCのハードウェアスペックの問題で分割
		for(int index = 0; index < 60000; index += size) {
			System.out.printf("training: %5d of 60000\n", index + size);
			train(index, Math.min(size, 60000 - index));
			trained = Math.min(index + size, 60000);
			test();
		}
	}
	
	void train(int index, int n) {
		float[][] trainData = MnistData.readImage(MnistData.TRAIN_IMAGE, index, n);
		ae.train(trainData, epoch, batchSize, learnRate, noiseRate, weightDecay);
	}
	
	void test() {
		float[][] testData = MnistData.readImage(MnistData.TEST_IMAGE, 0, 100);
		float[][] outData = new float[100][];
		for(int i = 0; i < 100; i++) {
			outData[i] = new float[testData[i].length];
			outData[i] = ae.test(testData[i]);
		}
		new MnistView("traind: " + trained).view(outData);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		AutoEncodeTest aetest = new AutoEncodeTest();
		aetest.run();
	}

}
