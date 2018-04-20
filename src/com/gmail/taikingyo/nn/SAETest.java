package com.gmail.taikingyo.nn;

import java.io.PrintWriter;

public class SAETest {
	Perceptron p;
	float[][] trainData;
	float[][] trainLabel;
	int[] struct = {784, 400, 400, 10};

	//並列数、データ数、トレーニング回数は開発PCの性能上、低く抑えています
	int trainDataN = 20000;	//できればフルセット60000
	int epoch = 1;			//10回程度？
	
	int batchSize = 10;
	float learnRate = 0.03f;
	float noiseRate = 0.3f;
	float weightDecay = 0.0002f;
	
	public SAETest() {
		trainData = MnistData.readImage(MnistData.TRAIN_IMAGE, 0, trainDataN);
		int[] label = MnistData.readLabel(MnistData.TRAIN_LABEL, 0, trainDataN);
		trainLabel = Perceptron.oneHotVector(label, 10);
		
		preTrain();
		fineTune();
	}
	
	void preTrain() {
		System.out.println("pre training...");
		StackedAutoEncoder sae = new StackedAutoEncoder(struct);
		sae.preTrain(trainData, epoch, batchSize, learnRate, noiseRate, weightDecay);
		p = new Perceptron(sae.getWeight(), Perceptron.Sigmoid, Perceptron.DSigmoid);
	}
	
	void fineTune() {
		System.out.println("fine tuning...");
		int size = 10000;
		for(int i = 0; i < 60000; i += size) {
			trainData = MnistData.readImage(MnistData.TRAIN_IMAGE, i, size);
			int[] label = MnistData.readLabel(MnistData.TRAIN_LABEL, i, size);
			trainLabel = Perceptron.oneHotVector(label, 10);
			p.train(trainData, trainLabel, epoch, learnRate);
			System.out.printf("tune %5d / 60000\n", i + size);
			test();
		}
	}
	
	void test() {
		float[][] testData = MnistData.readImage(MnistData.TEST_IMAGE, 0, 10000);
		int[] testLabel = MnistData.readLabel(MnistData.TEST_LABEL, 0, 10000);
		
		PrintWriter pw = new PrintWriter(System.out);
		p.test(testData, testLabel, pw);
		pw.close();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		new SAETest();
	}
}
