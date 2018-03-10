package com.gmail.taikingyo.nn;

public class SAETest {
	Perceptron p;
	double[][] trainData;
	double[][] trainLabel;
	int[] struct = {784, 400, 400, 10};

	//並列数、データ数、トレーニング回数は開発PCの性能上、低く抑えています
	int paraN = 4;			//コア数次第で任意
	int trainDataN = 20000;	//できればフルセット60000
	int trainN = 1;			//10回程度？
	
	int batchSize = 10;
	double learnRate = 0.03;
	double noiseRate = 0.3;
	double weightDecay = 0.0002;
	
	public SAETest() {
		trainData = MnistData.readImage(MnistData.TRAIN_IMAGE, 0, trainDataN);
		int[] label = MnistData.readLabel(MnistData.TRAIN_LABEL, 0, trainDataN);
		trainLabel = Perceptron.oneHotVector(label);
		
		preTrain();
		fineTune();
	}
	
	void preTrain() {
		System.out.println("pre training...");
		StackedAutoEncoder sae = new StackedAutoEncoder(struct, paraN);
		sae.preTrain(trainData, trainN, batchSize, learnRate, noiseRate, weightDecay);
		p = new Perceptron(sae.getWeight(), Perceptron.Sigmoid, Perceptron.DSigmoid, paraN);
	}
	
	void fineTune() {
		System.out.println("fine tuning...");
		int size = 10000;
		for(int i = 0; i < 60000; i += size) {
			trainData = MnistData.readImage(MnistData.TRAIN_IMAGE, i, size);
			int[] label = MnistData.readLabel(MnistData.TRAIN_LABEL, i, size);
			trainLabel = Perceptron.oneHotVector(label);
			p.train(trainData, trainLabel, trainN, learnRate);
			System.out.printf("tune %5d / 60000\n", i + size);
			test();
		}
	}
	
	void test() {
		double[][] testData = MnistData.readImage(MnistData.TEST_IMAGE, 0, 10000);
		int[] testLabel = MnistData.readLabel(MnistData.TEST_LABEL, 0, 10000);
		
		int a = 0;
		for(int i = 0; i < testData.length; i++) {
			p.forward(testData[i]);
			if(testLabel[i] == p.getResult()) a++;
		}
		System.out.printf("accuracy: %4f\n", (double)a / 10000);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		new SAETest();
	}
}