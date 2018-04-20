package com.gmail.taikingyo.nn;

public class XORTest {
	static int trainN = 10000;
	static float learnRate = 0.1f;
	
	static float[][] trainData = {
			{0, 0}, {0, 1}, {1, 0}, {1, 1}
	};
	
	static float[][] teachData = {
			{0}, {1}, {1}, {0}
	};
	
	static float[][] testData = {
			{0, 0}, {0, 1}, {1, 0}, {1, 1}
	};

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Perceptron p = new Perceptron(new int[] {2, 2, 1});
		p.train(trainData, teachData, trainN, learnRate);
		
		System.out.println("test");
		p.forward(testData);
		float[][] out = p.output();
		
		for(int i = 0; i < testData.length; i++) {
			for(double d : testData[i]) System.out.printf("%.0f  ", d);
			System.out.printf(" ans %.2f\n", out[i][0]);
		}
	}
}
