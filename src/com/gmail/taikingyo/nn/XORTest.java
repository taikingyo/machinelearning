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
		for(float[] in : testData) {
			for(double d : in) System.out.printf("%.0f  ", d);
			p.forward(in);
			float[] out = p.output();
			System.out.printf(" ans %.2f\n", out[0]);
		}
	}
}
