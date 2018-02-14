package com.gmail.taikingyo.nn;

public class Perceptron {
	private int layerN;		//レイヤー数
	private int[] unitN;	//ユニット数
	private double[][] unit;
	private double[][][] weight;
	private double[][] delta;
	private double err;
	
	public Perceptron(int[] unitN) {
		this.unitN = unitN;
		layerN = unitN.length;
		
		unit = new double[layerN][];
		delta = new double[layerN - 1][];	//中間層＋出力層分
		weight = new double[layerN - 1][][];
		
		//データ形式の初期化
		//出力層以外はバイアス用ユニット（1固定）追加
		for(int l = 0; l < layerN - 1; l++) {
			unit[l] = new double[unitN[l] + 1];
			unit[l][unitN[l]] = 1.0;
			delta[l] = new double[unitN[l + 1]];
			weight[l] = new double[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new double[unitN[layerN - 1]];
		
		//ウェイトの初期化
		for(int l = 0; l < layerN - 1; l++) {			//layer
			for(int i = 0; i < unitN[l + 1]; i++) {		//post neuron
				for(int j = 0; j < unitN[l] + 1; j++) {	//pre neuron
					weight[l][i][j] = Math.random() * 2 - 1;	//-1.0~1.0の一様分布
				}
			}
		}
	}
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	
	public static double[] softmax(double[] x) {
		double[] y = new double[x.length];
		double s = 0.0;
		
		for(int i = 0; i < x.length; i++) {
			y[i] = Math.exp(x[i]);
			s += y[i];
		}
		
		for(int i = 0; i < x.length; i++) {
			y[i] /= s;
		}
		
		return y;
	}
	
	public void forward(double[] x) {
		//入力層のセット
		System.arraycopy(x, 0, unit[0], 0, x.length);
		
		//中間層の計算
		for(int l = 0; l < layerN - 2; l++) {			//layer
			for(int i = 0; i < unitN[l + 1]; i++) {		//post neuron
				double s = 0.0;
				
				for(int j = 0; j < unitN[l] + 1; j++) {	//pre neuron
					s += weight[l][i][j] * unit[l][j];
				}
				unit[l + 1][i] = sigmoid(s);
			}
		}
		
		//出力層の計算
		double[] out = new double[unitN[layerN - 1]];
		for(int i = 0; i < unitN[layerN - 1]; i++) {			//post neuron
			out[i] = 0.0;
			for(int j = 0; j < unitN[layerN - 2] + 1; j++) {	//pre neuron
				out[i] += weight[layerN - 2][i][j] * unit[layerN - 2][j];
			}
		}
		//多クラス分類ならsoftmax、二分ならsigmoid関数
		if(unitN[layerN - 1] != 1) unit[layerN - 1] = softmax(out);
		else unit[layerN - 1][0] = sigmoid(out[0]);
	}
	
	private void backPropagate(double[] t) {
		//出力層の誤差計算
		for(int i = 0; i < unitN[layerN - 1]; i++) {
			double e = t[i] - unit[layerN - 1][i];
			delta[layerN - 2][i] = e;
			err += e * e;
		}
		
		//中間層の誤差計算
		for(int l = layerN - 2; l > 0; l--) {			//layer
			for(int i = 0; i < unitN[l]; i++) {			//pre neuron
				double df = unit[l][i] * (1.0 - unit[l][i]);
				double s = 0.0;
				for(int j = 0; j < unitN[l + 1]; j++) {	//post neuron
					s += delta[l][j] * weight[l][j][i];
				}
				delta[l - 1][i] = df * s;
			}
		}
	}
	
	private void update(double rate) {
		for(int l = layerN - 2; l >= 0; l--) {		//layer
			for(int i = 0; i < unitN[l + 1]; i++) {	//post neuron
				for(int j = 0; j < unitN[l] + 1; j++) {	//pre neuron
					weight[l][i][j] += rate * delta[l][i] * unit[l][j];	//入力層分の有無でdeltaとunitのカウンタにズレ
				}
			}
		}
	}
	
	public void train(double[][] data, double[] [] teach, int trainN, double learnRate) {
		for(int t = 0; t < trainN; t++) {
			err = 0.0;
			for(int p = 0; p < data.length; p++) {
				forward(data[p]);
				backPropagate(teach[p]);
				update(learnRate);
			}
			
			if(t % 1000 == 0) {
				System.out.printf("train: %5d err: %f\n", t, err / data.length / unitN[layerN - 1]);
			}
		}
	}
	
	public double[] output() {
		double[] out = new double[unitN[layerN - 1]];
		System.arraycopy(unit[layerN - 1], 0, out, 0, unitN[layerN - 1]);
		
		return out;
	}
	
	public int getResult() {
		double max = 0;
		int idx = -1;
		double out[] = unit[layerN - 1];
		for(int i = 0; i < out.length; i++) {
			if(out[i] > max) {
				max = out[i];
				idx = i;
			}
		}
		
		return idx;
	}
}
