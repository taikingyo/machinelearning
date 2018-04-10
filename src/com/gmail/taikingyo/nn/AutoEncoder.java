package com.gmail.taikingyo.nn;

import java.util.Arrays;

public class AutoEncoder {
	private int n,m;			//入力層・中間層ユニット数
	private float[][] x;		//入力層
	private float[][] nx;		//ノイズ付加入力層
	private float[][] y;		//中間層
	private float[][] z;		//出力層
	private float[][] weight;
	private float[] bias1;
	private float[] bias2;
	private float[][] wDelta;
	private float[][] b1Delta;
	private float[][] b2Delta;
	
	public AutoEncoder(int n, int m) {
		this.n = n;
		this.m = m;
		
		x = new float[n][1];
		nx = new float[n][1];
		y = new float[m][1];
		z = new float[n][1];
		
		weight = new float[m][n];
		for(int j = 0; j < m; j++) {
			for(int i = 0; i < n; i++) {
				weight[j][i] = (float) (Math.random() * 0.02 - 0.01);
			}
		}
		
		bias1 = new float[m];
		bias2 = new float[n];
		Arrays.fill(bias1, 0);
		Arrays.fill(bias2, 0);
		
		wDelta = new float[m][n];
		b1Delta = new float[m][1];
		b2Delta = new float[n][1];
	}
	
	private float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(-x)));
	}
	
	private void addNoise(float noiseRate) {
		for(int i = 0; i < n; i++) nx[i][0] = Math.random() < noiseRate? 0 : x[i][0];
	}
	
	private void encode() {
		float[][] multi = LinearAlgebra.multi(weight, nx);
		for(int j = 0; j < m; j++) y[j][0] = sigmoid(multi[j][0] + bias1[j]);
	}
	
	private void decode() {
		float[][] tY = LinearAlgebra.trans(y);
		float[][] multi = LinearAlgebra.multi(tY, weight);
		for(int i = 0; i < n; i++) z[i][0] = sigmoid(multi[0][i] + bias2[i]);
	}
	
	private void reconstruct() {
		encode();
		decode();
	}
	
	private void initDelta() {
		b1Delta = new float[m][1];
		b2Delta = new float[n][1];
		for(int j = 0; j < m; j++) Arrays.fill(wDelta[j], 0);
	}
	
	private void accumDelta() {
		float[][] eZ = LinearAlgebra.sub(x, z);	//出力層の誤差
		b2Delta = LinearAlgebra.add(b2Delta.clone(), eZ);
		
		float[][] eY = new float[m][1];	//中間層の誤差
		float[][] s = LinearAlgebra.multi(weight, eZ);
		for(int j = 0; j < m; j++) eY[j][0] = s[j][0] * y[j][0] * (1 - y[j][0]);
		b1Delta = LinearAlgebra.add(b1Delta.clone(), eY);
		
		float[][] mat1 = LinearAlgebra.multi(eY, LinearAlgebra.trans(nx));
		float[][] mat2 = LinearAlgebra.multi(eZ, LinearAlgebra.trans(y));
		float[][] mat3 = LinearAlgebra.add(mat1, LinearAlgebra.trans(mat2));
		wDelta = LinearAlgebra.add(wDelta.clone(), mat3);
	}
	
	public void train(float[][] data, int epoch, int batchSize, float learnRate, float noiseRate, float weightDecay) {
		int pattern = data.length;
		
		for(int t = 0; t < epoch * pattern / batchSize; t++) {
			initDelta();
			
			for(int i = 0; i < batchSize; i++) {
				int index = (t * batchSize + i) % pattern;
				for(int j = 0; j < n; j++) x[j][0] = data[index][j];
				addNoise(noiseRate);
				reconstruct();
				accumDelta();
			}
			
			for(int i = 0; i < n; i++) bias2[i] += b2Delta[i][0] * learnRate;

			for(int j = 0; j < m; j++) {
				bias1[j] += b1Delta[j][0] * learnRate;
				for(int i = 0; i < n; i++) weight[j][i] += wDelta[j][i] * learnRate - weight[j][i] * weightDecay;
			}
		}
	}
	
	public float[] test(float[] data) {
		float[] out = new float[n];
		x = LinearAlgebra.columnVector(data);
		nx = x.clone();
		reconstruct();
		for(int i = 0; i < n; i++) out[i] = z[i][0];
		return out;
	}
	
	public float[] encode(float[] data) {
		float[] out = new float[m];
		x = LinearAlgebra.columnVector(data);
		nx = x.clone();
		encode();
		for(int i = 0; i < m; i++) out[i] = y[i][0];
		return out;
	}
	
	public float[][] getWeight() {
		float[][] param = new float[m][n + 1];
		for(int j = 0; j < m; j++) {
			System.arraycopy(weight[j], 0, param[j], 0, n);
			param[j][n] = bias1[j];
		}
		
		return param;
	}
}
