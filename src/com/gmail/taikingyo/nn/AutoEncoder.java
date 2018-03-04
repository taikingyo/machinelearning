package com.gmail.taikingyo.nn;

import java.util.Arrays;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class AutoEncoder {
	private int n,m;		//入力層・中間層ユニット数
	private double[] x;		//入力層
	private double[] nx;	//ノイズ付加入力層
	private double[] y;		//中間層
	private double[] z;		//出力層
	private double[][] weight;
	private double[] bias1;
	private double[] bias2;
	private double[][] wDelta;
	private double[] b1Delta;
	private double[] b2Delta;
	
	private int paraN;		//並列数
	
	public AutoEncoder(int n, int m) {
		this(n, m, 4);
	}
	
	public AutoEncoder(int n, int m, int paraN) {
		this.n = n;
		this.m = m;
		
		this.paraN = paraN;
		
		x = new double[n];
		nx = new double[n];
		y = new double[m];
		z = new double[n];
		
		weight = new double[m][n];
		for(int j = 0; j < m; j++) {
			for(int i = 0; i < n; i++) {
				weight[j][i] = Math.random() * 0.02 - 0.01;
			}
		}
		
		bias1 = new double[m];
		bias2 = new double[n];
		Arrays.fill(bias1, 0.0);
		Arrays.fill(bias2, 0.0);
		
		wDelta = new double[m][n];
		b1Delta = new double[m];
		b2Delta = new double[n];
	}
	
	private void loop(int n, IntConsumer body) {
		int unitLength = (int) Math.ceil((double)n / paraN);
		IntStream.range(0, paraN).parallel().forEach(i -> {
			int start = i * unitLength;
			int end = Math.min((i + 1) * unitLength, n);
			for(int j = start; j < end; j++) body.accept(j);
		});
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	private void addNoise(double noiseRate) {
		for(int i = 0; i < n; i++) nx[i] = Math.random() < noiseRate? 0 : x[i];
	}
	
	private void encode() {
		loop(m, j -> {
			double s = 0;
			for(int i = 0; i < n; i++) s += weight[j][i] * nx[i];
			y[j] = sigmoid(s + bias1[j]);
		});
	}
	
	private void decode() {
		loop(n, i -> {
			double s = 0;
			for(int j = 0; j < m; j++) s += weight[j][i] * y[j];
			z[i] = sigmoid(s + bias2[i]);
		});
	}
	
	private void reconstruct() {
		encode();
		decode();
	}
	
	private void initDelta() {
		Arrays.fill(b1Delta, 0.0);
		Arrays.fill(b2Delta, 0.0);		
		for(int j = 0; j < m; j++) Arrays.fill(wDelta[j], 0.0);
	}
	
	private void accumDelta() {
		double[] eZ = new double[n];	//出力層の誤差
		double[] eY = new double[m];	//中間層の誤差
		for(int i = 0; i < n; i++) {
			eZ[i] = x[i] - z[i];
			b2Delta[i] += eZ[i];
		}

		loop(m, j -> {
			double s = 0;
			for(int i = 0; i < n; i++) s += weight[j][i] * eZ[i];
			eY[j] = s * y[j] * (1 - y[j]);
			b1Delta[j] += eY[j];
			for(int i = 0; i < n; i++) wDelta[j][i] += eY[j] * nx[i] + eZ[i] * y[j];
		});
	}
	
	public void train(double[][] data, int trainN, int batchSize, double learnRate, double noiseRate, double weightDecay) {
		int patN = data.length;
		
		for(int t = 0; t < trainN * patN / batchSize; t++) {
			initDelta();
			
			for(int i = 0; i < batchSize; i++) {
				int index = (t * batchSize + i) % patN;
				System.arraycopy(data[index], 0, x, 0, n);
				addNoise(noiseRate);
				reconstruct();
				accumDelta();
			}
			
			for(int i = 0; i < n; i++) bias2[i] += b2Delta[i] * learnRate;

			for(int j = 0; j < m; j++) {
				bias1[j] += b1Delta[j] * learnRate;
				for(int i = 0; i < n; i++) weight[j][i] += wDelta[j][i] * learnRate - weight[j][i] * weightDecay;
			}
		}
	}
	
	public double[] test(double[] data) {
		double[] out = new double[n];
		System.arraycopy(data, 0, x, 0, n);
		System.arraycopy(data, 0, nx, 0, n);
		reconstruct();
		System.arraycopy(z, 0, out, 0, n);
		return out;
	}
	
	public double[][] getWeight() {
		double[][] param = new double[m][n + 1];
		for(int j = 0; j < m; j++) {
			System.arraycopy(weight[j], 0, param[j], 0, n);
			param[j][n] = bias1[j];
		}
		
		return param;
	}
}
