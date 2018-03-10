package com.gmail.taikingyo.nn;

import java.util.Arrays;
import java.util.function.DoubleFunction;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Perceptron {
	private int layerN;		//レイヤー数
	private int[] unitN;	//ユニット数
	private double[][] unit;
	private double[][][] weight;
	private double[][] errSignal;	//誤差信号
	private double[][][] grad;		//勾配
	
	private int paraN;		//並列数
	
	private DoubleFunction<Double> activate;
	private DoubleFunction<Double> dActivate;
	
	//シグモイド関数
	public static final DoubleFunction<Double> Sigmoid = new DoubleFunction<Double>() {

		@Override
		public Double apply(double value) {
			// TODO Auto-generated method stub
			return 1 / (1 + Math.exp(-value));
		}};
	
	//シグモイド関数の導関数
	public static final DoubleFunction<Double> DSigmoid = new DoubleFunction<Double>() {

		@Override
		public Double apply(double value) {
			// TODO Auto-generated method stub
			return value * (1 - value);
		}};
	
	//ランプ関数
	public static final DoubleFunction<Double> ReLU = new DoubleFunction<Double>() {

		@Override
		public Double apply(double value) {
			// TODO Auto-generated method stub
			return Math.max(value, 0);
		}
	};
	
	//ランプ関数の導関数
	public static final DoubleFunction<Double> DReLU = new DoubleFunction<Double>() {

		@Override
		public Double apply(double value) {
			// TODO Auto-generated method stub
			return (value > 0)? 1.0 : 0;
		}
	};
	
	public Perceptron(int[] unitN) {
		this(unitN, Sigmoid, DSigmoid, 4);
	}
	
	public Perceptron(int[] unitN, DoubleFunction<Double> activate, DoubleFunction<Double> dActivate, int paraN) {
		this.unitN = unitN;
		layerN = unitN.length;
		this.activate = activate;
		this.dActivate = dActivate;
		this.paraN = paraN;
		
		unit = new double[layerN][];
		weight = new double[layerN - 1][][];
		
		errSignal = new double[layerN - 1][];
		grad = new double[layerN - 1][][];
		
		//データ形式の初期化
		//出力層以外はバイアス用ユニット（1固定）追加
		for(int l = 0; l < layerN - 1; l++) {
			unit[l] = new double[unitN[l] + 1];
			unit[l][unitN[l]] = 1.0;
			weight[l] = new double[unitN[l + 1]][unitN[l] + 1];
			errSignal[l] = new double[unitN[l + 1]];
			grad[l] = new double[unitN[l + 1]][unitN[l] + 1];
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
	
	public Perceptron(double[][][] weight, DoubleFunction<Double> activate, DoubleFunction<Double> dActivate, int paraN) {
		this.weight = weight;
		this.activate = activate;
		this.dActivate = dActivate;
		this.paraN = paraN;
		
		layerN = weight.length + 1;
		unitN = new int[layerN];
		unit = new double[layerN][];
		
		errSignal = new double[layerN - 1][];
		grad = new double[layerN - 1][][];
		
		for(int l = 0; l < layerN - 1; l++) unitN[l] = weight[l][0].length - 1;
		unitN[layerN - 1] = weight[layerN - 2].length;
		
		for(int l = 0; l < layerN - 1; l++) {
			unit[l] = new double[unitN[l] + 1];
			unit[l][unitN[l]] = 1.0;
			errSignal[l] = new double[unitN[l + 1]];
			grad[l] = new double[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new double[unitN[layerN - 1]];
	}
	
	public static double[][] oneHotVector(int[] data) {
		double[][] vec = new double[data.length][10];
		for(int i = 0; i < data.length; i++) {
			Arrays.fill(vec[i], 0.0);
			vec[i][data[i]] = 1.0;
		}
		return vec;
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
	
	private void loop(int n, IntConsumer body) {
		int unitLength = (int) Math.ceil((double)n / paraN);
		IntStream.range(0, paraN).parallel().forEach(i -> {
			int start = i * unitLength;
			int end = Math.min((i + 1) * unitLength, n);
			for(int j = start; j < end; j++) body.accept(j);
		});
	}
	
	public void forward(double[] x) {
		//入力層のセット
		System.arraycopy(x, 0, unit[0], 0, x.length);
		
		//中間層の計算
		for(int l = 0; l < layerN - 2; l++) {
			final int layer = l;
			loop(unitN[l + 1], i -> {
				double s = 0.0;
				for(int j = 0; j < unitN[layer] + 1; j++) s += weight[layer][i][j] * unit[layer][j];
				unit[layer + 1][i] = activate.apply(s);
			});
		}
		
		//出力層の計算
		double[] out = new double[unitN[layerN - 1]];
		loop(unitN[layerN - 1], i -> {
			out[i] = 0.0;
			for(int j = 0; j < unitN[layerN - 2] + 1; j++) {
				out[i] += weight[layerN - 2][i][j] * unit[layerN - 2][j];
			}
		});
		
		//多クラス分類ならsoftmax、二分ならsigmoid関数
		if(unitN[layerN - 1] != 1) unit[layerN - 1] = softmax(out);
		else unit[layerN - 1][0] = Sigmoid.apply(out[0]);
	}
	
	private void backPropagate(double[] t) {
		//出力層の誤差計算
		for(int i = 0; i < unitN[layerN - 1]; i++) {
			double e = t[i] - unit[layerN - 1][i];
			errSignal[layerN - 2][i] = e;
			for(int j = 0; j < unitN[layerN - 2] + 1; j++) grad[layerN - 2][i][j] += errSignal[layerN - 2][i] * unit[layerN - 2][j];
		}
		
		//中間層の誤差計算
		for(int l = layerN - 2; l > 0; l--) {
			final int layer = l;
			loop(unitN[l], i -> {
				double df = dActivate.apply(unit[layer][i]);
				double s = 0.0;
				for(int j = 0; j < unitN[layer + 1]; j++) s += errSignal[layer][j] * weight[layer][j][i];
				errSignal[layer - 1][i] = df * s;
				for(int j = 0; j < unitN[layer - 1] + 1; j++) grad[layer - 1][i][j] += errSignal[layer - 1][i] * unit[layer - 1][j];
			});
		}
	}
	
	private void update(double rate) {
		for(int l = layerN - 2; l >= 0; l--) {
			final int layer = l;
			loop(unitN[l + 1], i -> {
				for(int j = 0; j < unitN[layer] + 1; j++) weight[layer][i][j] += rate * grad[layer][i][j];
			});
		}
	}
	
	private void initGrad() {
		for(double[][] dd : grad) {
			for(double[] d : dd) Arrays.fill(d, 0);
		}
	}
	
	public void train(double[][] data, double[][] teach, int epoch, double learnRate) {
		int pattern = data.length;
		
		for(int e = 0; e < epoch; e++) {
			for(int p = 0; p < pattern; p++) {
				initGrad();
				forward(data[p]);
				backPropagate(teach[p]);
				update(learnRate);
			}
		}
	}
	
	public void train(double[][] data, double[][] teach, int epoch, int batchSize, double learnRate) {
		int pattern = data.length;
		
		for(int e = 0; e < epoch * pattern / batchSize; e++) {
			initGrad();
			for(int i = 0; i < batchSize; i++) {
				int index = (e * batchSize + i) % pattern;
				forward(data[index]);
				backPropagate(teach[index]);
			}
			update(learnRate);
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
