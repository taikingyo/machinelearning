package com.gmail.taikingyo.nn;

import java.util.Arrays;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Perceptron {
	private int layerN;		//レイヤー数
	private int[] unitN;	//ユニット数
	private float[][] unit;
	private float[][][] weight;
	private float[][] errSignal;	//誤差信号
	private float[][][] grad;		//勾配
	
	private int paraN;		//並列数
	
	private Function<Float, Float> activate;
	private Function<Float, Float> dActivate;
	
	//シグモイド関数
	public static final Function<Float, Float> Sigmoid = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			// TODO Auto-generated method stub
			return (float) (1 / (1 + Math.exp(-t)));
		}};
	
	//シグモイド関数の導関数
	public static final Function<Float, Float> DSigmoid = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			// TODO Auto-generated method stub
			return t * (1 - t);
		}};
	
	//ランプ関数
	public static final Function<Float, Float> ReLU = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			// TODO Auto-generated method stub
			return Math.max(t, 0);
		}
	};
	
	//ランプ関数の導関数
	public static final Function<Float, Float> DReLU = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			// TODO Auto-generated method stub
			return (t > 0)? 1.0f : 0;
		}
	};
	
	public Perceptron(int[] unitN) {
		this(unitN, Sigmoid, DSigmoid, 4);
	}
	
	public Perceptron(int[] unitN, Function<Float, Float> activate, Function<Float, Float> dActivate, int paraN) {
		this.unitN = unitN;
		layerN = unitN.length;
		this.activate = activate;
		this.dActivate = dActivate;
		this.paraN = paraN;
		
		unit = new float[layerN][];
		weight = new float[layerN - 1][][];
		
		errSignal = new float[layerN - 1][];
		grad = new float[layerN - 1][][];
		
		//データ形式の初期化
		//出力層以外はバイアス用ユニット（1固定）追加
		for(int l = 0; l < layerN - 1; l++) {
			unit[l] = new float[unitN[l] + 1];
			unit[l][unitN[l]] = 1.0f;
			weight[l] = new float[unitN[l + 1]][unitN[l] + 1];
			errSignal[l] = new float[unitN[l + 1]];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new float[unitN[layerN - 1]];
		
		//ウェイトの初期化
		for(int l = 0; l < layerN - 1; l++) {			//layer
			for(int i = 0; i < unitN[l + 1]; i++) {		//post neuron
				for(int j = 0; j < unitN[l] + 1; j++) {	//pre neuron
					weight[l][i][j] = (float) (Math.random() * 2 - 1);	//-1.0~1.0の一様分布
				}
			}
		}
	}
	
	public Perceptron(float[][][] weight, Function<Float, Float> activate, Function<Float, Float> dActivate, int paraN) {
		this.weight = weight;
		this.activate = activate;
		this.dActivate = dActivate;
		this.paraN = paraN;
		
		layerN = weight.length + 1;
		unitN = new int[layerN];
		unit = new float[layerN][];
		
		errSignal = new float[layerN - 1][];
		grad = new float[layerN - 1][][];
		
		for(int l = 0; l < layerN - 1; l++) unitN[l] = weight[l][0].length - 1;
		unitN[layerN - 1] = weight[layerN - 2].length;
		
		for(int l = 0; l < layerN - 1; l++) {
			unit[l] = new float[unitN[l] + 1];
			unit[l][unitN[l]] = 1.0f;
			errSignal[l] = new float[unitN[l + 1]];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new float[unitN[layerN - 1]];
	}
	
	public static float[][] oneHotVector(int[] data) {
		float[][] vec = new float[data.length][10];
		for(int i = 0; i < data.length; i++) {
			Arrays.fill(vec[i], 0);
			vec[i][data[i]] = 1.0f;
		}
		return vec;
	}
	
	public static float[] softmax(float[] x) {
		float[] y = new float[x.length];
		float s = 0;
		
		for(int i = 0; i < x.length; i++) {
			y[i] = (float) Math.exp(x[i]);
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
	
	public void forward(float[] x) {
		//入力層のセット
		System.arraycopy(x, 0, unit[0], 0, x.length);
		
		//中間層の計算
		for(int l = 0; l < layerN - 2; l++) {
			final int layer = l;
			loop(unitN[l + 1], i -> {
				float s = 0;
				for(int j = 0; j < unitN[layer] + 1; j++) s += weight[layer][i][j] * unit[layer][j];
				unit[layer + 1][i] = activate.apply(s);
			});
		}
		
		//出力層の計算
		float[] out = new float[unitN[layerN - 1]];
		loop(unitN[layerN - 1], i -> {
			out[i] = 0.0f;
			for(int j = 0; j < unitN[layerN - 2] + 1; j++) {
				out[i] += weight[layerN - 2][i][j] * unit[layerN - 2][j];
			}
		});
		
		//多クラス分類ならsoftmax、二分ならsigmoid関数
		if(unitN[layerN - 1] != 1) unit[layerN - 1] = softmax(out);
		else unit[layerN - 1][0] = Sigmoid.apply(out[0]);
	}
	
	private void backPropagate(float[] t) {
		//出力層の誤差計算
		for(int i = 0; i < unitN[layerN - 1]; i++) {
			float e = t[i] - unit[layerN - 1][i];
			errSignal[layerN - 2][i] = e;
			for(int j = 0; j < unitN[layerN - 2] + 1; j++) grad[layerN - 2][i][j] += errSignal[layerN - 2][i] * unit[layerN - 2][j];
		}
		
		//中間層の誤差計算
		for(int l = layerN - 2; l > 0; l--) {
			final int layer = l;
			loop(unitN[l], i -> {
				float df = dActivate.apply(unit[layer][i]);
				float s = 0;
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
		for(float[][] ff : grad) {
			for(float[] f : ff) Arrays.fill(f, 0);
		}
	}
	
	public void train(float[][] data, float[][] teach, int epoch, float learnRate) {
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
	
	public void train(float[][] data, float[][] teach, int epoch, int batchSize, float learnRate) {
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
	
	public float[] output() {
		float[] out = new float[unitN[layerN - 1]];
		System.arraycopy(unit[layerN - 1], 0, out, 0, unitN[layerN - 1]);
		
		return out;
	}
	
	public int getResult() {
		float max = 0;
		int idx = -1;
		float out[] = unit[layerN - 1];
		for(int i = 0; i < out.length; i++) {
			if(out[i] > max) {
				max = out[i];
				idx = i;
			}
		}
		
		return idx;
	}
}
