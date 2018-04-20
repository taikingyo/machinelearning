package com.gmail.taikingyo.nn;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class Perceptron {
	private int layerN;		//レイヤー数
	private int[] unitN;	//ユニット数
	private float[][][] unit;
	private float[][][] weight;
	private float[][][] errSignal;	//誤差信号
	private float[][][] grad;		//勾配
	
	private Function<Float, Float> activate;
	private Function<Float, Float> dActivate;
	
	//シグモイド関数
	public static final Function<Float, Float> Sigmoid = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			return (float) (1 / (1 + Math.exp(-t)));
		}};
	
	//シグモイド関数の導関数
	public static final Function<Float, Float> DSigmoid = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			return t * (1 - t);
		}};
	
	//ランプ関数
	public static final Function<Float, Float> ReLU = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			return (float) Math.max(t, 0);
		}
	};
	
	//ランプ関数の導関数
	public static final Function<Float, Float> DReLU = new Function<Float, Float>() {

		@Override
		public Float apply(Float t) {
			return (t > 0)? 1.0f : 0;
		}
	};
	
	public Perceptron(int[] unitN) {
		this(unitN, Sigmoid, DSigmoid, 0);
	}
	
	public Perceptron(int[] unitN, Function<Float, Float> activate, Function<Float, Float> dActivate, long seed) {
		this.unitN = unitN;
		layerN = unitN.length;
		this.activate = activate;
		this.dActivate = dActivate;
		
		unit = new float[layerN][][];
		weight = new float[layerN - 1][][];
		
		errSignal = new float[layerN - 1][][];
		grad = new float[layerN - 1][][];
		Random rnd = new Random(seed);
		
		//データ形式の初期化
		//出力層以外はバイアス用ユニット（1固定）追加
		for(int l = 0; l < layerN - 1; l++) {
			weight[l] = new float[unitN[l + 1]][unitN[l] + 1];
			errSignal[l] = new float[unitN[l + 1]][1];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
		
		//ウェイトの初期化
		for(int l = 0; l < layerN - 1; l++) {			//layer
			double sd = Math.pow(unitN[l], -0.5);	//プレニューロンのユニット数から標準偏差を決定
			for(int i = 0; i < unitN[l + 1]; i++) {		//post neuron
				for(int j = 0; j < unitN[l] + 1; j++) {	//pre neuron
					weight[l][i][j] = (float) (rnd.nextGaussian() * sd);	//正規分布乱数
				}
			}
		}
	}
	
	public Perceptron(float[][][] weight, Function<Float, Float> activate, Function<Float, Float> dActivate) {
		this.weight = weight;
		this.activate = activate;
		this.dActivate = dActivate;
		
		layerN = weight.length + 1;
		unitN = new int[layerN];
		unit = new float[layerN][][];
		
		errSignal = new float[layerN - 1][][];
		grad = new float[layerN - 1][][];
		
		for(int l = 0; l < layerN - 1; l++) unitN[l] = weight[l][0].length - 1;
		unitN[layerN - 1] = weight[layerN - 2].length;
		
		for(int l = 0; l < layerN - 1; l++) {
			errSignal[l] = new float[unitN[l + 1]][1];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
	}
	
	public static float[][] oneHotVector(int[] label, int classN) {
		float[][] vec = new float[label.length][classN];
		for(int i = 0; i < label.length; i++) {
			Arrays.fill(vec[i], 0);
			vec[i][label[i]] = 1;
		}
		return vec;
	}
	
	private static float[] softmax(float[] x) {
		float[] y = new float[x.length];
		
		//ReLU使用時、計算中にfloat最大値を超えNaNとなるのを避けるためdouble型を使用
		double[] tmp = new double[x.length];
		double s = 0;
		
		for(int i = 0; i < x.length; i++) {
			tmp[i] = Math.exp(x[i]);
			s += tmp[i];
		}
		
		for(int i = 0; i < x.length; i++) {
			y[i] = (float) (tmp[i] / s);
		}
		
		return y;
	}
	
	//ウェイトからバイアス部分を取り除いたもの
	private static float[][] removeBias(float[][] weight) {
		float[][] f = new float[weight.length][weight[0].length - 1];
		for(int i = 0; i < f.length; i++) System.arraycopy(weight[i], 0, f[i], 0, f[0].length);
		return f;
	}
	
	public void forward(float[][] x) {
		int dataN = x.length;
		
		//入力層のセット
		unit[0] = new float[unitN[0] + 1][dataN];
		float[][] in = LinearAlgebra.trans(x);
		for(int i = 0; i < unitN[0]; i++) System.arraycopy(in[i], 0, unit[0][i], 0, dataN);
		Arrays.fill(unit[0][unitN[0]], 1);
		
		//中間層の計算
		for(int l = 0; l < layerN - 2; l++) {
			unit[l + 1] = new float[unitN[l + 1] + 1][dataN];
			float[][] s = LinearAlgebra.multi(weight[l], unit[l]);
			for(int i = 0; i < unitN[l + 1]; i++) {
				for(int j = 0; j < dataN; j++) {
					unit[l + 1][i][j] = activate.apply(s[i][j]);
				}
			}
			Arrays.fill(unit[l + 1][unitN[l + 1]], 1);
		}
		
		//出力層の計算
		float[][] out = LinearAlgebra.trans(LinearAlgebra.multi(weight[layerN - 2], unit[layerN - 2]));
		float[][] normal = new float[dataN][];
		
		//多クラス分類ならsoftmax、二分ならsigmoid関数
		if(unitN[layerN - 1] != 1) {
			for(int i = 0; i < dataN; i++) normal[i] = softmax(out[i]);
		}else {
			for(int i = 0; i < dataN; i++) {
				normal[i] = new float[1];
				normal[i][0] = Sigmoid.apply(out[i][0]);
			}
		}
		
		unit[layerN - 1] = LinearAlgebra.trans(normal);
	}
	
	private void backPropagate(float[][] t) {
		int dataN = t.length;
		
		//出力層の誤差計算
		errSignal[layerN - 2] = LinearAlgebra.sub(LinearAlgebra.trans(t), unit[layerN - 1]);
		grad[layerN - 2] = LinearAlgebra.multi(errSignal[layerN - 2], LinearAlgebra.trans(unit[layerN - 2]));
		
		//中間層の誤差計算
		for(int l = layerN - 2; l > 0; l--) {
			float[][] df = new float[unitN[l]][dataN];
			for(int i = 0; i < unitN[l]; i++) {
				for(int j = 0; j < dataN; j++) df[i][j] = dActivate.apply(unit[l][i][j]);
			}
			float[][] tW = LinearAlgebra.trans(removeBias(weight[l]));	//ウェイトからバイアスを除いた行列の転地
			errSignal[l - 1] = LinearAlgebra.hadamard(LinearAlgebra.multi(tW, errSignal[l]), df);
			grad[l - 1] = LinearAlgebra.multi(errSignal[l - 1], LinearAlgebra.trans(unit[l - 1]));
		}
	}
	
	private void update(float rate, float weightDecay) {
		for(int l = layerN - 2; l >= 0; l--) {
			float[][] decay = LinearAlgebra.multi(1 - weightDecay, weight[l]);
			float[][] delta = LinearAlgebra.multi(rate, grad[l]);
			weight[l] = LinearAlgebra.add(decay, delta);
		}
	}
	
	private float[][] miniBatch(float[][] data, int start, int batchSize) {
		float[][] miniBatch = new float[batchSize][data[0].length];
		for(int i = 0; i < batchSize; i++) {
			int index = (start + i) % data.length;
			System.arraycopy(data[index], 0, miniBatch[i], 0, data[0].length);
		}
		
		return miniBatch;
	}
	
	public void train(float[][] data, float[][] teach, int epoch, float learnRate) {
		train(data, teach, epoch, 1, learnRate, 0);
	}
	
	public void train(float[][] data, float[][] teach, int epoch, int batchSize, float learnRate, float weightDecay) {
		int pattern = data.length;
		
		for(int e = 0; e < (epoch * pattern) / batchSize; e++) {
			int index = (e * batchSize) % pattern;
			float[][] batchData = miniBatch(data, index, batchSize);
			float[][] batchTeach = miniBatch(teach, index, batchSize);
			forward(batchData);
			backPropagate(batchTeach);
			update(learnRate, weightDecay);
		}
	}
	
	public void test(float[][] testData, int[] testLabel, PrintWriter pw) {
		int batchSize = 500;	//メモリ容量次第
		int pattern = testData.length;
		float[][] target = oneHotVector(testLabel, unitN[layerN - 1]);
		float err = 0;
		int acc = 0;
		
		for(int i = 0; i < pattern; i += batchSize) {
			int size = Math.min(batchSize, pattern - i);
			float[][] batchData = miniBatch(testData, i, size);
			float[][] batchTarget = miniBatch(target, i, size);
			
			forward(batchData);
			float[][] e = LinearAlgebra.trans(LinearAlgebra.sub(LinearAlgebra.trans(batchTarget), unit[layerN - 1]));
			for(float[] f : e) err += LinearAlgebra.norm(f);
			int[] res = getResult();
			for(int j = 0; j < size; j++) if(res[j] == testLabel[i + j]) acc++;
		}

		pw.printf("error: %.4f, accuracy: %.4f\n", err / pattern, (float) acc / pattern);
	}
	
	public float[][] output() {
		return LinearAlgebra.trans(unit[layerN - 1]);
	}
	
	public int[] getResult() {
		float[][] out = output();
		int dataN = out.length;
		int[] results = new int[dataN];
		
		for(int d = 0; d < dataN; d++) {
			float max = 0;
			int index = -1;
			
			for(int i = 0; i < out[0].length; i++) {
				if(out[d][i] > max) {
					max = out[d][i];
					index = i;
				}
			}
			
			results[d] = index;
		}
		
		return results;
	}
}
