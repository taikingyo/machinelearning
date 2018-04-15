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
			unit[l] = new float[unitN[l] + 1][1];
			unit[l][unitN[l]][0] = 1;
			weight[l] = new float[unitN[l + 1]][unitN[l] + 1];
			errSignal[l] = new float[unitN[l + 1]][1];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new float[unitN[layerN - 1]][1];
		
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
			unit[l] = new float[unitN[l] + 1][1];
			unit[l][unitN[l]][0] = 1;
			errSignal[l] = new float[unitN[l + 1]][1];
			grad[l] = new float[unitN[l + 1]][unitN[l] + 1];
		}
		unit[layerN - 1] = new float[unitN[layerN - 1]][1];
	}
	
	public static float[][] oneHotVector(int[] label, int classN) {
		float[][] vec = new float[label.length][classN];
		for(int i = 0; i < label.length; i++) {
			Arrays.fill(vec[i], 0);
			vec[i][label[i]] = 1;
		}
		return vec;
	}
	
	public static float[] softmax(float[] x) {
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
	
	public void forward(float[] x) {
		//入力層のセット
		for(int i = 0; i < x.length; i++) unit[0][i][0] = x[i];
		
		//中間層の計算
		for(int l = 0; l < layerN - 2; l++) {
			float[][] post = LinearAlgebra.multi(weight[l], unit[l]);
			for(int i = 0; i < post.length; i++) unit[l + 1][i][0] = activate.apply(post[i][0]);
		}
		
		//出力層の計算
		float[][] out = LinearAlgebra.multi(weight[layerN - 2], unit[layerN - 2]);
		
		//多クラス分類ならsoftmax、二分ならsigmoid関数
		if(unitN[layerN - 1] != 1) {
			float[] normal = softmax(LinearAlgebra.trans(out)[0]);
			for(int i = 0; i < normal.length; i++) unit[layerN - 1][i][0] = normal[i];
		}else unit[layerN - 1][0][0] = Sigmoid.apply(out[0][0]);
	}
	
	private void backPropagate(float[] t) {
		//出力層の誤差計算
		errSignal[layerN - 2] = LinearAlgebra.sub(LinearAlgebra.columnVector(t), unit[layerN - 1]);
		grad[layerN - 2] = LinearAlgebra.add(grad[layerN - 2].clone(), LinearAlgebra.multi(errSignal[layerN - 2], LinearAlgebra.trans(unit[layerN - 2])));

		//中間層の誤差計算
		for(int l = layerN - 2; l > 0; l--) {
			float df[][] = new float[unitN[l]][1];
			for(int i = 0; i < unitN[l]; i++) df[i][0] = dActivate.apply(unit[l][i][0]);
			float[][] tW = LinearAlgebra.trans(removeBias(weight[l]));	//ウェイトからバイアスを除いた行列の転地
			errSignal[l - 1] = LinearAlgebra.hadamard(LinearAlgebra.multi(tW, errSignal[l]), df);
			grad[l - 1] = LinearAlgebra.add(grad[l - 1].clone(), LinearAlgebra.multi(errSignal[l - 1], LinearAlgebra.trans(unit[l - 1])));
		}
	}
	
	private void update(float rate, float weightDecay) {
		for(int l = layerN - 2; l >= 0; l--) {
			float[][] decay = LinearAlgebra.multi(1 - weightDecay, weight[l]);
			float[][] delta = LinearAlgebra.multi(rate, grad[l]);
			weight[l] = LinearAlgebra.add(decay, delta);
		}
	}
	
	private void initGrad() {
		for(int l = 0; l < grad.length; l++) {
			for(int i = 0; i < grad[l].length; i++) Arrays.fill(grad[l][i], 0);
		}
	}
	
	public void train(float[][] data, float[][] teach, int epoch, float learnRate) {
		int pattern = data.length;
		
		for(int e = 0; e < epoch; e++) {
			for(int p = 0; p < pattern; p++) {
				initGrad();
				forward(data[p]);
				backPropagate(teach[p]);
				update(learnRate, 0);
			}
		}
	}
	
	public void train(float[][] data, float[][] teach, int epoch, int batchSize, float learnRate, float weightDecay) {
		int pattern = data.length;
		
		for(int e = 0; e < epoch * pattern / batchSize; e++) {
			initGrad();
			for(int i = 0; i < batchSize; i++) {
				int index = (e * batchSize + i) % pattern;
				forward(data[index]);
				backPropagate(teach[index]);
			}
			update(learnRate, weightDecay);
		}
	}
	
	public void test(float[][] testData, int[] testLabel, PrintWriter pw) {
		int pattern = testData.length;
		int classN = unitN[layerN - 1];
		float[][] testTarget = oneHotVector(testLabel, classN);
		float err = 0;
		int acc = 0;
		
		for(int i = 0; i < pattern; i++) {
			forward(testData[i]);
			float[][] sub = LinearAlgebra.sub(LinearAlgebra.columnVector(testTarget[i]), unit[layerN - 1]);
			err += LinearAlgebra.norm(LinearAlgebra.trans(sub)[0]);
			if(testLabel[i] == getResult()) acc++;
		}
		
		pw.printf("error: %.4f, accuracy: %.4f\n", err / pattern, (float) acc / pattern);
	}
	
	public float[] output() {
		return LinearAlgebra.trans(unit[layerN - 1])[0];
	}
	
	public int getResult() {
		float max = 0;
		int idx = -1;
		float out[] = output();
		for(int i = 0; i < out.length; i++) {
			if(out[i] > max) {
				max = out[i];
				idx = i;
			}
		}
		
		return idx;
	}
}
