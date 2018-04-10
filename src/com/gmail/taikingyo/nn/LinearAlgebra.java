package com.gmail.taikingyo.nn;

import java.util.stream.IntStream;

public class LinearAlgebra {
	
	//行列の加算
	public static float[][] add(float[][] mat1, float[][] mat2) {
		if(mat1.length == mat2.length && mat1[0].length == mat2[0].length) {
			int rows = mat1.length;
			int columns = mat1[0].length;
			float[][] mat = new float[rows][columns];
			
			if(columns == 1) {	//列ベクトルに対しての無駄な並列処理を回避
				for(int r = 0; r < rows; r++) mat[r][0] = mat1[r][0] + mat2[r][0];
			}else IntStream.range(0, rows).parallel().forEach(r -> {
				for(int c = 0; c < columns; c++) {
					mat[r][c] = mat1[r][c] + mat2[r][c];
				}
			});
			
			return mat;
		}else {
			System.err.printf("LinearAlgebra.add: mat1 rows:%d columns:%d mat2 rows:%d columns:%d\n", mat1.length, mat1[0].length, mat2.length, mat2[0].length);
			return null;
		}
	}
	
	//行列の減算
	public static float[][] sub(float[][] mat1, float[][] mat2) {
		if(mat1.length == mat2.length && mat1[0].length == mat2[0].length) {
			int rows = mat1.length;
			int columns = mat1[0].length;
			float[][] mat = new float[rows][columns];

			if(columns == 1) {	//列ベクトルに対しての無駄な並列処理を回避
				for(int r = 0; r < rows; r++) mat[r][0] = mat1[r][0] - mat2[r][0];
			}else IntStream.range(0, rows).parallel().forEach(r -> {
				for(int c = 0; c < columns; c++) {
					mat[r][c] = mat1[r][c] - mat2[r][c];
				}
			});
			
			return mat;
		}else {
			System.err.printf("LinearAlgebra.sub: mat1 rows:%d columns:%d mat2 rows:%d columns:%d\n", mat1.length, mat1[0].length, mat2.length, mat2[0].length);
			return null;
		}
	}
	
	//行列のスカラー倍
	public static float[][] multi(float lamda, float[][] mat) {
		int rows = mat.length;
		int columns = mat[0].length;
		float[][] sMat = new float[rows][columns];

		if(columns == 1) {	//列ベクトルに対しての無駄な並列処理を回避
			for(int r = 0; r < rows; r++) sMat[r][0] = lamda * mat[r][0];
		}else IntStream.range(0, rows).parallel().forEach(r -> {
			for(int c = 0; c < columns; c++) {
				sMat[r][c] = lamda * mat[r][c];
			}
		});
		
		return sMat;
	}
	
	//行列の内積
	public static float[][] multi(float[][] left, float[][] right) {
		if(left[0].length == right.length) {
			int leftRows = left.length;
			int rightColumns = right[0].length;
			int n = left[0].length;
			float[][] mat = new float[leftRows][rightColumns];

			IntStream.range(0, leftRows).parallel().forEach(r -> {
				for(int c = 0; c < rightColumns; c++) {
					float element = 0;
					for(int i = 0; i < n; i++) {
						element += left[r][i] * right[i][c];
					}
					mat[r][c] = element;
				}
			});
			
			return mat;
		}else {
			System.err.printf("LinearAlgebra.multi: left rows:%d columns:%d right rows:%d columns:%d\n", left.length, left[0].length, right.length, right[0].length);
			return null;
		}
	}
	
	//行列のアダマール積
	public static float[][] hadamard(float[][] mat1, float[][] mat2) {
		if(mat1.length == mat2.length && mat1[0].length == mat2[0].length) {
			int rows = mat1.length;
			int columns = mat1[0].length;
			float[][] mat = new float[rows][columns];
			
			if(columns == 1) {	//列ベクトルに対しての無駄な並列処理を回避
				for(int r = 0; r < rows; r++) mat[r][0] = mat1[r][0] * mat2[r][0];
			}else IntStream.range(0, rows).parallel().forEach(r -> {
				for(int c = 0; c < columns; c++) mat[r][c] = mat1[r][c] * mat2[r][c];
			});
			
			return mat;
			
		}else {
			System.err.printf("LinearAlgebra.hadamard: mat1 rows:%d columns:%d mat2 rows:%d columns:%d\n", mat1.length, mat1[0].length, mat2.length, mat2[0].length);
			return null;
		}
	}
	
	//行列の転地
	public static float[][] trans(float[][] mat) {
		int rows = mat[0].length;
		int columns = mat.length;
		float[][] tMat = new float[rows][columns];
		
		if(columns == 1) {	//列ベクトルに対しての無駄な並列処理を回避
			for(int r = 0; r < rows; r++) tMat[r][0] = mat[0][r];
		}else IntStream.range(0, rows).parallel().forEach(r -> {
			for(int c = 0; c < columns; c++) tMat[r][c] = mat[c][r];
		});
		
		return tMat;
	}
	
	//単位行列
	public static float[][] elementMatrix(int n) {
		float[][] mat = new float[n][n];
		for(int i = 0; i < n; i++) mat[i][i] = 1.0f;
		
		return mat;
	}
	
	//ベクトルのノルム
	public static float norm(float[] vec) {
		double s = 0;
		for(float f : vec) s += f * f;
		return (float) Math.sqrt(s);
	}
	
	//ベクトルのユークリッド距離
	public static float euclidean(float[] vec1, float[] vec2) {
		double s = 0;
		for(int i = 0; i < vec1.length; i++) {
			s += Math.pow(vec1[i] - vec2[i], 2);
		}
		
		return (float) Math.sqrt(s);
	}
	
	//1次元配列を2次元行列形式の行ベクトルに変換
	public static float[][] rowVector(float[] vec) {
		float[][] rV = new float[1][];
		rV[0] = vec.clone();
		return rV;
	}
	
	//1次元配列を2次元行列形式の列ベクトルに変換
	public static float[][] columnVector(float[] vec) {
		return trans(rowVector(vec));
	}
	
	public static void printMatrix(float[][] mat) {
		int rows = mat.length;
		int columns = mat[0].length;
		System.out.printf("rows: %2d columns: %2d\n", rows, columns);
		
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < columns; c++) {
				System.out.printf("%6.2f, ", mat[r][c]);
			}
			System.out.println();
		}
	}
	
	public static void size(float[][] mat) {
		int rows = mat.length;
		int columns = mat[0].length;
		System.out.printf("rows: %2d columns: %2d\n", rows, columns);
	}
}
