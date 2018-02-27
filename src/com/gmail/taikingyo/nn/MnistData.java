package com.gmail.taikingyo.nn;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MnistData {
	public static final String TRAIN_IMAGE = "train-images.idx3-ubyte";
	public static final String TRAIN_LABEL = "train-labels.idx1-ubyte";
	public static final String TEST_IMAGE = "t10k-images.idx3-ubyte";
	public static final String TEST_LABEL = "t10k-labels.idx1-ubyte";
	
	private static int readInt(BufferedInputStream bis) {
		byte[] buff = new byte[4];
		try {
			bis.read(buff);
			return ByteBuffer.wrap(buff).getInt();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return -1;
		}
	}
	
	private static void skip(BufferedInputStream bis, long n) {
		try {
			long s = bis.skip(n);
			if(s != n) skip(bis, n - s);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//MNISTデータのstartからn個のイメージを取得
	public static double[][] readImage(String path, int start, int n) {
		BufferedInputStream bis;
		try {
			bis = new BufferedInputStream(new FileInputStream(path));
			readInt(bis);	//magic number
			int num = readInt(bis);
			int height = readInt(bis);
			int width = readInt(bis);
			int size = height * width;
			int len = Math.min(num - start, n);
			double[][] img = new double[len][size];
			byte[][] imgData = new byte[len][size];
			
			skip(bis, start * size);
			for(int i = 0; i < len; i++) bis.read(imgData[i], 0, size);
			bis.close();
			
			for(int i = 0; i < len; i++) {
				for(int j = 0; j < size; j++) img[i][j] = (imgData[i][j] & 0xFF) / 255.0;	//符号をつけ値の範囲を変換
			}
			
			return img;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	//MNISTデータのstartからn個のラベルを取得
	public static int[] readLabel(String path, int start, int n) {
		BufferedInputStream bis;
		try {
			bis = new BufferedInputStream(new FileInputStream(path));
			readInt(bis);	//magic number
			int num = readInt(bis);
			byte[] b = new byte[Math.min(num - start, n)];
			skip(bis, start);
			bis.read(b);
			int[] label = new int[b.length];
			for(int i = 0; i < b.length; i++) label[i] = b[i];
			return label;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}
}
