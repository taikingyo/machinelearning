package com.gmail.taikingyo.nn;

import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.transform.Affine;
import javafx.stage.Stage;

public class MnistView {
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	private static final int ROW = 10;
	private static final int COLUMM = 10;
	private float[][] images;
	private GraphicsContext g;
	
	public MnistView(String title) {
		new javafx.embed.swing.JFXPanel();
		javafx.application.Platform.runLater(new Runnable() {

			@Override
			public void run() {
				// TODO Auto-generated method stub
				while(images == null) {
					try {
						Thread.sleep(1);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				Stage stage = new Stage();
				Canvas canvas = new Canvas(2 * ((WIDTH + 1) * COLUMM + 1), 2 * ((HEIGHT + 1) * ROW + 1));
				StackPane pane = new StackPane();
				pane.getChildren().add(canvas);
				stage.setScene(new Scene(pane));
				stage.setTitle(title);
				stage.show();
				g = canvas.getGraphicsContext2D();
				g.setTransform(new Affine(2, 0, 0, 0, 2, 0));
			}
			
		});
	}
	
	private void draw() {
		Platform.runLater(new Runnable() {

			@Override
			public void run() {
				// TODO Auto-generated method stub
				for(int i = 0; i < ROW; i++) {
					for(int j = 0; j < COLUMM; j++) {
						int index = i * COLUMM + j;
						if(images.length > index) g.drawImage(float2image(images[index]), j * (WIDTH + 1) + 1, i * (HEIGHT + 1) + 1);
					}
				}
			}
			
		});
	}
	
	private Image float2image(float[] data) {
		WritableImage image = new WritableImage(WIDTH, HEIGHT);
		PixelWriter writer = image.getPixelWriter();
		for(int i = 0; i < HEIGHT; i++) {
			for(int j = 0; j < WIDTH; j++) writer.setColor(j, i, Color.gray(data[i * WIDTH + j]));
		}
		return image;
	}
	
	//MNISTイメージを100件まで表示
	public void view(float[][] images) {
		this.images = images;
		draw();
	}

}
